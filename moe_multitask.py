from datasets import load_dataset
from transformers import AutoTokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 加载IMDB数据集
imdb_dataset = load_dataset("imdb")
# 加载SNLI数据集
snli_dataset = load_dataset("snli")

# 使用transformers的预训练分词器
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# 定义最大序列长度
MAX_SEQ_LEN = 200

# 数据预处理函数
def preprocess_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=MAX_SEQ_LEN)

# 对IMDB数据集进行预处理
imdb_tokenized_datasets = imdb_dataset.map(preprocess_function, batched=True)
imdb_tokenized_datasets = imdb_tokenized_datasets.remove_columns(["text"])
imdb_tokenized_datasets = imdb_tokenized_datasets.rename_column("label", "labels")
imdb_tokenized_datasets.set_format("torch")

# 对SNLI数据集进行预处理
def preprocess_snli_function(examples):
    inputs = tokenizer(examples["premise"], examples["hypothesis"], padding="max_length", truncation=True, max_length=MAX_SEQ_LEN)
    inputs["labels"] = examples["label"]
    return inputs

snli_tokenized_datasets = snli_dataset.map(preprocess_snli_function, batched=True)
snli_tokenized_datasets = snli_tokenized_datasets.filter(lambda x: x["labels"] != -1)  # 过滤掉无效标签
snli_tokenized_datasets = snli_tokenized_datasets.remove_columns(["premise", "hypothesis"])
snli_tokenized_datasets.set_format("torch")

from torch.utils.data import DataLoader

BATCH_SIZE = 64

# 构建训练和测试数据加载器
imdb_train_dataloader = DataLoader(imdb_tokenized_datasets["train"], batch_size=BATCH_SIZE, shuffle=True)
imdb_test_dataloader = DataLoader(imdb_tokenized_datasets["test"], batch_size=BATCH_SIZE, shuffle=False)

snli_train_dataloader = DataLoader(snli_tokenized_datasets["train"], batch_size=BATCH_SIZE, shuffle=True)
snli_test_dataloader = DataLoader(snli_tokenized_datasets["validation"], batch_size=BATCH_SIZE, shuffle=False)


class Expert(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Expert, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, output_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts):
        super(GatingNetwork, self).__init__()
        self.fc = nn.Linear(input_dim, num_experts)
    
    def forward(self, x):
        return F.softmax(self.fc(x), dim=1)


class MoEsMultiTaskModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_experts, num_classes_task1, num_classes_task2):
        super(MoEsMultiTaskModel, self).__init__()
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # 专家网络
        self.experts = nn.ModuleList([Expert(embed_dim, embed_dim) for _ in range(num_experts)])
        
        # 门控网络
        self.gating_network = GatingNetwork(embed_dim, num_experts)
        
        # 两个任务的输出层
        self.task1_output_layer = nn.Linear(embed_dim, num_classes_task1)  # 情感分类（2类）
        self.task2_output_layer = nn.Linear(embed_dim, num_classes_task2)  # 文本蕴含（3类）
    
    def forward(self, x, task_id):
        # 嵌入层
        x = self.embedding(x)
        x = torch.mean(x, dim=1)  # 池化
        
        # 门控网络输出
        gate_outputs = self.gating_network(x)
        
        # 专家输出
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        
        # 根据门控网络的输出加权专家的输出
        combined_output = torch.sum(gate_outputs.unsqueeze(2) * expert_outputs, dim=1)
        
        # 根据任务选择输出层
        if task_id == 1:  # 情感分类任务
            return self.task1_output_layer(combined_output)
        elif task_id == 2:  # 文本蕴含任务
            return self.task2_output_layer(combined_output)


def train(model, dataloaders, optimizer, criterion, device):
    model.train()
    total_loss, total_acc_task1, total_acc_task2 = 0, 0, 0
    
    # 训练两个任务的数据集
    for imdb_batch, snli_batch in zip(dataloaders["imdb"], dataloaders["snli"]):
        # 情感分类任务
        imdb_inputs, imdb_labels = imdb_batch["input_ids"].to(device), imdb_batch["labels"].to(device)
        optimizer.zero_grad()
        imdb_outputs = model(imdb_inputs, task_id=1)
        imdb_loss = criterion(imdb_outputs, imdb_labels)
        imdb_loss.backward()
        optimizer.step()
        
        total_loss += imdb_loss.item()
        total_acc_task1 += (imdb_outputs.argmax(1) == imdb_labels).sum().item()
        
        # 文本蕴含任务
        snli_inputs, snli_labels = snli_batch["input_ids"].to(device), snli_batch["labels"].to(device)
        optimizer.zero_grad()
        snli_outputs = model(snli_inputs, task_id=2)
        snli_loss = criterion(snli_outputs, snli_labels)
        snli_loss.backward()
        optimizer.step()
        
        total_loss += snli_loss.item()
        total_acc_task2 += (snli_outputs.argmax(1) == snli_labels).sum().item()
    
    return total_loss / (len(dataloaders["imdb"]) + len(dataloaders["snli"])), total_acc_task1 / len(dataloaders["imdb"].dataset), total_acc_task2 / len(dataloaders["snli"].dataset)

def evaluate(model, dataloaders, criterion, device):
    model.eval()
    total_loss, total_acc_task1, total_acc_task2 = 0, 0, 0
    
    with torch.no_grad():
        for imdb_batch, snli_batch in zip(dataloaders["imdb"], dataloaders["snli"]):
            # 情感分类任务
            imdb_inputs, imdb_labels = imdb_batch["input_ids"].to(device), imdb_batch["labels"].to(device)
            imdb_outputs = model(imdb_inputs, task_id=1)
            imdb_loss = criterion(imdb_outputs, imdb_labels)
            total_loss += imdb_loss.item()
            total_acc_task1 += (imdb_outputs.argmax(1) == imdb_labels).sum().item()
            
            # 文本蕴含任务
            snli_inputs, snli_labels = snli_batch["input_ids"].to(device), snli_batch["labels"].to(device)
            snli_outputs = model(snli_inputs, task_id=2)
            snli_loss = criterion(snli_outputs, snli_labels)
            total_loss += snli_loss.item()
            total_acc_task2 += (snli_outputs.argmax(1) == snli_labels).sum().item()
    
    return total_loss / (len(dataloaders["imdb"]) + len(dataloaders["snli"])), total_acc_task1 / len(dataloaders["imdb"].dataset), total_acc_task2 / len(dataloaders["snli"].dataset)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab_size = tokenizer.vocab_size
embed_dim = 128  # 嵌入维度

# 初始化模型
model = MoEsMultiTaskModel(vocab_size=vocab_size, embed_dim=embed_dim, num_experts=4, num_classes_task1=2, num_classes_task2=3).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

NUM_EPOCHS = 5

dataloaders = {
    "imdb": imdb_train_dataloader,
    "snli": snli_train_dataloader,
}

for epoch in range(NUM_EPOCHS):
    train_loss, train_acc_task1, train_acc_task2 = train(model, dataloaders, optimizer, criterion, device)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}:")
    print(f"  Train Loss: {train_loss:.4f}, Sentiment Accuracy: {train_acc_task1:.4f}, NLI Accuracy: {train_acc_task2:.4f}")

# 推理函数
def predict(text, model, tokenizer, task_id, device):
    model.eval()
    with torch.no_grad():
        if task_id == 1:
            # 情感分类任务
            inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=MAX_SEQ_LEN).to(device)
            output = model(inputs["input_ids"], task_id=task_id)
            prediction = output.argmax(1).item()
            return "Positive" if prediction == 1 else "Negative"
        elif task_id == 2:
            # 文本蕴含任务
            premise, hypothesis = text
            inputs = tokenizer(premise, hypothesis, return_tensors="pt", padding="max_length", truncation=True, max_length=MAX_SEQ_LEN).to(device)
            output = model(inputs["input_ids"], task_id=task_id)
            prediction = output.argmax(1).item()
            return ["Entailment", "Contradiction", "Neutral"][prediction]

# 示例推理
example_text = "This movie was absolutely fantastic!"
sentiment_prediction = predict(example_text, model, tokenizer, task_id=1, device=device)
print(f"Sentiment Prediction: {sentiment_prediction}")

example_premise = "A man is playing a guitar."
example_hypothesis = "A man is performing on stage."
nli_prediction = predict([example_premise, example_hypothesis], model, tokenizer, task_id=2, device=device)
print(f"NLI Prediction: {nli_prediction}")

