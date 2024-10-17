from datasets import load_dataset
from transformers import AutoTokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 加载IMDB数据集
dataset = load_dataset("imdb")

# 使用transformers的预训练分词器
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# 定义最大序列长度
MAX_SEQ_LEN = 200

# 数据预处理函数
def preprocess_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=MAX_SEQ_LEN)

# 对训练集和测试集进行预处理
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# 只保留模型需要的输入
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

from torch.utils.data import DataLoader

BATCH_SIZE = 64

# 构建训练和测试数据加载器
train_dataloader = DataLoader(tokenized_datasets["train"], batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(tokenized_datasets["test"], batch_size=BATCH_SIZE, shuffle=False)


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
        # 输出为专家选择的概率分布
        return F.softmax(self.fc(x), dim=1)


class MoEsModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_experts, num_classes):
        super(MoEsModel, self).__init__()
        # 添加嵌入层，将 input_ids 映射为词向量
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # 专家网络和门控网络保持不变
        self.experts = nn.ModuleList([Expert(embed_dim, num_classes) for _ in range(num_experts)])
        self.gating_network = GatingNetwork(embed_dim, num_experts)
    
    def forward(self, x):
        # 通过嵌入层，将 input_ids 转换为词嵌入
        x = self.embedding(x)  # 输出形状: (batch_size, max_seq_len, embed_dim)
        
        # 对嵌入进行池化操作，例如取平均值，得到固定维度的特征表示
        x = torch.mean(x, dim=1)  # 输出形状: (batch_size, embed_dim)
        
        # 获取门控网络的输出
        gate_outputs = self.gating_network(x)
        
        # 计算每个专家的输出
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        
        # 根据门控网络的输出加权专家的输出
        output = torch.sum(gate_outputs.unsqueeze(2) * expert_outputs, dim=1)
        
        return output


def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss, total_acc = 0, 0
    for batch in dataloader:
        inputs, labels = batch["input_ids"].to(device), batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_acc += (outputs.argmax(1) == labels).sum().item()
    
    return total_loss / len(dataloader), total_acc / len(dataloader.dataset)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss, total_acc = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch["input_ids"].to(device), batch["labels"].to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            total_acc += (outputs.argmax(1) == labels).sum().item()
    
    return total_loss / len(dataloader), total_acc / len(dataloader.dataset)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 获取词汇表大小
vocab_size = tokenizer.vocab_size

# 定义嵌入维度
embed_dim = 128  # 你可以根据需要调整这个值

# 初始化模型
model = MoEsModel(vocab_size=vocab_size, embed_dim=embed_dim, num_experts=4, num_classes=2).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

NUM_EPOCHS = 5

for epoch in range(NUM_EPOCHS):
    train_loss, train_acc = train(model, train_dataloader, optimizer, criterion, device)
    test_loss, test_acc = evaluate(model, test_dataloader, criterion, device)
    
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}:")
    print(f"  Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
    print(f"  Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")


def predict(text, model, tokenizer, device):
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=MAX_SEQ_LEN).to(device)
        output = model(inputs["input_ids"])
        prediction = output.argmax(1).item()
        return "Positive" if prediction == 1 else "Negative"

# 示例推理
example_text = "This movie was absolutely fantastic!"
prediction = predict(example_text, model, tokenizer, device)
print(f"Prediction: {prediction}")
