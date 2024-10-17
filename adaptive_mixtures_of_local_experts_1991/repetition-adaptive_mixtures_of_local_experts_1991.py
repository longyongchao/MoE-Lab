import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

batch_size = 4096
num_experts = 8

# 定义数据预处理
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# 加载 MNIST 和 Fashion-MNIST 数据集
mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

fashion_mnist_train = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
fashion_mnist_test = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

# 修改 Fashion-MNIST 的标签，使其与 MNIST 标签区分开来（加上 10）
fashion_mnist_train.targets = fashion_mnist_train.targets + 10
fashion_mnist_test.targets = fashion_mnist_test.targets + 10

# 合并训练集和测试集
combined_train_data = ConcatDataset([mnist_train, fashion_mnist_train])
combined_test_data = ConcatDataset([mnist_test, fashion_mnist_test])

# 打印合并后的数据集信息
print(f"训练集样本数: {len(combined_train_data)}")
print(f"测试集样本数: {len(combined_test_data)}")

# 定义 MNIST 和 Fashion-MNIST 的类别名称
mnist_classes = [str(i) for i in range(10)]  # MNIST 类别是 '0' 到 '9'
fashion_mnist_classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                         'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']  # Fashion-MNIST 类别

# 合并 MNIST 和 Fashion-MNIST 的类别名称
combined_classes = mnist_classes + fashion_mnist_classes

# 创建数据加载器
train_loader = DataLoader(combined_train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(combined_test_data, batch_size=batch_size, shuffle=False)

# Define a simple feedforward network (single expert)
class SingleExpert(nn.Module):
    def __init__(self, input_dim=28*28, output_dim=10):
        super(SingleExpert, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define MoE model with two experts
class MoE(nn.Module):
    def __init__(self, input_dim=28*28, output_dim=10, num_experts=2):
        super(MoE, self).__init__()
        self.num_experts = num_experts
        
        # Define experts
        self.experts = nn.ModuleList([SingleExpert(input_dim, output_dim) for _ in range(num_experts)])
        
        # Define gating network (decides which expert to use)
        self.gating_network = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        x_flat = x.view(-1, 28*28)

        # Get gating probabilities
        gate_outputs = torch.softmax(self.gating_network(x_flat), dim=1)  # Shape: [batch_size, num_experts]
        
        # Get outputs from each expert
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)  # Shape: [batch_size, num_experts, output_dim]
        
        # Randomly select an expert based on the gating probabilities
        expert_indices = torch.multinomial(gate_outputs, num_samples=1).squeeze()  # Shape: [batch_size]
        
        # Gather the expert outputs corresponding to the sampled indices
        final_output = expert_outputs[torch.arange(x.size(0)), expert_indices]  # Shape: [batch_size, output_dim]
        
        return final_output, expert_outputs, gate_outputs, expert_indices

# Custom loss function for MoE
def moe_loss(targets, expert_outputs, gate_outputs):
    """
    Compute the loss for the Mixture of Experts model.
    
    Arguments:
    - targets: Ground truth output vectors, shape [batch_size, output_dim]
    - expert_outputs: Output vectors from each expert, shape [batch_size, num_experts, output_dim]
    - gate_outputs: Gating network outputs (probabilities for each expert), shape [batch_size, num_experts]
    
    Returns:
    - loss: Computed loss value
    """
    # Compute the squared error between targets and each expert's output
    errors = torch.sum((expert_outputs - targets.unsqueeze(1))**2, dim=2)  # Shape: [batch_size, num_experts]
    
    # Compute the exponentials of the negative half errors (Gaussian likelihood)
    weighted_errors = torch.exp(-0.5 * errors)  # Shape: [batch_size, num_experts]
    
    # Weight the errors by the gate outputs (probabilities)
    weighted_errors = gate_outputs * weighted_errors  # Shape: [batch_size, num_experts]
    
    # Sum over experts and take the log to get the negative log likelihood
    loss = -torch.log(torch.sum(weighted_errors, dim=1) + 1e-8)  # Shape: [batch_size]
    
    # Return the mean loss over the batch
    return loss.mean()

def one_hot_encoding(labels, num_classes=10):
    # Ensure the identity matrix is created on the same device as labels
    return torch.eye(num_classes, device=labels.device)[labels]

# Training loop with expert selection tracking
def train(model, dataloader, optimizer, num_epochs=10):
    model.train()
    
    # 初始化专家选择计数器
    expert_selection_count = torch.zeros(model.num_experts, device=device)
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        # 使用tqdm显示进度
        for inputs, labels in tqdm(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            
            # Forward pass
            final_output, expert_outputs, gate_outputs, expert_indices = model(inputs)
            
            # 统计专家选择次数
            for idx in expert_indices:
                expert_selection_count[idx] += 1
            
            # Convert labels to one-hot encoding
            one_hot_labels = one_hot_encoding(labels, num_classes=len(combined_classes))
            
            # Compute MoE loss
            loss = moe_loss(one_hot_labels, expert_outputs, gate_outputs)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(final_output, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}, Accuracy: {100 * correct / total:.2f}%')

    # 输出专家选择次数
    print("\n专家选择次数统计（训练集）：")
    for i, count in enumerate(expert_selection_count):
        print(f'专家 {i}: 被选择 {count.item()} 次')

    return expert_selection_count

# Test loop with expert selection tracking
def test(model, dataloader, dataset_name=""):
    model.eval()
    correct = 0
    total = 0
    
    # 初始化专家选择计数器
    expert_selection_count = torch.zeros(model.num_experts, device=device)
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            final_output, _, _, expert_indices = model(inputs)
            
            # 统计专家选择次数
            for idx in expert_indices:
                expert_selection_count[idx] += 1
            
            # Get predictions
            _, predicted = torch.max(final_output, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'\n{dataset_name} 测试集准确率: {accuracy:.2f}%')

    # 输出专家选择次数
    print(f"\n专家选择次数统计（{dataset_name}）：")
    for i, count in enumerate(expert_selection_count):
        print(f'专家 {i}: 被选择 {count.item()} 次')

    return accuracy, expert_selection_count

# # Main experiment
device = torch.device('cuda:1')

# MoE Model
moe_model = MoE(output_dim=len(combined_classes), num_experts=num_experts).to(device)
optimizer_moe = optim.Adam(moe_model.parameters(), lr=0.001)

print("\nTraining MoE Model...")
train_expert_selection = train(moe_model, train_loader, optimizer_moe, num_epochs=10)

print("Testing MoE Model on MNIST and Fashion-MNIST...")
# Test the model
print("Testing on combined MNIST and Fashion-MNIST dataset...")
test_accuracy_combined, combined_expert_selection = test(moe_model, test_loader, dataset_name="MNIST + Fashion-MNIST")

# Test on individual datasets
print("Testing on MNIST dataset...")
mnist_test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=False)
test_accuracy_mnist, mnist_expert_selection = test(moe_model, mnist_test_loader, dataset_name="MNIST")

print("Testing on Fashion-MNIST dataset...")
fashion_mnist_test_loader = DataLoader(fashion_mnist_test, batch_size=batch_size, shuffle=False)
test_accuracy_fashion_mnist, fashion_mnist_expert_selection = test(moe_model, fashion_mnist_test_loader, dataset_name="Fashion-MNIST")

# Show the test results
print(f"Combined Test Accuracy: {test_accuracy_combined:.2f}%")
print(f"MNIST Test Accuracy: {test_accuracy_mnist:.2f}%")
print(f"Fashion-MNIST Test Accuracy: {test_accuracy_fashion_mnist:.2f}%")

# 按选择次数从高到低排序专家选择结果
def print_sorted_expert_selection(expert_selection_count, dataset_name):
    sorted_indices = torch.argsort(expert_selection_count, descending=True)
    print(f"\n{dataset_name} 专家选择次数排序：")
    for idx in sorted_indices:
        print(f'专家 {idx.item()}: 被选择 {expert_selection_count[idx].item()} 次')

# 输出排序后的专家选择情况
print_sorted_expert_selection(train_expert_selection, "训练集")
print_sorted_expert_selection(combined_expert_selection, "MNIST + Fashion-MNIST 测试集")
print_sorted_expert_selection(mnist_expert_selection, "MNIST 测试集")
print_sorted_expert_selection(fashion_mnist_expert_selection, "Fashion-MNIST 测试集")
