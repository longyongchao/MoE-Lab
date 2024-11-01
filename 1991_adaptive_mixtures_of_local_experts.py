import torch
import torch.optim as optim
import numpy as np
import random

from local_models import adaptive_mixtures_of_local_experts as Model
from local_datasets import mnist_and_fashion_mnist
from evaluation import test
from local_trainer import adaptive_mixtures_of_local_experts as Trainer

# 设置随机种子以保证实验可重复
def set_seed(seed=42):
    random.seed(seed)  # Python 原生随机数生成器
    np.random.seed(seed)  # Numpy 随机数生成器
    torch.manual_seed(seed)  # PyTorch CPU 随机数生成器
    torch.cuda.manual_seed(seed)  # PyTorch GPU 随机数生成器
    torch.cuda.manual_seed_all(seed)  # 如果有多个 GPU
    torch.backends.cudnn.deterministic = True  # 确保每次结果一致
    torch.backends.cudnn.benchmark = False  # 禁用 cuDNN 的自动优化

# 调用 set_seed 函数，设置随机种子
set_seed(42)
torch.autograd.set_detect_anomaly(True)

batch_size = 64
num_experts = 8
device = torch.device('cuda:0')
num_epochs = 15
lr = 0.001


# 创建数据加载器
train_loader, test_loader = mnist_and_fashion_mnist.get_combined_datasets(batch_size=batch_size)

# 合并 MNIST 和 Fashion-MNIST 的类别名称
combined_classes = mnist_and_fashion_mnist.get_combined_label()

# MoE Model
moe_model = Model.MoE(output_dim=len(combined_classes), num_experts=num_experts).to(device)
optimizer_moe = optim.Adam(moe_model.parameters(), lr=lr)

print("\nTraining MoE Model...")
train_expert_selection = Trainer.train(moe_model, train_loader, optimizer_moe, device, combined_classes, num_epochs=num_epochs)

test(combined_classes=combined_classes, moe_model=moe_model, device=device)
