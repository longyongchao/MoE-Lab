import torch
import torch.optim as optim

from local_models import learning_factored_representations_in_a_deep_mixture_of_experts as Model
from local_datasets import mnist_and_fashion_mnist
from evaluation import test
from local_trainer import learning_factored_representations_in_a_deep_mixture_of_experts as Trainer
from utils import set_seed, save_model
import time


# 调用 set_seed 函数，设置随机种子
set_seed(0)
torch.autograd.set_detect_anomaly(True)

batch_size = 128
num_experts = 4
device = torch.device('cuda')
lr = 0.0005
margin_threshold = 0.01

"""
num_epochs = (m, n)是两阶段的训练方式：
    1. m表示第一阶段训练的epoch数，采用learning factored representations in a deep mixture of experts（DMoE）的硬约束方法，放置专家分配的马太效应。
    2. n表示第二阶段训练的epoch数，取消DMoE的硬约束。
如果希望复现1991年Adaptive Mixtures of Local Experts的方法，请设置num_epochs = (0, n)
"""
# num_epochs = (0, 10)
# num_epochs = (10, 0)
# num_epochs = (2, 8)
num_epochs = (8, 2)

method = "DMoE"

if num_epochs[0] == 0:
    method = "VanillaMoE"


# 创建数据加载器
train_loader, test_loader = mnist_and_fashion_mnist.get_combined_datasets(batch_size=batch_size)

# 合并 MNIST 和 Fashion-MNIST 的类别名称
combined_classes = mnist_and_fashion_mnist.get_combined_label()

# MoE Model
moe_model = Model.MoE(output_dim=len(combined_classes), num_experts=num_experts, margin_threshold=margin_threshold).to(device)
optimizer_moe = optim.Adam(moe_model.parameters(), lr=lr)
torch.nn.utils.clip_grad_norm_(moe_model.parameters(), max_norm=1.0) # 梯度裁剪，防止梯度爆炸

print("\nTraining MoE Model...")
train_expert_selection = Trainer.train(moe_model, train_loader, optimizer_moe, device, combined_classes, num_epochs=num_epochs)

timestamp = time.strftime("%Y%m%d-%H%M%S")

file_name = f"{method}_{timestamp}_{num_experts}_epochs_{num_epochs}_lr_{lr}_margin_{margin_threshold}"

test(
    combined_classes=combined_classes, 
    moe_model=moe_model, 
    device=device,
    heatmap_file_name=file_name,
)

save_model(moe_model, optimizer_moe, save_dir="saved_models", model_name=file_name)

# 如果需要加载模型，可以使用以下代码
# load_path = "saved_models/moe_model_20231101-173000.pth"  # 替换为实际的文件路径
# moe_model, optimizer_moe = load_model(moe_model, optimizer_moe, load_path, device)