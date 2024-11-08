import torch
import torch.optim as optim

from local_models import models as Model

# 切换数据集
from local_datasets import mnist_family as dataset
# from local_datasets import rgb_mix as dataset

from evaluation import test
from local_trainer import trainer as Trainer
from utils import set_seed, save_model
import time

timestamp = time.strftime("%Y%m%d-%H%M%S")

# 调用 set_seed 函数，设置随机种子
set_seed(19940329)
torch.autograd.set_detect_anomaly(True)

device = torch.device('cuda:0')

input_dim = dataset.get_input_dim()
combined_classes = dataset.get_combined_label()
dataset_split_point = dataset.get_split_point()
datasets_info = dataset.get_datasets_info()

expert_hidden_dim = [] # 专家隐藏层维度，如果需要单线性层，则设置[]
gating_hidden_dim = [50] # 门控隐藏层维度，如果需要单线性层，则设置[]
num_experts = 4
DMoE_margin_threshold = 0.5
SGMoE_w_importance = 0.1

enable_hard_constraint_DMoE = False
enable_sparsely_gated_noise_SGMoE = True
enable_soft_constraint_SGMoE = True
top_k = 2

batch_size = 64
lr = 0.0005

"""
num_epochs = (m, n)是两阶段的训练方式：
    1. m表示第一阶段训练的epoch数，采用learning factored representations in a deep mixture of experts（DMoE）的硬约束方法，放置专家分配的马太效应。
    2. n表示第二阶段训练的epoch数，取消DMoE的硬约束。
如果希望复现1991年Adaptive Mixtures of Local Experts的方法，请设置num_epochs = (0, n)
"""
# num_epochs = (0, 10)
# num_epochs = (2, 8)
# num_epochs = (4, 6)
# num_epochs = (8, 2)
num_epochs = (10, 0)

expert_dim = f"expert784-{expert_hidden_dim}-20_"
gating_dim = f"gating784-{gating_hidden_dim}-{num_experts}_"

method = "He_DMoE_"

if num_epochs[0] == 0:
    method = "He_VanillaMoE_"

method = method + f"{expert_dim}{gating_dim}"

# 创建数据加载器
train_loader, test_loader = dataset.get_combined_datasets(batch_size=batch_size)


# MoE Model
moe_model = Model.MoE(
    input_dim=input_dim,
    output_dim=len(combined_classes), 
    expert_hidden_dim=expert_hidden_dim,
    gating_hidden_dim=gating_hidden_dim,
    num_experts=num_experts, 
    margin_threshold=DMoE_margin_threshold,
    SGMoE_w_importance=SGMoE_w_importance,
    enable_hard_constraint=enable_hard_constraint_DMoE,
    enable_sparsely_gated_noise=enable_sparsely_gated_noise_SGMoE,
    enable_soft_constraint=enable_soft_constraint_SGMoE,
    top_k=top_k
).to(device)


optimizer_moe = optim.Adam(moe_model.parameters(), lr=lr)
torch.nn.utils.clip_grad_norm_(moe_model.parameters(), max_norm=1.0) # 梯度裁剪，防止梯度爆炸

print("\nTraining MoE Model...")
train_expert_selection = Trainer.train(moe_model, train_loader, optimizer_moe, device, combined_classes, num_epochs=num_epochs)


file_name = f"{timestamp}_{method}_epochs{num_epochs}_lr{lr}_margin{DMoE_margin_threshold}"

test(
    combined_classes=combined_classes, 
    moe_model=moe_model, 
    device=device,
    heatmap_file_name=file_name,
    split_point=dataset_split_point,
    datasets_info=datasets_info,
)

save_model(moe_model, optimizer_moe, save_dir="saved_models", model_name=file_name)

# 如果需要加载模型，可以使用以下代码
# load_path = "saved_models/moe_model_20231101-173000.pth"  # 替换为实际的文件路径
# moe_model, optimizer_moe = load_model(moe_model, optimizer_moe, load_path, device)
