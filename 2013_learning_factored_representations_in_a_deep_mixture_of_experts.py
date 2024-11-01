import torch
import torch.optim as optim

from local_models import learning_factored_representations_in_a_deep_mixture_of_experts as Model
from local_datasets import mnist_and_fashion_mnist
from evaluation import test
from local_trainer import adaptive_mixtures_of_local_experts as Trainer

batch_size = 1024
num_experts = 4
device = torch.device('cuda:1')
num_epochs = 10

# 创建数据加载器
train_loader, test_loader = mnist_and_fashion_mnist.get_combined_datasets(batch_size=batch_size)

# 合并 MNIST 和 Fashion-MNIST 的类别名称
combined_classes = mnist_and_fashion_mnist.get_combined_label()

# MoE Model
moe_model = Model.MoE(output_dim=len(combined_classes), num_experts=num_experts).to(device)
optimizer_moe = optim.Adam(moe_model.parameters(), lr=0.001)

print("\nTraining MoE Model...")
train_expert_selection = Trainer.train(moe_model, train_loader, optimizer_moe, device, combined_classes, num_epochs=num_epochs)

test(combined_classes=combined_classes, moe_model=moe_model, device=device)
