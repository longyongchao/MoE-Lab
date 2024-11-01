import torch
import numpy as np
import random
import os


# 设置随机种子以保证实验可重复
def set_seed(seed=42):
    random.seed(seed)  # Python 原生随机数生成器
    np.random.seed(seed)  # Numpy 随机数生成器
    torch.manual_seed(seed)  # PyTorch CPU 随机数生成器
    torch.cuda.manual_seed(seed)  # PyTorch GPU 随机数生成器
    torch.cuda.manual_seed_all(seed)  # 如果有多个 GPU
    torch.backends.cudnn.deterministic = True  # 确保每次结果一致
    torch.backends.cudnn.benchmark = False  # 禁用 cuDNN 的自动优化

# 保存模型函数
def save_model(model, optimizer, save_dir="saved_models", model_name="moe_model"):
    # 创建保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    # 保存模型和优化器的 state_dict
    save_path = os.path.join(save_dir, f"{model_name}.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, save_path)

    print(f"模型已保存到: {save_path}")

# 读取模型函数
def load_model(model, optimizer, load_path, device):
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"模型文件 {load_path} 不存在！")

    checkpoint = torch.load(load_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print(f"模型已从 {load_path} 加载")
    return model, optimizer

