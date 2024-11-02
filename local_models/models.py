import torch
import torch.nn as nn


class SingleExpert(nn.Module):
    def __init__(self, input_dim: int, expert_hidden_dim: list, num_classes: int):  
        super(SingleExpert, self).__init__()

        self.layers = nn.ModuleList()
        if len(expert_hidden_dim) > 0:
            self.layers.append(nn.Linear(input_dim, expert_hidden_dim[0]))
            for i in range(1, len(expert_hidden_dim)):
                self.layers.append(nn.Linear(expert_hidden_dim[i-1], expert_hidden_dim[i]))
            self.layers.append(nn.Linear(expert_hidden_dim[-1], num_classes))
            self.relu = nn.ReLU()
        else:
            self.layers.append(nn.Linear(input_dim, num_classes))

        self._initialize_weights()

    def _initialize_weights(self):
        # He 初始化用于 ReLU 激活函数
        for layer in self.layers:
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        x = self.layers[0](x)
        for layer in self.layers[1:]:
            x = layer(self.relu(x))
        return x


class SparselyGatedNoise(nn.Module):
    def __init__(self, input_dim, num_experts):
        super(SparselyGatedNoise, self).__init__()
        self.num_experts = num_experts

        # noise是一个可学习的参数，用于生成噪声，初始分布是正态
        self.noise_linear = nn.Linear(input_dim, num_experts, bias=False)

        # softplus激活
        self.softplus = nn.Softplus()
    
    def _init_weights(self):
        nn.init.normal_(self.noise_linear.weight, std=0.01)

    def forward(self, x):
        noise = self.softplus(self.noise_linear(x))
        standard_normal = torch.randn(noise.shape, device=noise.device)
        return standard_normal * noise


class GatingNetwork(nn.Module):
    def __init__(
        self, 
        input_dim, 
        gating_hidden_dim, 
        num_experts, 
        sparsely_gated_noise=False
    ):
        super(GatingNetwork, self).__init__()

        self.layers = nn.ModuleList()
        if len(gating_hidden_dim) > 0:
            self.layers.append(nn.Linear(input_dim, gating_hidden_dim[0]))
            for i in range(1, len(gating_hidden_dim)):
                self.layers.append(nn.Linear(gating_hidden_dim[i-1], gating_hidden_dim[i]))
            self.layers.append(nn.Linear(gating_hidden_dim[-1], num_experts))
            self.relu = nn.ReLU()
        else:
            self.layers.append(nn.Linear(input_dim, num_experts))
        
        if sparsely_gated_noise:
            self.noise = SparselyGatedNoise(input_dim, num_experts)
        else:
            self.noise = None

        self.softmax = nn.Softmax(dim=1)

        self._initialize_weights()

    def _initialize_weights(self):
        # He 初始化用于 ReLU 激活函数
        for layer in self.layers:
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            nn.init.zeros_(layer.bias)

    def forward(self, x, training=False, enable_softmax=False):
        sg_noise = self.noise(x) if (self.noise is not None and training) else 0
        x = self.layers[0](x)
        for layer in self.layers[1:]:
            x = layer(self.relu(x))
        return self.softmax(x + sg_noise) if enable_softmax else x + sg_noise


class MoE(nn.Module):
    def __init__(
        self, 
        input_dim, 
        output_dim, 
        expert_hidden_dim=[],
        gating_hidden_dim=[50], 
        num_experts=4, 
        margin_threshold=0.1,
        SGMoE_w_importance=0.1,
        enable_sparsely_gated_noise=False,
        enable_hard_constraint=False,
        enable_soft_constraint=False,
        top_k=1,
    ):
        super(MoE, self).__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.margin_threshold = margin_threshold  # m
        self.SGMoE_w_importance = SGMoE_w_importance  # w
        self.top_k = top_k
        self.enable_hard_constraint = enable_hard_constraint
        self.enable_soft_constraint = enable_soft_constraint
        
        # 定义专家
        self.experts = nn.ModuleList([SingleExpert(input_dim, expert_hidden_dim, output_dim) for _ in range(num_experts)])
        
        # 定义门控网络
        self.gating_network = GatingNetwork(
            input_dim=input_dim, 
            gating_hidden_dim=gating_hidden_dim, 
            num_experts=num_experts, 
            sparsely_gated_noise=enable_sparsely_gated_noise
        )
        
        # 初始化专家选择概率累计计数
        self.expert_selection_count = torch.zeros(num_experts, requires_grad=False)
    

    def forward(
        self, 
        x, 
        training=False, 
        hard_constraint=False
    ):
        self.training = training
        x_flat = x.view(-1, self.input_dim)  # 将图片展平输入
        
        # 获取门控网络的输出概率
        gate_outputs = self.gating_network(x_flat, training=training, enable_softmax=False)

        if training and (self.enable_hard_constraint or self.enable_soft_constraint):
            # 计算每个专家的“重要性”，即每个专家在当前 batch 中的门控值总和
            self.experts_importance = gate_outputs.sum(dim=0)  # 形状: [num_experts]

            self.importance_mean = self.experts_importance.mean()

            
        # 如果 constraint 为 True，则施加专家分配约束
        if self.enable_hard_constraint and hard_constraint and training:

            overused_experts = (self.experts_importance - self.importance_mean) > self.margin_threshold
            
            # 将过度使用的专家的门控概率设为 0
            gate_outputs[:, overused_experts] = 0
            
            # 检查是否有任何行的和为 0
            row_sums = gate_outputs.sum(dim=1, keepdim=True)
            zero_row_mask = (row_sums == 0)
            
            # 为避免除以 0，添加一个小常数 (epsilon)
            gate_outputs = gate_outputs + zero_row_mask * 1e-10
            
        # 获取每个样本的前 top_k 个最大值和对应的索引
        topk_values, topk_indices = torch.topk(gate_outputs, self.top_k, dim=1)

        # 对 top_k 的最大值进行 softmax 归一化
        topk_softmax_values = torch.softmax(topk_values, dim=1)

        # 构造一个与 gate_outputs 形状相同的全零矩阵
        topk_gate_outputs = torch.zeros_like(gate_outputs)

        # 将 softmax 后的 top_k 值放入对应的位置
        topk_gate_outputs.scatter_(1, topk_indices, topk_softmax_values)

        # 使用归一化后的 gate 输出进行后续操作
        gate_outputs = topk_gate_outputs
        
        expert_indices = torch.multinomial(gate_outputs, num_samples=1).squeeze()  # 形状: [batch_size]

        # 获取每个专家的输出
        expert_outputs = torch.stack([expert(x_flat) for expert in self.experts], dim=1)  # 形状: [batch_size, num_experts, output_dim]
        expert_outputs = torch.softmax(expert_outputs, dim=2) # 因为是分类任务，所以对每个专家的输出进行 softmax

        # 收集与采样的专家对应的输出
        final_output = expert_outputs[torch.arange(x.size(0)), expert_indices]  # 形状: [batch_size, output_dim]
        
        return final_output, expert_outputs, gate_outputs, expert_indices
    

    def compute_soft_constraint_loss(self, device):
        if self.enable_soft_constraint and self.training:
            # 计算变异系数 CV
            importance_std = self.experts_importance.std()
            cv_importance = importance_std / (self.importance_mean + 1e-8)  # 避免除以 0

            # 计算软约束损失 L_importance
            soft_constraint_loss = self.SGMoE_w_importance * (cv_importance ** 2)
            
            # 记录 soft_constraint_loss 以便在训练时加入到总损失中
            self.soft_constraint_loss = soft_constraint_loss
        else:
            self.soft_constraint_loss = torch.tensor(0.0, device=device)
        return self.soft_constraint_loss
