import torch
import torch.nn as nn


# Define a simple feedforward network (single expert)
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

    def forward(self, x, training=False, if_softmax=True):
        sg_noise = self.noise(x) if (self.noise is not None and training) else 0
        x = self.layers[0](x)
        for layer in self.layers[1:]:
            x = layer(self.relu(x))
        return self.softmax(x + sg_noise) if if_softmax else x + sg_noise


"""
- 引入专家分配约束：我们需要在每个训练步骤中跟踪每个专家的选择次数，并计算这些选择次数的平均值。如果某个专家的选择次数超过了平均值加上一个预设的阈值m，则将该专家的选择概率设为零。
- 重新归一化 gating 输出：当某个专家的选择概率被设为零后，我们需要重新归一化剩下的专家的 gating 输出，使得这些输出仍然是一个有效的概率分布（即所有 gating 输出的和为 1）。
"""
class MoE(nn.Module):
    def __init__(
        self, 
        input_dim, 
        output_dim, 
        expert_hidden_dim=[],
        gating_hidden_dim=[50], 
        num_experts=4, 
        margin_threshold=0.1,
        enable_sparsely_gated_noise=False,
        enable_hard_constraint=False,
        top_k=1,
    ):
        super(MoE, self).__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.margin_threshold = margin_threshold  # m
        self.top_k = top_k
        self.enable_hard_constraint = enable_hard_constraint
        
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
        x_flat = x.view(-1, self.input_dim)  # 将图片展平输入
        
        # 获取门控网络的输出概率
        gate_outputs = self.gating_network(x_flat, training=training) # 形状: [batch_size, num_experts]

        # 克隆 gate_outputs 以避免原地修改
        gate_outputs_cloned = gate_outputs.clone()

        # 如果 constraint 为 True，则施加专家分配约束
        if self.enable_hard_constraint and hard_constraint and training:

            # 每个 step 重新初始化专家选择概率累计计数
            self.expert_selection_count = gate_outputs.sum(dim=0)  # 形状: [num_experts]
            mean_selection = self.expert_selection_count.mean()  # 计算专家分配的平均值
            overused_experts = (self.expert_selection_count - mean_selection) > self.margin_threshold
            
            # 将过度使用的专家的门控概率设为 0
            gate_outputs_cloned[:, overused_experts] = 0
            
            # 检查是否有任何行的和为 0
            row_sums = gate_outputs_cloned.sum(dim=1, keepdim=True)
            zero_row_mask = (row_sums == 0)
            
            # 为避免除以 0，添加一个小常数 (epsilon)
            gate_outputs_cloned = gate_outputs_cloned + zero_row_mask * 1e-10
            
            # 重新归一化门控概率
            gate_outputs_cloned = gate_outputs_cloned / gate_outputs_cloned.sum(dim=1, keepdim=True)

        # 调试：检查 gate_outputs_cloned 中是否有无效值
        if torch.isnan(gate_outputs_cloned).any() or torch.isinf(gate_outputs_cloned).any() or (gate_outputs_cloned < 0).any():
            print("Invalid values detected in gate_outputs_cloned!")
            print("gate_outputs_cloned:", gate_outputs_cloned)
            raise ValueError("gate_outputs_cloned contains invalid values (nan, inf, or negative numbers)")

        # 根据门控概率随机选择一个专家
        expert_indices = torch.multinomial(gate_outputs_cloned, num_samples=self.top_k).squeeze()  # 形状: [batch_size]

        # 获取每个专家的输出
        expert_outputs = torch.stack([expert(x_flat) for expert in self.experts], dim=1)  # 形状: [batch_size, num_experts, output_dim]
        expert_outputs = torch.softmax(expert_outputs, dim=2)

        # 收集与采样的专家对应的输出
        final_output = expert_outputs[torch.arange(x.size(0)), expert_indices]  # 形状: [batch_size, output_dim]
        
        return final_output, expert_outputs, gate_outputs_cloned, expert_indices
