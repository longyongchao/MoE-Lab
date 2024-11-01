import torch
import torch.nn as nn


# Define a simple feedforward network (single expert)
class SingleExpert(nn.Module):
    def __init__(self, input_dim=28*28, output_dim=20):
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
    def __init__(self, input_dim=28*28, output_dim=20, num_experts=4):
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
