import torch
from tqdm import tqdm


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


def train(model, dataloader, optimizer, device, classes, num_epochs=(5, 10)):
    model.train()
    
    for epoch in range(num_epochs[0] + num_epochs[1]):
        running_loss = 0.0
        correct = 0
        total = 0

        hard_constraint = True if epoch <= num_epochs[0] else False
        
        # 使用 tqdm 显示进度
        for step, (inputs, labels) in enumerate(tqdm(dataloader)):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            
            # Forward pass
            final_output, expert_outputs, gate_outputs, expert_indices = model(
                inputs, 
                training=True,
                hard_constraint=hard_constraint
            )
            
            # Convert labels to one-hot encoding
            one_hot_labels = one_hot_encoding(labels, num_classes=len(classes))
            
            # Compute MoE loss
            loss = moe_loss(one_hot_labels, expert_outputs, gate_outputs)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(final_output, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}, Accuracy: {100 * correct / total:.2f}%')

    return model.expert_selection_count
