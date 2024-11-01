import torch
import seaborn as sns
import matplotlib.pyplot as plt
import os
from local_datasets import mnist_and_fashion_mnist


# Save heatmap function
def save_heatmap(data, xlabel, ylabel, title, yticklabels, save_dir="heatmaps", heatmap_file_name="heatmap.png"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.figure(figsize=(10, 8))
    sns.heatmap(data, annot=True, cmap="YlGnBu", fmt=".2f", yticklabels=yticklabels)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    save_path = os.path.join(save_dir, f"{heatmap_file_name}.png")
    plt.savefig(save_path)
    print('Heatmap saved to:', save_path)
    plt.close()

# Test function with expert statistics
def test_with_expert_statistics(model, dataloader, device, num_classes=20):
    model.eval()
    correct, total = 0, 0
    class_expert_selection_count = torch.zeros(num_classes, model.num_experts, device=device)
    class_sample_count = torch.zeros(num_classes, device=device)
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            final_output, _, _, expert_indices = model(inputs)
            
            for i, label in enumerate(labels):
                # Ensure labels are correctly adjusted for Fashion-MNIST (label_offset = 10)
                adjusted_label = label.item()
                assert 0 <= adjusted_label < num_classes, f"Adjusted label {adjusted_label} out of bounds for {num_classes} classes."
                
                expert_idx = expert_indices[i]
                class_expert_selection_count[adjusted_label, expert_idx] += 1
                class_sample_count[adjusted_label] += 1
            
            _, predicted = torch.max(final_output, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy, class_expert_selection_count, class_sample_count

# Main test function
def test(combined_classes, moe_model, device, batch_size=64, heatmap_file_name=""):
    # Initialize combined expert selection and sample count for both datasets
    combined_class_expert_selection_count = torch.zeros(20, moe_model.num_experts, device=device)
    combined_class_sample_count = torch.zeros(20, device=device)

    # Test on MNIST dataset
    _, mnist_test_loader = mnist_and_fashion_mnist.get_minist_datasets(batch_size=batch_size)
    mnist_acc, mnist_class_expert_selection, mnist_class_sample_count = test_with_expert_statistics(
        moe_model, mnist_test_loader, device, num_classes=20)
    combined_class_expert_selection_count[:10, :] = mnist_class_expert_selection[:10, :]
    combined_class_sample_count[:10] = mnist_class_sample_count[:10]

    # Test on Fashion-MNIST dataset
    _, fashion_mnist_test_loader = mnist_and_fashion_mnist.get_fashion_mnist_datasets(batch_size=batch_size)
    fashion_mnist_acc, fashion_mnist_class_expert_selection, fashion_mnist_class_sample_count = test_with_expert_statistics(
        moe_model, fashion_mnist_test_loader, device, num_classes=20)
    combined_class_expert_selection_count[10:, :] = fashion_mnist_class_expert_selection[10:, :]
    combined_class_sample_count[10:] = fashion_mnist_class_sample_count[10:]

    # Calculate combined expert selection percentages
    combined_expert_selection_percentages = torch.zeros_like(combined_class_expert_selection_count)
    for class_idx in range(20):
        if combined_class_sample_count[class_idx] > 0:
            combined_expert_selection_percentages[class_idx] = 100 * combined_class_expert_selection_count[class_idx] / combined_class_sample_count[class_idx]
    
    all_acc = (mnist_acc + fashion_mnist_acc) / 2

    # Save heatmap
    save_heatmap(
        combined_expert_selection_percentages.cpu().numpy(), 
        xlabel="Experts", 
        ylabel="Classes", 
        title="Combined MNIST and Fashion-MNIST Expert Selection Heatmap",
        yticklabels=combined_classes,
        heatmap_file_name=heatmap_file_name + f"_{all_acc:.2f}_{mnist_acc:.2f}_{fashion_mnist_acc:.2f}",
    )


    # Output results
    print(f"Total Test Accuracy: {all_acc:.2f}%")
    print(f"MNIST Test Accuracy: {mnist_acc:.2f}%")
    print(f"Fashion-MNIST Test Accuracy: {fashion_mnist_acc:.2f}%")
