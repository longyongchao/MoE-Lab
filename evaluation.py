import torch
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Save heatmap function
def save_heatmap(data, xlabel, ylabel, title, yticklabels, save_dir="heatmaps", heatmap_file_name="heatmap.png"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.figure(figsize=(10, 18))
    sns.heatmap(data, annot=True, cmap="YlGnBu", fmt=".1f", yticklabels=yticklabels)
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
def test(
    combined_classes, 
    moe_model, 
    device, 
    batch_size=64, 
    heatmap_file_name="", 
    split_point=None,  # Now a dictionary
    datasets_info=None  # Information about datasets to load
):
    """Test function for multiple datasets with split points.

    Args:
        combined_classes (list): List of class labels.
        moe_model (torch.nn.Module): The model to test.
        device (torch.device): Device to run the test on.
        batch_size (int, optional): Batch size for dataloaders. Defaults to 64.
        heatmap_file_name (str, optional): Filename for saving the heatmap. Defaults to "".
        split_point (dict, optional): Dictionary with dataset names and their split points.
        datasets_info (list of dict, optional): List of dataset information, each dict contains 'name' and 'loader_func'.
    """
    assert split_point is not None, "split_point dictionary must be provided."
    assert datasets_info is not None, "datasets_info list must be provided."

    classes_len = len(combined_classes)
    
    # Initialize combined expert selection and sample count for all datasets
    combined_class_expert_selection_count = torch.zeros(classes_len, moe_model.num_experts, device=device)
    combined_class_sample_count = torch.zeros(classes_len, device=device)

    total_acc = 0
    total_samples = 0

    current_class_offset = 0  # Track the current starting index for the next dataset

    # Iterate through datasets and perform testing
    for dataset_info in datasets_info:
        dataset_name = dataset_info['name']
        data_loader_func = dataset_info['loader_func']
        split_idx = split_point[dataset_name]

        # Load dataset
        _, test_loader = data_loader_func(batch_size=batch_size)

        # Test on the current dataset
        dataset_acc, dataset_class_expert_selection, dataset_class_sample_count = test_with_expert_statistics(
            moe_model, test_loader, device, num_classes=classes_len)
        
        print(f"Test Accuracy for {dataset_name}: {dataset_acc:.2f}%")

        # Update the combined expert selection and sample count
        combined_class_expert_selection_count[current_class_offset:split_idx, :] = dataset_class_expert_selection[current_class_offset:split_idx, :]
        combined_class_sample_count[current_class_offset:split_idx] = dataset_class_sample_count[current_class_offset:split_idx]

        # Calculate accuracy weighted by dataset size
        dataset_size = len(test_loader.dataset)
        total_acc += dataset_acc * dataset_size
        total_samples += dataset_size

        # Update the class offset for the next dataset
        current_class_offset = split_idx

    # Calculate overall accuracy
    all_acc = total_acc / total_samples

    # Calculate combined expert selection percentages
    combined_expert_selection_percentages = torch.zeros_like(combined_class_expert_selection_count)
    for class_idx in range(classes_len):
        if combined_class_sample_count[class_idx] > 0:
            combined_expert_selection_percentages[class_idx] = 100 * combined_class_expert_selection_count[class_idx] / combined_class_sample_count[class_idx]
    
    # Save heatmap
    save_heatmap(
        combined_expert_selection_percentages.cpu().numpy(), 
        xlabel="Experts", 
        ylabel="Classes", 
        title="Combined Dataset Expert Selection Heatmap",
        yticklabels=combined_classes,
        heatmap_file_name=heatmap_file_name + f"_{all_acc:.2f}",
    )

    # Output results
    print(f"Total Test Accuracy: {all_acc:.2f}%")
