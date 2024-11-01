import torch
from local_datasets import mnist_and_fashion_mnist


# Test loop with expert selection tracking (for both dataset-level and class-level)
def test_with_expert_statistics(model, dataloader, device, dataset_name="", num_classes=20):
    model.eval()
    correct = 0
    total = 0
    
    # 初始化专家选择计数器（数据集层面）
    expert_selection_count = torch.zeros(model.num_experts, device=device)
    
    # 初始化每个类别的专家选择计数器（类别层面）
    class_expert_selection_count = torch.zeros(num_classes, model.num_experts, device=device)
    class_sample_count = torch.zeros(num_classes, device=device)  # 每个类别的样本计数
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            final_output, _, _, expert_indices = model(inputs)
            
            # 统计专家选择次数（数据集层面）
            for idx in expert_indices:
                expert_selection_count[idx] += 1
            
            # 统计专家选择次数（类别层面）
            for i, label in enumerate(labels):
                class_expert_selection_count[label, expert_indices[i]] += 1
                class_sample_count[label] += 1  # 统计每个类别的样本数量
            
            # Get predictions
            _, predicted = torch.max(final_output, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'\n{dataset_name} 测试集准确率: {accuracy:.2f}%')

    # 输出专家选择次数统计（数据集层面）
    print(f"\n{dataset_name} 专家选择次数统计（数据集层面）：")
    for i, count in enumerate(expert_selection_count):
        print(f'专家 {i}: 被选择 {count.item()} 次，占比 {100 * count.item() / total:.2f}%')

    # 输出专家选择次数统计（类别层面）
    print(f"\n{dataset_name} 专家选择次数统计（类别层面）：")
    for class_idx in range(num_classes):
        print(f'类别 {class_idx} 的专家选择情况:')
        for expert_idx in range(model.num_experts):
            count = class_expert_selection_count[class_idx, expert_idx]
            if class_sample_count[class_idx] > 0:
                percentage = 100 * count.item() / class_sample_count[class_idx].item()
            else:
                percentage = 0.0
            print(f'  专家 {expert_idx}: 被选择 {count.item()} 次，占比 {percentage:.2f}%')

    return accuracy, expert_selection_count, class_expert_selection_count, class_sample_count


def test(combined_classes, moe_model, device, batch_size=64):
    
    print("Testing MoE Model on MNIST and Fashion-MNIST...")

    # Test on individual datasets
    print("Testing on MNIST dataset...")
    _, mnist_test_loader = mnist_and_fashion_mnist.get_minist_datasets(batch_size=batch_size)
    test_accuracy_mnist, mnist_expert_selection, mnist_class_expert_selection, mnist_class_sample_count = test_with_expert_statistics(
        moe_model, mnist_test_loader, device, dataset_name="MNIST", num_classes=len(combined_classes))

    print("Testing on Fashion-MNIST dataset...")
    _, fashion_mnist_test_loader = mnist_and_fashion_mnist.get_fashion_mnist_datasets(batch_size=batch_size)
    test_accuracy_fashion_mnist, fashion_mnist_expert_selection, fashion_mnist_class_expert_selection, fashion_mnist_class_sample_count = test_with_expert_statistics(
        moe_model, fashion_mnist_test_loader, device, dataset_name="Fashion-MNIST", num_classes=len(combined_classes))

    # Combine MNIST and Fashion-MNIST results for dataset-level statistics
    combined_expert_selection = mnist_expert_selection + fashion_mnist_expert_selection
    total_samples = mnist_class_sample_count.sum().item() + fashion_mnist_class_sample_count.sum().item()

    # 输出合并后的专家选择情况（数据集层面）
    print(f"\n合并后的专家选择次数统计（数据集层面）：")
    for i, count in enumerate(combined_expert_selection):
        print(f'专家 {i}: 被选择 {count.item()} 次，占比 {100 * count.item() / total_samples:.2f}%')

    # 输出排序后的专家选择情况（数据集层面）
    def print_sorted_expert_selection(expert_selection_count, total_samples, dataset_name):
        sorted_indices = torch.argsort(expert_selection_count, descending=True)
        print(f"\n{dataset_name} 专家选择次数排序（数据集层面）：")
        for idx in sorted_indices:
            print(f'专家 {idx.item()}: 被选择 {expert_selection_count[idx].item()} 次，占比 {100 * expert_selection_count[idx].item() / total_samples:.2f}%')

    # 输出排序后的专家选择情况
    print_sorted_expert_selection(combined_expert_selection, total_samples, "MNIST + Fashion-MNIST")

    # 输出类别层面的专家选择情况
    def print_class_expert_selection(class_expert_selection_count, class_sample_count, dataset_name):
        print(f"\n{dataset_name} 专家选择次数统计（类别层面）：")
        for class_idx in range(class_expert_selection_count.size(0)):
            print(f'类别 {class_idx} 的专家选择情况:')
            for expert_idx in range(class_expert_selection_count.size(1)):
                count = class_expert_selection_count[class_idx, expert_idx]
                if class_sample_count[class_idx] > 0:
                    percentage = 100 * count.item() / class_sample_count[class_idx].item()
                else:
                    percentage = 0.0
                print(f'  专家 {expert_idx}: 被选择 {count.item()} 次，占比 {percentage:.2f}%')

    # 输出类别层面的专家选择情况
    print_class_expert_selection(mnist_class_expert_selection, mnist_class_sample_count, "MNIST")
    print_class_expert_selection(fashion_mnist_class_expert_selection, fashion_mnist_class_sample_count, "Fashion-MNIST")
