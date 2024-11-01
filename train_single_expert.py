import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from tqdm import tqdm

from local_models.models import SingleExpert
from local_datasets import mnist_and_fashion_mnist


# Train the model
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in tqdm(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total * 100
    return epoch_loss, epoch_acc

# Test the model
def test(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total * 100
    return epoch_loss, epoch_acc

# Function to calculate class distribution in the training set
def calculate_class_distribution(train_loader):
    class_counts = Counter()
    
    for _, labels in train_loader:
        class_counts.update(labels.numpy())
    
    total_samples = sum(class_counts.values())
    class_distribution = {k: v / total_samples for k, v in class_counts.items()}
    
    return class_distribution

# Main function to train and test the model
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Hyperparameters
    input_dim = 28 * 28
    output_dim = 20
    num_epochs = 10
    batch_size = 64
    learning_rate = 0.001

    # Initialize the model, loss function, and optimizer
    model = SingleExpert(input_dim=input_dim, output_dim=output_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Load datasets
    combined_train_loader, combined_test_loader = mnist_and_fashion_mnist.get_combined_datasets(batch_size)
    _, mnist_test_loader = mnist_and_fashion_mnist.get_minist_datasets(batch_size)
    _, fashion_mnist_test_loader = mnist_and_fashion_mnist.get_fashion_mnist_datasets(batch_size)

    # Calculate class distribution in the combined training set
    class_distribution = calculate_class_distribution(combined_train_loader)
    print("Class distribution in the combined training set:")
    for class_id, proportion in class_distribution.items():
        print(f"Class {class_id}: {proportion * 100:.2f}%")

    # Training loop
    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, combined_train_loader, criterion, optimizer, device)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")

    # Test on combined dataset
    combined_test_loss, combined_test_acc = test(model, combined_test_loader, criterion, device)
    print(f"Combined Test Loss: {combined_test_loss:.4f}, Combined Test Accuracy: {combined_test_acc:.2f}%")

    # Test on MNIST dataset
    mnist_test_loss, mnist_test_acc = test(model, mnist_test_loader, criterion, device)
    print(f"MNIST Test Loss: {mnist_test_loss:.4f}, MNIST Test Accuracy: {mnist_test_acc:.2f}%")

    # Test on Fashion-MNIST dataset
    fashion_test_loss, fashion_test_acc = test(model, fashion_mnist_test_loader, criterion, device)
    print(f"Fashion-MNIST Test Loss: {fashion_test_loss:.4f}, Fashion-MNIST Test Accuracy: {fashion_test_acc:.2f}%")

if __name__ == "__main__":
    main()
