import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm

from local_datasets import mnist_family
from local_models import models

# Train the model
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in tqdm(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
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
        for inputs, labels in tqdm(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    epoch_loss = running_loss / total
    epoch_acc = correct / total * 100
    return epoch_loss, epoch_acc

# Main function to train and test the model
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Hyperparameters
    input_dim = 28 * 28
    output_dim = 57  # 10 (Fashion-MNIST) + 47 (EMNIST)
    num_epochs = 10
    batch_size = 64
    learning_rate = 0.001

    # Dataset choice: "EMNIST", "Fashion MNIST", or "COMBINED"
    dataset_choice = input("Choose dataset (EMNIST, Fashion MNIST, COMBINED): ").strip().upper()


    # Load datasets based on user's choice
    if dataset_choice == "EMNIST":
        train_loader, test_loader = mnist_family.get_emnist_datasets(batch_size, origin=True)
        output_dim = 47
    elif dataset_choice == "FASHION MNIST":
        train_loader, test_loader = mnist_family.get_fashion_mnist_datasets(batch_size)
        output_dim = 10
    elif dataset_choice == "COMBINED":
        train_loader, test_loader = mnist_family.get_combined_datasets(batch_size)
        _, fashion_mnist_test_loader = mnist_family.get_fashion_mnist_datasets(batch_size)
        _, emnist_test_loader = mnist_family.get_emnist_datasets(batch_size)
    else:
        print("Invalid dataset choice. Please choose from 'EMNIST', 'Fashion MNIST', or 'COMBINED'.")
        return

    # Initialize the model, loss function, and optimizer
    model = models.SingleExpert(input_dim=input_dim, num_classes=output_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")

    # Test the model based on the user's dataset choice
    if dataset_choice == "EMNIST":
        test_loss, test_acc = test(model, test_loader, criterion, device)
        print(f"EMNIST Test Loss: {test_loss:.4f}, EMNIST Test Accuracy: {test_acc:.2f}%")
    elif dataset_choice == "FASHION MNIST":
        test_loss, test_acc = test(model, test_loader, criterion, device)
        print(f"Fashion-MNIST Test Loss: {test_loss:.4f}, Fashion-MNIST Test Accuracy: {test_acc:.2f}%")
    elif dataset_choice == "COMBINED":
        # Test on combined dataset
        combined_test_loss, combined_test_acc = test(model, test_loader, criterion, device)
        print(f"Combined Test Loss: {combined_test_loss:.4f}, Combined Test Accuracy: {combined_test_acc:.2f}%")

        # Test on Fashion-MNIST dataset
        fashion_mnist_test_loss, fashion_mnist_test_acc = test(model, fashion_mnist_test_loader, criterion, device)
        print(f"Fashion-MNIST Test Loss: {fashion_mnist_test_loss:.4f}, Fashion-MNIST Test Accuracy: {fashion_mnist_test_acc:.2f}%")

        # Test on EMNIST dataset
        emnist_test_loss, emnist_test_acc = test(model, emnist_test_loader, criterion, device)
        print(f"EMNIST Test Loss: {emnist_test_loss:.4f}, EMNIST Test Accuracy: {emnist_test_acc:.2f}%")

if __name__ == "__main__":
    main()
