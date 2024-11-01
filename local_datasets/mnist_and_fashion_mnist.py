from torch.utils.data import ConcatDataset, DataLoader
from torchvision import datasets, transforms


# Function to load and combine datasets
def get_combined_datasets(batch_size=64):
    # Define transformations
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    # Load MNIST and Fashion-MNIST datasets
    mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    fashion_mnist_train = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    fashion_mnist_test = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    # Load EMNIST dataset (Balanced split)
    emnist_train = datasets.EMNIST(root='./data', split='balanced', train=True, download=True, transform=transform)
    emnist_test = datasets.EMNIST(root='./data', split='balanced', train=False, download=True, transform=transform)

    # Adjust labels to avoid conflicts
    fashion_mnist_train.targets = fashion_mnist_train.targets + 10
    fashion_mnist_test.targets = fashion_mnist_test.targets + 10

    emnist_train.targets = emnist_train.targets + 20
    emnist_test.targets = emnist_test.targets + 20

    # Combine datasets
    combined_train_data = ConcatDataset([mnist_train, fashion_mnist_train, emnist_train])
    combined_test_data = ConcatDataset([mnist_test, fashion_mnist_test, emnist_test])

    # Print dataset sizes
    print(f"Training set size: {len(combined_train_data)}")
    print(f"Test set size: {len(combined_test_data)}")

    # Create data loaders
    combined_train_loader = DataLoader(combined_train_data, batch_size=batch_size, shuffle=True)
    combined_test_loader = DataLoader(combined_test_data, batch_size=batch_size, shuffle=False)

    return combined_train_loader, combined_test_loader

# Function to load MNIST dataset
def get_mnist_datasets(batch_size=64):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    mnist_train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
    mnist_test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=False)
    return mnist_train_loader, mnist_test_loader

# Function to load Fashion-MNIST dataset
def get_fashion_mnist_datasets(batch_size=64):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    fashion_mnist_train = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    fashion_mnist_test = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    fashion_mnist_train.targets = fashion_mnist_train.targets + 10
    fashion_mnist_test.targets = fashion_mnist_test.targets + 10
    fashion_mnist_train_loader = DataLoader(fashion_mnist_train, batch_size=batch_size, shuffle=True)
    fashion_mnist_test_loader = DataLoader(fashion_mnist_test, batch_size=batch_size, shuffle=False)
    return fashion_mnist_train_loader, fashion_mnist_test_loader

# Function to load EMNIST dataset
def get_emnist_datasets(batch_size=64):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    emnist_train = datasets.EMNIST(root='./data', split='balanced', train=True, download=True, transform=transform)
    emnist_test = datasets.EMNIST(root='./data', split='balanced', train=False, download=True, transform=transform)
    emnist_train.targets = emnist_train.targets + 20
    emnist_test.targets = emnist_test.targets + 20
    emnist_train_loader = DataLoader(emnist_train, batch_size=batch_size, shuffle=True)
    emnist_test_loader = DataLoader(emnist_test, batch_size=batch_size, shuffle=False)
    return emnist_train_loader, emnist_test_loader


def get_combined_label():
    mnist_classes = [str(i) for i in range(10)]  # MNIST 类别是 '0' 到 '9'
    fashion_mnist_classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                         'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']  # Fashion-MNIST 类别
    return mnist_classes + fashion_mnist_classes

