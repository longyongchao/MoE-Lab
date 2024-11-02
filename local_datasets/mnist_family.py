from torch.utils.data import ConcatDataset, DataLoader
from torchvision import datasets, transforms

# Function to load and combine Fashion-MNIST and EMNIST datasets
def get_combined_datasets(batch_size=64):
    # Define transformations
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    # Load Fashion-MNIST dataset
    fashion_mnist_train = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    fashion_mnist_test = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    # Load EMNIST dataset (Balanced split)
    emnist_train = datasets.EMNIST(root='./data', split='balanced', train=True, download=True, transform=transform)
    emnist_test = datasets.EMNIST(root='./data', split='balanced', train=False, download=True, transform=transform)

    # Adjust labels to avoid conflicts
    fashion_mnist_train.targets = fashion_mnist_train.targets + 0  # Fashion-MNIST labels start from 0
    fashion_mnist_test.targets = fashion_mnist_test.targets + 0

    emnist_train.targets = emnist_train.targets + 10  # EMNIST labels start from 10
    emnist_test.targets = emnist_test.targets + 10

    # Combine datasets
    combined_train_data = ConcatDataset([fashion_mnist_train, emnist_train])
    combined_test_data = ConcatDataset([fashion_mnist_test, emnist_test])

    # Print dataset sizes
    print(f"Training set size: {len(combined_train_data)}")
    print(f"Test set size: {len(combined_test_data)}")

    # Create data loaders
    combined_train_loader = DataLoader(combined_train_data, batch_size=batch_size, shuffle=True)
    combined_test_loader = DataLoader(combined_test_data, batch_size=batch_size, shuffle=False)

    return combined_train_loader, combined_test_loader

# Function to load Fashion-MNIST dataset
def get_fashion_mnist_datasets(batch_size=64):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    fashion_mnist_train = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    fashion_mnist_test = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    fashion_mnist_train_loader = DataLoader(fashion_mnist_train, batch_size=batch_size, shuffle=True)
    fashion_mnist_test_loader = DataLoader(fashion_mnist_test, batch_size=batch_size, shuffle=False)
    return fashion_mnist_train_loader, fashion_mnist_test_loader

# Function to load EMNIST dataset
def get_emnist_datasets(batch_size=64, origin=False):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    emnist_train = datasets.EMNIST(root='./data', split='balanced', train=True, download=True, transform=transform)
    emnist_test = datasets.EMNIST(root='./data', split='balanced', train=False, download=True, transform=transform)
    if not origin:
        emnist_train.targets = emnist_train.targets + 10  # EMNIST labels start from 10
        emnist_test.targets = emnist_test.targets + 10
    emnist_train_loader = DataLoader(emnist_train, batch_size=batch_size, shuffle=True)
    emnist_test_loader = DataLoader(emnist_test, batch_size=batch_size, shuffle=False)
    return emnist_train_loader, emnist_test_loader

FASHION_MNIST_LABELS = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot',
]

EMNIST_LABELS = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z',
    'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't'
]

def get_combined_label():
    # Combine all label mappings into one dictionary
    COMBINED_LABEL = FASHION_MNIST_LABELS + EMNIST_LABELS
    return COMBINED_LABEL

def get_split_point():
    return len(FASHION_MNIST_LABELS)