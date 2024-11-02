
"""更复杂的数据集组合，包括CIFAR-10、CIFAR-100、Oxford-IIIT Pet"""

from torch.utils.data import ConcatDataset, DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import CIFAR10, CIFAR100, OxfordIIITPet

# Function to load and combine CIFAR-10, CIFAR-100, Oxford-IIIT Pet
def get_combined_datasets(batch_size=64, image_size=64):
    # Define transformations: Resize to a common size and normalize
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # For RGB images
    ])

    # Load CIFAR-10 dataset
    cifar10_train = CIFAR10(root='./data', train=True, download=True, transform=transform)
    cifar10_test = CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Load CIFAR-100 dataset
    cifar100_train = CIFAR100(root='./data', train=True, download=True, transform=transform)
    cifar100_test = CIFAR100(root='./data', train=False, download=True, transform=transform)

    # Load Oxford-IIIT Pet dataset
    pet_train = OxfordIIITPet(root='./data', split='trainval', download=True, transform=transform)
    pet_test = OxfordIIITPet(root='./data', split='test', download=True, transform=transform)


    # Adjust labels to avoid conflicts
    cifar10_train.targets = [label + 0 for label in cifar10_train.targets]  # CIFAR-10 labels start from 0
    cifar10_test.targets = [label + 0 for label in cifar10_test.targets]

    cifar100_train.targets = [label + 10 for label in cifar100_train.targets]  # CIFAR-100 labels start from 10
    cifar100_test.targets = [label + 10 for label in cifar100_test.targets]

    pet_train.targets = [label + 110 for label in pet_train.targets]  # Oxford-IIIT Pet labels start from 110
    pet_test.targets = [label + 110 for label in pet_test.targets]

    # Combine datasets
    combined_train_data = ConcatDataset([cifar10_train, cifar100_train, pet_train])
    combined_test_data = ConcatDataset([cifar10_test, cifar100_test, pet_test])

    # Print dataset sizes
    print(f"Training set size: {len(combined_train_data)}")
    print(f"Test set size: {len(combined_test_data)}")

    # Create data loaders
    combined_train_loader = DataLoader(combined_train_data, batch_size=batch_size, shuffle=True)
    combined_test_loader = DataLoader(combined_test_data, batch_size=batch_size, shuffle=False)

    return combined_train_loader, combined_test_loader

# Function to load CIFAR-10 dataset
def get_cifar10_datasets(batch_size=64, image_size=64):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    cifar10_train = CIFAR10(root='./data', train=True, download=True, transform=transform)
    cifar10_test = CIFAR10(root='./data', train=False, download=True, transform=transform)
    cifar10_train_loader = DataLoader(cifar10_train, batch_size=batch_size, shuffle=True)
    cifar10_test_loader = DataLoader(cifar10_test, batch_size=batch_size, shuffle=False)
    return cifar10_train_loader, cifar10_test_loader

# Function to load CIFAR-100 dataset
def get_cifar100_datasets(batch_size=64, image_size=64):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    cifar100_train = CIFAR100(root='./data', train=True, download=True, transform=transform)
    cifar100_test = CIFAR100(root='./data', train=False, download=True, transform=transform)
    cifar100_train_loader = DataLoader(cifar100_train, batch_size=batch_size, shuffle=True)
    cifar100_test_loader = DataLoader(cifar100_test, batch_size=batch_size, shuffle=False)
    return cifar100_train_loader, cifar100_test_loader

# Function to load Oxford-IIIT Pet dataset
def get_pet_datasets(batch_size=64, image_size=64):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    pet_train = OxfordIIITPet(root='./data', split='trainval', download=True, transform=transform)
    pet_test = OxfordIIITPet(root='./data', split='test', download=True, transform=transform)
    pet_train_loader = DataLoader(pet_train, batch_size=batch_size, shuffle=True)
    pet_test_loader = DataLoader(pet_test, batch_size=batch_size, shuffle=False)
    return pet_train_loader, pet_test_loader

# Define label names for each dataset
CIFAR10_LABELS = [
    'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'
]

CIFAR100_LABELS = [f'class_{i}' for i in range(100)]  # Placeholder for CIFAR-100 labels

PET_LABELS = ['Abyssinian', 'American Bulldog', 'American Pit Bull Terrier', 'Basset Hound', 'Beagle', 'Bengal', 
              'Birman', 'Bombay', 'Boxer', 'British Shorthair', 'Chihuahua', 'Egyptian Mau', 'English Cocker Spaniel', 
              'English Setter', 'German Shorthaired', 'Great Pyrenees', 'Havanese', 'Japanese Chin', 'Keeshond', 
              'Leonberger', 'Maine Coon', 'Miniature Pinscher', 'Newfoundland', 'Persian', 'Pomeranian', 'Pug', 
              'Ragdoll', 'Russian Blue', 'Saint Bernard', 'Samoyed', 'Scottish Terrier', 'Shiba Inu', 'Siamese', 
              'Sphynx', 'Staffordshire Bull Terrier', 'Wheaten Terrier', 'Yorkshire Terrier']

def get_combined_label():
    # Combine all label mappings into one dictionary
    COMBINED_LABEL = CIFAR10_LABELS + CIFAR100_LABELS + PET_LABELS
    return COMBINED_LABEL
