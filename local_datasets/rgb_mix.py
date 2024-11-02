"""更复杂的数据集组合，包括 CIFAR-10、CIFAR-100、Oxford-IIIT Pet"""

from torch.utils.data import ConcatDataset, DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, OxfordIIITPet

# 常量定义
IMAGE_SIZE = 64
CIFAR10_LABELS = ['cifar10_airplane', 'cifar10_automobile', 'cifar10_bird', 'cifar10_cat', 'cifar10_deer', 'cifar10_dog', 'cifar10_frog', 'cifar10_horse', 'cifar10_ship', 'cifar10_truck']

CIFAR100_LABELS = ['cifar100_apple', 'cifar100_aquarium_fish', 'cifar100_baby', 'cifar100_bear', 'cifar100_beaver', 'cifar100_bed', 'cifar100_bee', 'cifar100_beetle', 'cifar100_bicycle', 'cifar100_bottle', 'cifar100_bowl', 'cifar100_boy', 'cifar100_bridge', 'cifar100_bus', 'cifar100_butterfly', 'cifar100_camel', 'cifar100_can', 'cifar100_castle', 'cifar100_caterpillar', 'cifar100_cattle', 'cifar100_chair', 'cifar100_chimpanzee', 'cifar100_clock', 'cifar100_cloud', 'cifar100_cockroach', 'cifar100_couch', 'cifar100_crab', 'cifar100_crocodile', 'cifar100_cup', 'cifar100_dinosaur', 'cifar100_dolphin', 'cifar100_elephant', 'cifar100_flatfish', 'cifar100_forest', 'cifar100_fox', 'cifar100_girl', 'cifar100_hamster', 'cifar100_house', 'cifar100_kangaroo', 'cifar100_keyboard', 'cifar100_lamp', 'cifar100_lawn_mower', 'cifar100_leopard', 'cifar100_lion', 'cifar100_lizard', 'cifar100_lobster', 'cifar100_man', 'cifar100_maple_tree', 'cifar100_motorcycle', 'cifar100_mountain', 'cifar100_mouse', 'cifar100_mushroom', 'cifar100_oak_tree', 'cifar100_orange', 'cifar100_orchid', 'cifar100_otter', 'cifar100_palm_tree', 'cifar100_pear', 'cifar100_pickup_truck', 'cifar100_pine_tree', 'cifar100_plain', 'cifar100_plate', 'cifar100_poppy', 'cifar100_porcupine', 'cifar100_possum', 'cifar100_rabbit', 'cifar100_raccoon', 'cifar100_ray', 'cifar100_road', 'cifar100_rocket', 'cifar100_rose', 'cifar100_sea', 'cifar100_seal', 'cifar100_shark', 'cifar100_shrew', 'cifar100_skunk', 'cifar100_skyscraper', 'cifar100_snail', 'cifar100_snake', 'cifar100_spider', 'cifar100_squirrel', 'cifar100_streetcar', 'cifar100_sunflower', 'cifar100_sweet_pepper', 'cifar100_table', 'cifar100_tank', 'cifar100_telephone', 'cifar100_television', 'cifar100_tiger', 'cifar100_tractor', 'cifar100_train', 'cifar100_trout', 'cifar100_tulip', 'cifar100_turtle', 'cifar100_wardrobe', 'cifar100_whale', 'cifar100_willow_tree', 'cifar100_wolf', 'cifar100_woman', 'cifar100_worm']

PET_LABELS = [
    'Abyssinian', 'American Bulldog', 'American Pit Bull Terrier', 'Basset Hound', 'Beagle', 'Bengal', 
    'Birman', 'Bombay', 'Boxer', 'British Shorthair', 'Chihuahua', 'Egyptian Mau', 'English Cocker Spaniel', 
    'English Setter', 'German Shorthaired', 'Great Pyrenees', 'Havanese', 'Japanese Chin', 'Keeshond', 
    'Leonberger', 'Maine Coon', 'Miniature Pinscher', 'Newfoundland', 'Persian', 'Pomeranian', 'Pug', 
    'Ragdoll', 'Russian Blue', 'Saint Bernard', 'Samoyed', 'Scottish Terrier', 'Shiba Inu', 'Siamese', 
    'Sphynx', 'Staffordshire Bull Terrier', 'Wheaten Terrier', 'Yorkshire Terrier'
]

# 通用的图像变换
def get_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 针对RGB图像的归一化
    ])

# 加载 CIFAR-10 数据集
def get_cifar10_datasets(batch_size=64):
    transform = get_transform()
    cifar10_train = CIFAR10(root='./data', train=True, download=True, transform=transform)
    cifar10_test = CIFAR10(root='./data', train=False, download=True, transform=transform)
    cifar10_train_loader = DataLoader(cifar10_train, batch_size=batch_size, shuffle=True)
    cifar10_test_loader = DataLoader(cifar10_test, batch_size=batch_size, shuffle=False)

    return cifar10_train_loader, cifar10_test_loader

# 加载 CIFAR-100 数据集
def get_cifar100_datasets(batch_size=64, origin=False):
    transform = get_transform()
    cifar100_train = CIFAR100(root='./data', train=True, download=True, transform=transform)
    cifar100_test = CIFAR100(root='./data', train=False, download=True, transform=transform)
    if not origin:
        cifar100_train.targets = [i + len(CIFAR10_LABELS) for i in cifar100_train.targets]
        cifar100_test.targets = [i + len(CIFAR10_LABELS) for i in cifar100_test.targets]
    cifar100_train_loader = DataLoader(cifar100_train, batch_size=batch_size, shuffle=True)
    cifar100_test_loader = DataLoader(cifar100_test, batch_size=batch_size, shuffle=False)

    return cifar100_train_loader, cifar100_test_loader

# 加载 Oxford-IIIT Pet 数据集
def get_pet_datasets(batch_size=64, origin=False):
    transform = get_transform()
    pet_train = OxfordIIITPet(root='./data', split='trainval', download=True, transform=transform)
    pet_test = OxfordIIITPet(root='./data', split='test', download=True, transform=transform)
    if not origin:
        pet_train.targets = [i + len(CIFAR10_LABELS) + len(CIFAR100_LABELS) for i in pet_train._labels]
        pet_test.targets = [i + len(CIFAR10_LABELS) + len(CIFAR100_LABELS) for i in pet_test._labels]
    pet_train_loader = DataLoader(pet_train, batch_size=batch_size, shuffle=True)
    pet_test_loader = DataLoader(pet_test, batch_size=batch_size, shuffle=False)
    return pet_train_loader, pet_test_loader

# 加载并组合 CIFAR-10、CIFAR-100 和 Oxford-IIIT Pet 数据集
def get_combined_datasets(batch_size=64):
    # 加载单独的数据集
    cifar10_train_loader, cifar10_test_loader = get_cifar10_datasets(batch_size)
    cifar100_train_loader, cifar100_test_loader = get_cifar100_datasets(batch_size)
    pet_train_loader, pet_test_loader = get_pet_datasets(batch_size)

    # 组合数据集
    combined_train_data = ConcatDataset([
        cifar10_train_loader.dataset, 
        cifar100_train_loader.dataset, 
        # pet_train_loader.dataset
    ])
    combined_test_data = ConcatDataset([
        cifar10_test_loader.dataset, 
        cifar100_test_loader.dataset, 
        # pet_test_loader.dataset
    ])

    # 创建数据加载器
    combined_train_loader = DataLoader(combined_train_data, batch_size=batch_size, shuffle=True)
    combined_test_loader = DataLoader(combined_test_data, batch_size=batch_size, shuffle=False)

    return combined_train_loader, combined_test_loader

# 获取输入维度
def get_input_dim():
    return IMAGE_SIZE * IMAGE_SIZE * 3

# 获取组合后的标签
def get_combined_label():
    return CIFAR10_LABELS + CIFAR100_LABELS # + PET_LABELS

# 获取各数据集的分割点
def get_split_point():
    return {
        'cifar10': len(CIFAR10_LABELS),
        'cifar100': len(CIFAR10_LABELS) + len(CIFAR100_LABELS),
        # 'pet': len(CIFAR10_LABELS) + len(CIFAR100_LABELS) + len(PET_LABELS)
    }

# 获取数据集信息
def get_datasets_info():
    return [
        {'name': 'cifar10', 'loader_func': get_cifar10_datasets},
        {'name': 'cifar100', 'loader_func': get_cifar100_datasets},
        # {'name': 'pet', 'loader_func': get_pet_datasets}
    ]
