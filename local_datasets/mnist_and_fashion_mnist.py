import torchvision
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset, DataLoader

def get_combined_datasets(batch_size=64):
    # 定义数据预处理
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    # 加载 MNIST 和 Fashion-MNIST 数据集
    mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    fashion_mnist_train = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    fashion_mnist_test = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    # 修改 Fashion-MNIST 的标签，使其与 MNIST 标签区分开来（加上 10）
    fashion_mnist_train.targets = fashion_mnist_train.targets + 10
    fashion_mnist_test.targets = fashion_mnist_test.targets + 10

    # 合并训练集和测试集
    combined_train_data = ConcatDataset([mnist_train, fashion_mnist_train])
    combined_test_data = ConcatDataset([mnist_test, fashion_mnist_test])

    # 打印合并后的数据集信息
    print(f"训练集样本数: {len(combined_train_data)}")
    print(f"测试集样本数: {len(combined_test_data)}")

    # 创建数据加载器
    combined_train_loader = DataLoader(combined_train_data, batch_size=batch_size, shuffle=True)
    combined_test_loader = DataLoader(combined_test_data, batch_size=batch_size, shuffle=False)

    return combined_train_loader, combined_test_loader


def get_combined_label():
    mnist_classes = [str(i) for i in range(10)]  # MNIST 类别是 '0' 到 '9'
    fashion_mnist_classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                         'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']  # Fashion-MNIST 类别
    return mnist_classes + fashion_mnist_classes


def get_minist_datasets(batch_size=64):
    # 定义数据预处理
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    # 加载 MNIST 和 Fashion-MNIST 数据集
    mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    mnist_train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
    mnist_test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=False)

    return mnist_train_loader, mnist_test_loader

def get_fashion_mnist_datasets(batch_size=64):
    # 定义数据预处理
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    # 加载 MNIST 和 Fashion-MNIST 数据集
    fashion_mnist_train = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    fashion_mnist_test = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    fashion_mnist_train_loader = DataLoader(fashion_mnist_train, batch_size=batch_size, shuffle=True)
    fashion_mnist_test_loader = DataLoader(fashion_mnist_test, batch_size=batch_size, shuffle=False)

    return fashion_mnist_train_loader, fashion_mnist_test_loader
    

