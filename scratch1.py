import os
import utils
import torchvision
import torchvision.transforms as transforms
if __name__ == '__main__':
    # train_transformer = transforms.Compose([
    #     transforms.RandomCrop(32, padding=4),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    # ])
    # trainset = torchvision.datasets.CIFAR10(root='C:/datasets/cifar10', train=True, download=True, transform=train_transformer)
    # print(trainset)
    exp_name = "baseline_standalone"
    model_dir = os.path.join("experiments", exp_name)
    print(model_dir)
    print(os.path.join(model_dir, 'train.log'))