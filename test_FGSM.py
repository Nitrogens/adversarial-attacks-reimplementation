import torch

import torchvision.transforms as transforms

from attackers.FGSMAttacker import FGSMAttacker
from datasets.TinyImageNet.TinyImageNet import TinyImageNetDataset

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = TinyImageNetDataset('./datasets/TinyImageNet', transform)

    attacker = FGSMAttacker(dataset)
    attacker.attack({'epsilon': 0.01})
    print(attacker.count)
    print(attacker.get_accuracy())
