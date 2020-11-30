import torch

import torchvision.transforms as transforms

from attackers.UniversalAttacker import UniversalAttacker
from datasets.TinyImageNet.TinyImageNet import TinyImageNetDataset

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = TinyImageNetDataset('./datasets/TinyImageNet', transform)

    attacker = UniversalAttacker(dataset)
    attacker.universal_attack({
        'max_iter_uni': 10,
        'max_iter': 50,
        'p': 2,
        'xi': 2000,
        'delta': 0.2,
        'sample_size': 200,
        'batch_size': 10,
        'eps': 1e-4,
        'overshoot': 0.02,
    })
    print(attacker.count)
    print(attacker.get_accuracy())
