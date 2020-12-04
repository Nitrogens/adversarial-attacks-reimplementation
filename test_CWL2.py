import torch

import torchvision.transforms as transforms

from attackers.CWL2Attacker import CWL2Attacker
from datasets.TinyImageNet.TinyImageNet import TinyImageNetDataset

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = TinyImageNetDataset('./datasets/TinyImageNet', transform)

    attacker = CWL2Attacker(dataset)
    attacker.attack({
        'max_iter': 10000,
        'max_iter_binary_search': 9,
        'const_initial_value': 1e-3,
        'target': 567, 
        'k': 0,
        'lr': 1e-2,
        'early_abort': True,
    })
    print(attacker.count)
    print(attacker.get_accuracy())
