import torch

import torchvision.transforms as transforms
import torchvision.models as models

from torchvision.datasets import ImageNet

from attackers.CWLiAttacker import CWLiAttacker
from datasets.TinyImageNet.TinyImageNet import TinyImageNetDataset

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = TinyImageNetDataset('./datasets/TinyImageNet', transform)

    attacker = CWLiAttacker(dataset)
    attacker.attack({
        'max_iter': 1000,
        'const_initial_value': 10,
        'const_max_value': 15,
        'target': 567, 
        'lr': 1e-2,
        'early_abort': True,
        'reduce_const': False,
        'const_factor': 2.0,
        'decrease_factor': 0.9,
    })
    print(attacker.count)
    print(attacker.get_accuracy())
