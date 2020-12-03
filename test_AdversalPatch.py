import torch

import torchvision.transforms as transforms

from attackers.AdversalPatchAttacker import AdversalPatchAttacker
from datasets.TinyImageNet.TinyImageNet import TinyImageNetDataset

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    dataset = TinyImageNetDataset('./datasets/TinyImageNet', transform)

    attacker = AdversalPatchAttacker(dataset)
    attacker.universal_attack({
        'noise_percentage': 0.1,
        'p_thr': 0.9,
        'max_iter': 100,
        'target': 567, 
        'lr': 1.0,
        'num_epoch': 20,
        'sample_size': 200,
    })
    print(attacker.count)
    print(attacker.get_accuracy())
