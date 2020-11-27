import numpy as np
import torch

import torchvision.models as models
import torchvision.transforms as transforms


device = torch.device('cuda')

class Attacker(object):
    def __init__(self, dataset, model=models.vgg16(pretrained=True)):
        self.model = model
        self.model.cuda()
        self.model.eval()
        
        self.dataset = dataset
        
        self.loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=1,
            shuffle=False
        )

        self.count = {
            'wrong': 0, 
            'success': 0, 
            'fail': 0
        }

    def get_accuracy(self):
        return self.count['success'] / (self.count['wrong'] + self.count['success'] + self.count['fail'])

    def generate_perturbed_image(self, image, pred, pred_correct, params):
        pass

    def attack(self, params):
        # Reset the statistical value
        for k, _ in self.count.items():
            self.count[k] = 0

        # Process datas
        for (data, target) in self.loader:
            data, target = data.to(device), target.to(device)
            data_raw = data
            data.requires_grad = True

            pred = self.model(data)
            pred_original = pred.max(1, keepdim=True)[1]

            # If the backbone network get the wrong answer, skip it
            if pred_original.item() != target.item():
                self.count['wrong'] += 1
                continue

            # Otherwise, add adversarial perturbation to the corresponding image
            perturbed_data = self.generate_perturbed_image(data, pred, target, params)

            target_perturbed = self.model(perturbed_data)
            pred_perturbed = target_perturbed.max(1, keepdim=True)[1]

            if target.item() == pred_perturbed.item():
                # Attack failed
                self.count['fail'] += 1
            else:
                # Attack successfully
                self.count['success'] += 1
