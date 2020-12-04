# FGSM Attacker
# Reference: https://colab.research.google.com/drive/1ePbuJwBwVsHkfztpXKjKuqaEZ3h27F_A
# Written by Nitrogens

import torch

import torch.nn.functional as F

import torchvision.models as models
import torchvision.transforms as transforms

from .Attacker import Attacker


class FGSMAttacker(Attacker):
    def __init__(self, dataset, model=models.vgg16(pretrained=True)):
        super().__init__(dataset, model)

    @staticmethod
    def generate_perturbed_image(self, image, pred, pred_correct, params):
        loss = F.nll_loss(pred, pred_correct)
        self.model.zero_grad()
        loss.backward()
        data_grad = image.grad.data
        sign_data_grad = data_grad.sign()
        return_image = image + params['epsilon'] * sign_data_grad
        return return_image
