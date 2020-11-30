# DeepFool Attacker
# Reference: https://github.com/LTS4/DeepFool/blob/master/Python/deepfool.py
# Written by Nitrogens

import torch
import copy
import numpy as np

import torch.nn.functional as F

import torchvision.models as models
import torchvision.transforms as transforms

from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients

from .Attacker import Attacker


class DeepFoolAttacker(Attacker):
    def __init__(self, dataset, model=models.vgg16(pretrained=True)):
        super().__init__(dataset, model)

    @staticmethod
    def generate_perturbed_image(self, image, pred, pred_correct, params):  
        i = 0
        num_classes = len(pred[0])
        max_iter = 50
        if params['max_iter'] is not None:
            max_iter = params['max_iter']
        min_w = np.zeros(image.data.cpu().numpy().shape)
        
        image = image.cuda()
        self.model = self.model.cuda()
        x = Variable(image.clone(), requires_grad=True)
        res = self.model.forward(x)
        label = pred_correct

        while label.item() == pred_correct.item() and i < params['max_iter']:
            min_pert = np.inf

            # zero_gradients(x)

            res[0, label[0]].backward(retain_graph=True)
            # grad_target = x.grad.data.cpu().numpy().copy()
            grad_target = copy.deepcopy(x.grad.data)

            for k in range(0, num_classes):
                if k == label[0]:
                    continue

                zero_gradients(x)

                res[0, k].backward(retain_graph=True)
                # grad_k = x.grad.data.cpu().numpy().copy()
                grad_k = copy.deepcopy(x.grad.data)

                w_k = (grad_k - grad_target).data.cpu().numpy().copy()
                f_k = (res[0, k] - res[0, label[0]]).data.cpu().numpy().copy()

                pert_k = abs(f_k) / np.linalg.norm(w_k.flatten())
                if pert_k < min_pert:
                    min_pert = pert_k
                    min_w = w_k
                
            r_i = (min_pert + params['eps']) * min_w / np.linalg.norm(min_w)
            x = Variable(x + (1 + params['overshoot']) * torch.from_numpy(r_i).cuda(), requires_grad=True)

            res = self.model.forward(x)
            label = res.max(1, keepdim=True)[1]

            i += 1
        
        # print((x - image).data)
        
        return x.data, (i < max_iter - 1)
