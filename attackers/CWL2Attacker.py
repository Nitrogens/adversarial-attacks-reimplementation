# C&W L2 Attacker
# Reference: https://github.com/carlini/nn_robust_attacks/blob/master/l2_attack.py
# Written by Nitrogens

import torch
import copy
import numpy as np

import torch.nn.functional as F

import torchvision.models as models
import torchvision.transforms as transforms

from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients

from tqdm import tqdm

from .Attacker import Attacker


def transform_inv(image, val_max=3.0, val_min=-3.0):
    fac_mul = (val_max - val_min) / 2.0
    fac_add = (val_max + val_min) / 2.0
    return torch.atanh((image - fac_add) / fac_mul * 0.999999)


def transform(image, val_max=3.0, val_min=-3.0):
    fac_mul = (val_max - val_min) / 2.0
    fac_add = (val_max + val_min) / 2.0
    return torch.tanh(image) * fac_mul + fac_add


def compare(output, target, params):
    if not isinstance(output, int):
        output = output.clone()
        # print(output)
        output[0, target] -= params['k']
        output = torch.argmax(output[0])
        # print(output)
    return output == target


class CWL2Attacker(Attacker):
    def __init__(self, dataset, model=models.vgg16(pretrained=True)):
        super().__init__(dataset, model)

    @staticmethod
    def generate_perturbed_image(self, image, pred, pred_correct, params):  
        num_classes = len(pred[0])
        max_iter = params['max_iter']
        max_iter_binary_search = params['max_iter_binary_search']
        c = params['const_initial_value']
        target = int(params['target'])
        k = params['k']

        left, right = 0, 1e10
        total_min_loss = 1e10
        total_min_loss_label = -1
        total_attack = image.clone()

        for binary_search_iter in tqdm(range(max_iter_binary_search), ascii=True, desc="Binary search"):
            min_loss = 1e10
            min_loss_label = -1
            loss_prev = np.inf
            delta = torch.from_numpy(np.random.rand(*list(image.shape))).cuda()
            # print(delta)
            delta.requires_grad = True
            optimizer = torch.optim.Adam([delta], lr=params['lr'])
            for iteration in tqdm(range(max_iter), ascii=True, desc="Iteration"):
                x = image.clone().cuda()
                w = transform_inv(x)
                x_pert = transform(w + delta)
                # print(x, w, x_pert)
                loss_a = torch.norm(x_pert - transform(w), p=2).cuda()

                optimizer.zero_grad()

                output = self.model(x_pert.type(torch.cuda.FloatTensor))
                one_hot = torch.from_numpy(np.zeros(list(output.shape))).cuda()
                one_hot[0, target] = 1
                confidence_target = torch.sum(one_hot * output)
                confidence_without_target = torch.max((1 - one_hot) * output - one_hot * 10000)
                # print(confidence_target, confidence_without_target)
                loss_b = torch.maximum(torch.tensor(0).cuda(), confidence_without_target - confidence_target + k).cuda()

                loss = loss_a + c * loss_b
                loss.backward()
                optimizer.step()

                # if iteration % (max_iter // 20) == 0:
                #     print("\n[LOSS] %s + %s * %s = %s" % (loss_a, c, loss_b, loss))
                # print("\n[LOSS] %s + %s * %s = %s" % (loss_a.item(), c, loss_b.item(), loss.item()))

                if params['early_abort'] is True:
                    if loss > loss_prev * 0.9999:
                        break
                    loss_prev = loss
                
                if loss < min_loss and compare(output, target, params):
                    min_loss = loss
                    min_loss_label = int(torch.argmax(output[0]).item())
                if loss < total_min_loss and compare(output, target, params):
                    total_min_loss = loss
                    total_min_loss_label = int(torch.argmax(output[0]).item())
                    total_attack = x_pert.clone()

            if compare(min_loss_label, target, params) and min_loss_label != -1:
                # Success
                right = min(c, right)
                if right < 1e9:
                    c = (left + right) / 2
            else:
                # Fail
                left = max(c, left)
                if right < 1e9:
                    c = (left + right) / 2
                else:
                    c *= 10

        return total_attack, True
