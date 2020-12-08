# C&W L0 Attacker
# Reference: https://github.com/carlini/nn_robust_attacks/blob/master/l0_attack.py
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
        output = torch.argmax(output[0])
    return output == target


class CWLiAttacker(Attacker):
    def __init__(self, dataset, model=models.vgg16(pretrained=True)):
        super().__init__(dataset, model)

    @staticmethod
    def generate_perturbed_image(self, image, pred, pred_correct, params):  
        num_classes = len(pred[0])
        max_iter = params['max_iter']
        c = params['const_initial_value']
        c_max = params['const_max_value']
        target = int(params['target'])

        image_prev = image.clone().cuda()
        total_attack = None

        tau = 1.0

        finish_flag_tau = False
        while tau > (1.0 / 256.0):
            delta = torch.from_numpy(np.random.rand(*list(image.shape))).cuda()
            delta.requires_grad = True
            optimizer = torch.optim.Adam([delta], lr=params['lr'])
            finish_flag = False
            while c < c_max:
                for iteration in tqdm(range(max_iter), ascii=True, desc="Iteration(c = %s, tau=%s)" % (c, tau)):
                    x_orig = image.clone().cuda()
                    x_prev = image_prev.clone().cuda()
                    w = transform_inv(x_prev).cuda()
                    x_new = transform(w + delta)

                    loss_a = torch.sum(torch.maximum(torch.tensor(0).cuda(), torch.abs(x_new - x_orig) - tau))

                    optimizer.zero_grad()

                    output = self.model(x_new.type(torch.cuda.FloatTensor))
                    one_hot = torch.from_numpy(np.zeros(list(output.shape))).cuda()
                    one_hot[0, target] = 1
                    confidence_target = torch.sum(one_hot * output)
                    confidence_without_target = torch.max((1 - one_hot) * output - one_hot * 10000)
                    # print(confidence_target, confidence_without_target)
                    loss_b = torch.maximum(torch.tensor(0).cuda(), confidence_without_target - confidence_target).cuda()

                    loss = loss_a + c * loss_b
                    loss.backward(retain_graph=True)
                    optimizer.step()

                    if loss < 0.0001 * c and params['early_abort']:
                        if compare(output, target, params):
                            image_prev = total_attack = x_new.clone()
                            finish_flag = True
                            if params['reduce_const']:
                                c /= 2
                            tau_max = torch.max(torch.abs(x_new - x_orig)).item()
                            if tau_max < tau:
                                tau = tau_max
                            else:
                                finish_flag_tau = True
                            tau *= params['decrease_factor']
                            break

                if finish_flag:
                    break

                c = c * params['const_factor']
            
            if not finish_flag:
                tau *= params['decrease_factor']

            if finish_flag_tau:
                break

        return total_attack, True
