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
        output = output.clone()
        # print(output)
        output[0, target] -= params['k']
        output = torch.argmax(output[0])
        # print(output)
    return output == target


class CWL0Attacker(Attacker):
    def __init__(self, dataset, model=models.vgg16(pretrained=True)):
        super().__init__(dataset, model)

    @staticmethod
    def generate_perturbed_image(self, image, pred, pred_correct, params):  
        num_classes = len(pred[0])
        max_iter = params['max_iter']
        c = params['const_initial_value']
        c_max = params['const_max_value']
        target = int(params['target'])

        mask = torch.from_numpy(np.ones(list(image.shape))).cuda()
        image_prev = image.clone().cuda()
        total_attack = None

        while True:
            delta = torch.from_numpy(np.random.rand(*list(image.shape))).cuda()
            delta.requires_grad = True
            optimizer = torch.optim.Adam([delta], lr=params['lr'])
            finish_flag = False
            while c < c_max:
                for iteration in tqdm(range(max_iter), ascii=True, desc="Iteration(c = %s)" % c):
                    x_orig = image.clone().cuda()
                    x_prev = image_prev.clone().cuda()
                    w = transform_inv(x_prev).cuda()
                    x_new = transform(w + delta) * mask + x_orig * (1 - mask)

                    loss_a = torch.norm(x_new - x_orig, p=2).cuda()

                    optimizer.zero_grad()

                    output = self.model(x_new.type(torch.cuda.FloatTensor))
                    one_hot = torch.from_numpy(np.zeros(list(output.shape))).cuda()
                    one_hot[0, target] = 1
                    confidence_target = torch.sum(one_hot * output)
                    confidence_without_target = torch.max((1 - one_hot) * output - one_hot * 10000)
                    # print(confidence_target, confidence_without_target)
                    loss_b = torch.maximum(torch.tensor(0).cuda(), confidence_without_target - confidence_target + 0.01).cuda()

                    loss = loss_a + c * loss_b
                    loss.backward(retain_graph=True)
                    optimizer.step()

                    if loss_b < 0.0001 and params['early_abort']:
                        image_prev = total_attack = x_new.clone()
                        delta_f = torch.sum(x_new[0] - x_orig[0], axis=0) * torch.sum(torch.abs(delta.grad[0]), axis=0)
                        delta_f = delta_f.flatten()
                        mask = mask.reshape((image.shape[1], image.shape[2] * image.shape[3]))
                        maximum_del = image.shape[2] * image.shape[3] - torch.sum(torch.all(torch.abs(x_new[0] - x_orig[0]) < 0.0001, axis=0))
                        total_del = 0
                        for idx in torch.argsort(delta_f):
                            if torch.all(mask.type(torch.bool), axis=0)[idx]:
                                mask[:, idx] = 0
                                total_del += 1
                                if delta_f[idx] > 0.01:
                                    break
                                if total_del >= 0.3 * (maximum_del ** 0.5):
                                    break
                        mask = mask.reshape((1, image.shape[1], image.shape[2], image.shape[3]))
                        finish_flag = True
                        if params['reduce_const']:
                            c /= 2
                        break

                if finish_flag:
                    break

                c = c * params['const_factor']

            if finish_flag:
                break

        return total_attack, True
