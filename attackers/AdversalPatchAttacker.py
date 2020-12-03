# Adversal Patch Attacker
# Reference: https://github.com/zhaojb17/Adversarial_Patch_Attack
# Written by Nitrogens

import torch
import copy
import cv2
import numpy as np

import torch.nn.functional as F

import torchvision.models as models
import torchvision.transforms as transforms

from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients

from tqdm import tqdm

from .Attacker import Attacker, device


std = [0.485, 0.456, 0.406]
mean = [0.229, 0.224, 0.225]


def patch_init(image_size, params):
    _, C, N, M = *image_size,
    patch_size = int(np.sqrt(params['noise_percentage'] * N * M))
    return np.random.rand(C, patch_size, patch_size)


def mask_generator(image_size, patch, params):
    _, C, N, M = *image_size,
    _, N_patch, M_patch = *patch.shape,
    n_rot = np.random.randint(low=0, high=4)
    for c in range(C):
        patch[c] = np.rot90(patch[c], n_rot)
    x = np.random.randint(low=0, high=N-N_patch)
    y = np.random.randint(low=0, high=M-M_patch)
    patch_inserted = np.zeros((C, N, M))
    patch_inserted[:, x:x+N_patch, y:y+M_patch] = patch
    mask = patch_inserted.copy()
    mask[mask != 0] = 1
    return x, y, mask, patch_inserted


def patch_update(image, patch_inserted, mask, model, params):
    p_thr = 0.9
    if params['p_thr'] is not None:
        p_thr = params['p_thr']
    max_iter = 100
    if params['max_iter'] is not None:
        max_iter = params['max_iter']
    p_current = 0.0
    i = 0

    image_pert = \
    torch.mul(torch.from_numpy(mask).cuda(), torch.from_numpy(patch_inserted).cuda()) +\
    torch.mul(torch.from_numpy(1 - mask).cuda(), image)
    image_pert = image_pert.type(torch.cuda.FloatTensor)
    image_pert.requires_grad = True

    while p_current < p_thr and i < max_iter:
        output = model(image_pert)

        softmax_value = torch.nn.functional.log_softmax(output, dim=1)[0][params['target']]
        softmax_value.backward()
        grad = image_pert.grad.clone()
        zero_gradients(image_pert)

        patch_inserted = patch_inserted + (params['lr'] * grad[0]).data.cpu().numpy()
        patch_inserted = np.clip(patch_inserted, -3, 3)

        image_pert = \
        torch.mul(torch.from_numpy(mask).cuda(), torch.from_numpy(patch_inserted).cuda()) +\
        torch.mul(torch.from_numpy(1 - mask).cuda(), image)
        image_pert = torch.clamp(image_pert, min=-3, max=3)
        image_pert = image_pert.type(torch.cuda.FloatTensor)
        image_pert.requires_grad = True

        output_pert = model(image_pert)
        p_current = torch.nn.functional.log_softmax(output_pert, dim=1)[0][params['target']]

        i += 1
    
    return image_pert, patch_inserted


class AdversalPatchAttacker(Attacker):
    def __init__(self, dataset, model=models.vgg16(pretrained=True)):
        super().__init__(dataset, model)

    def generate_universal_pert(self, params):
        num_epoch = 20
        if params['num_epoch'] is not None:
            num_epoch = params['num_epoch']
        if params['noise_percentage'] is None:
            params['noise_percentage'] = 0.1
        noise_percentage = params['noise_percentage']
        num_images = len(self.loader)

        patch = None

        for epoch in tqdm(range(num_epoch), ascii=True, desc="Epoch"):
        # for epoch in range(num_epoch):
            num_success = 0
            for (data, target) in tqdm(self.loader, ascii=True, desc="Generating pert"):
            # for (data, target) in self.loader:
                if patch is None:
                    patch = patch_init(list(data.shape), params)
                data, target = data.to(device), target.to(device)
                output = self.model(data)
                pred = output.max(1, keepdim=True)[1]
                if pred[0] == params['target']:
                    continue
                x, y, mask, patch_inserted = mask_generator(list(data.shape), patch, params)
                image_pert, patch_inserted = patch_update(data, patch_inserted, mask, self.model, params)
                output = self.model(image_pert)
                pred = output.max(1, keepdim=True)[1]
                if pred[0] == params['target']:
                    num_success += 1
                patch = patch_inserted[:, x:x+patch.shape[1], y:y+patch.shape[2]]
            patch_output = patch.copy()
            for idx in range(3):
                patch_output[idx] = patch_output[idx] * std[idx] + mean[idx]
            patch_output = patch_output * 255
            cv2.imwrite('output/AdversalPatchAttacker/patch_output_%d.png' % epoch, cv2.merge([patch_output[idx] for idx in range(2, -1, -1)]))
            print("[TRAIN] Epoch: %s, Successful rate: %s/%s" % (epoch, num_success, num_images))

        return_image = torch.tensor(np.zeros(tuple(data.shape)))
        return_image[:, x:x+data.shape[1], y:y+data.shape[2]] = patch

        return return_image.cuda().data
