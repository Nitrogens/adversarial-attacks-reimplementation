# Universal Attacker
# Reference: https://github.com/LTS4/universal/blob/master/python/universal_pert.py
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

from .Attacker import Attacker, device
from .DeepFoolAttacker import DeepFoolAttacker


def proj(v, xi, p):
    if p == 2:
        v = v * min(1, xi / np.linalg.norm(v.flatten()))
    elif p == np.inf:
        v = np.sign(v) * np.min(abs(v), xi)
    return v


class UniversalAttacker(Attacker):
    def __init__(self, dataset, model=models.vgg16(pretrained=True)):
        super().__init__(dataset, model)

    def generate_universal_pert(self, params):
        i = 0
        max_iter_uni = 50
        if params['max_iter_uni'] is not None:
            max_iter_uni = params['max_iter_uni']
        if params['p'] == 'inf':
            params['p'] = np.inf
        elif params['p'] == '2':
            params['p'] = 2
        num_images = len(self.loader)

        v = None
        fooling_rate = 0

        original_image_list = torch.tensor([]).to(device)
        for (data, _) in self.loader:
            original_image_list = torch.cat((original_image_list, data.to(device)), dim=0)

        while fooling_rate <= 1 - params['delta'] and i < max_iter_uni:
            j = 0

            for (data, target) in tqdm(self.loader, ascii=True, desc="Generating pert"):
                if v is None:
                    v = np.zeros(data.cpu().numpy().shape)
                data, target = data.to(device), target.to(device)
                x = Variable(data, requires_grad=True)
                output = self.model.forward(x)
                pred = output.max(1, keepdim=True)[1]

                x_v = Variable(x + torch.from_numpy(v).cuda(), requires_grad=True).type(torch.FloatTensor).cuda()
                output_v = self.model.forward(x_v)
                pred_v = output_v.max(1, keepdim=True)[1]

                if pred == pred_v:
                    perturbed_image, is_converged = DeepFoolAttacker.generate_perturbed_image(self, x_v, output_v, target, params)
                    delta = perturbed_image - x_v

                    if is_converged:
                        v = v + delta.data.cpu().numpy()
                        v = proj(v, params['xi'], params['p'])
                
                j += 1
                if j == params['sample_size']:
                    break
            
            i += 1

            # print(v)

            perturbed_image_list = torch.tensor([]).to(device)
            for (data, target) in self.loader:
                data_pert = (data + torch.from_numpy(v)).to(device)
                perturbed_image_list = torch.cat((perturbed_image_list, data_pert), dim=0)

            # print(perturbed_image_list.shape)

            batch_size = 100
            if params['batch_size'] is not None:
                batch_size = params['batch_size']
            num_batches = np.int(np.ceil(np.float(num_images / np.float(batch_size))))

            pred_original = np.zeros((num_images))
            pred_perturbed = np.zeros((num_images))
            for batch_idx in tqdm(range(num_batches), ascii=True, desc="Validating"):
                start, end = batch_idx * batch_size, (batch_idx + 1) * batch_size
                end = min(end, num_images)
                output_original = self.model.forward(original_image_list[start:end]).data
                pred_original[start:end] = output_original.max(1, keepdim=True)[1].flatten().data.cpu().numpy().copy()
                output_perturbed = self.model.forward(perturbed_image_list[start:end].type(torch.cuda.FloatTensor)).data
                pred_perturbed[start:end] = output_perturbed.max(1, keepdim=True)[1].flatten().data.cpu().numpy().copy()

            fooling_rate = float(np.sum(pred_original != pred_perturbed) / float(num_images))
            print('[Fooling Rate] %s' % fooling_rate)

        return torch.from_numpy(v).cuda().data
