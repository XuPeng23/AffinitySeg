import torch
import torch.nn as nn


def save_image_tensor(input_tensor: torch.tensor, filename):
    from torchvision import utils as vutils
    input_tensor = input_tensor.clone().detach()
    input_tensor = input_tensor.to(torch.device('cpu'))
    vutils.save_image(input_tensor, filename)

def wbce(pred, gt):
    pos = torch.eq(gt, 1).float()
    neg = torch.eq(gt, 0).float()
    num_pos = torch.sum(pos)
    num_neg = torch.sum(neg)
    num_total = num_pos + num_neg
    alpha_pos = num_neg / num_total
    alpha_neg = num_pos / num_total
    weights = alpha_pos * pos + alpha_neg * neg
    return nn.functional.binary_cross_entropy_with_logits(pred, gt, weights)


__all__ = ['save_image_tensor', 'wbce']
