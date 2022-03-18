import os
import torch
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter


class TensorboardSummary(object):
    def __init__(self, directory, args = None):
        self.writer = SummaryWriter(directory)
        
    def vis_image(self, image, seg_mask, seg_loss_mask, pre_seg, global_step, split = '', image_label = None):
        num = 4 if len(image) > 4 else len(image)
        grid_image = make_grid(image[:num].clone().cpu().data, num, normalize=True)
        self.writer.add_image(f'{split} Image', grid_image, global_step)

        grid_image = make_grid(seg_mask[:num].detach().cpu().numpy(), num, normalize=False, range=(0, 1))
        self.writer.add_image(f'{split} seg_mask', grid_image, global_step)
        
        grid_image = make_grid(seg_loss_mask[:num].detach().cpu().numpy(), num, normalize=False, range=(0, 2))
        self.writer.add_image(f'{split} seg_loss_mask', grid_image, global_step)
        
        grid_image = make_grid(pre_seg[:num].detach().cpu().numpy(), num, normalize=False, range=(0, 1))
        self.writer.add_image(f'{split} pre_seg', grid_image, global_step)

    def add_scalar(self, tag, item, item_0):
        self.writer.add_scalar(tag, item, item_0)
