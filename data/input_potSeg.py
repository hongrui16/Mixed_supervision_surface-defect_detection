import os
import numpy as np
import scipy.misc as m
from PIL import Image
# from torch.utils import data
from torchvision import transforms
# from dataloaders import joint_transforms as jo_trans
# from dataloaders.synthesis.synthesize_sample import AddNegSample as RandomAddNegSample

import logging
from os import listdir
from os.path import splitext
import torch
from torch.utils.data import Dataset
import cv2
import sys
import albumentations as albu
import random
from util.util import *
import data.custom_transforms as tr
from data.dataset import Dataset
from config import Config

class CustomPotSeg(Dataset):

    def __init__(self, cfg: Config, args, split="train"):
        self.cfg = cfg
        self.args = args
        self.root = args.DATASET_PATH
        self.split = split
        self.ignore_index = args.ignore_index
        self.image_size = (self.cfg.INPUT_WIDTH, self.cfg.INPUT_HEIGHT)
        self.grayscale: bool = self.cfg.INPUT_CHANNELS == 1

        self.num_negatives_per_one_positive: int = 1
        self.frequency_sampling: bool = self.cfg.FREQUENCY_SAMPLING and self.split == 'train'


        # print('args.ignore_index', args.ignore_index)
        self.spatial_trans, self.pixel_trans = self.albumentations_aug()
        if not self.args.use_txtfile:
            self.base_dir = os.path.join(self.root, self.split)
            if args.testset_dir:
                self.images_base =  args.testset_dir
                self.annotations_base = ''
            else:
                self.images_base = os.path.join(self.base_dir, 'image')
                self.annotations_base = os.path.join(self.base_dir, 'label')
            # print('annotations_base', self.annotations_base)
            # self.ids = [splitext(file)[0] for file in listdir(self.images_base) if not file.startswith('.')]
            # self.img_ids = [file for file in listdir(self.images_base) if not file.startswith('.')]
            # random.shuffle(self.img_ids)
            self.img_filepaths = []
            for file in listdir(self.images_base):
                img_filepath = os.path.join(self.images_base, file)
                self.img_filepaths.append(img_filepath)
            random.shuffle(self.img_filepaths)
            
        else:
            txt_filepath = os.path.join(self.root, f'{self.split}.txt')
            self.img_filepaths = read_txt_to_list(txt_filepath)

    def __len__(self):
        # return len(self.img_ids)
        return len(self.img_filepaths)

    def __getitem__(self, index):
        # print(f'calling {__file__}, {sys._getframe().f_lineno}')
    
        img_path = self.img_filepaths[index]
        if self.grayscale:
            img = cv2.imread(img_path, 0)
            im_h, im_w = img.shape
        else:
            img = cv2.imread(img_path)
            im_h, im_w, _ = img.shape
        # print('img_path', img.shape, img_path)
        img_name = img_path.split('/')[-1]
        # lbl_path = img_path.replace('image', 'label').replace('.jpg', '.png')
        seg_mask_path = img_path.replace('.jpg', '.png')
        
        # print('tar_h, tar_w', tar_h, tar_w)
        if not os.path.exists(seg_mask_path):  # good sample
            seg_mask = np.zeros((im_h, im_w))
            is_segmented = False
        else:
            seg_mask = cv2.imread(seg_mask_path, 0)
            # seg_mask = cv2.resize(seg_mask, (tar_w, tar_h), interpolation = cv2.INTER_NEAREST)
            is_segmented = True
        # print('seg_mask.shape', seg_mask.shape)
        seg_mask, img = self.encode_segmap(seg_mask, img)
        
        if np.max(seg_mask) == np.min(seg_mask):  # good sample
            seg_loss_mask = np.zeros((im_h, im_w))
        else:
            seg_loss_mask = self.distance_transform(seg_mask, self.cfg.WEIGHTED_SEG_LOSS_MAX, self.cfg.WEIGHTED_SEG_LOSS_P)
        
        # return image, seg_mask, seg_loss_mask, is_segmented, sample_name

        sample = {'image': Image.fromarray(img), 'label': Image.fromarray(seg_mask), 
                  'seg_loss_mask': Image.fromarray(seg_loss_mask), 'is_segmented':is_segmented, 'sample_name': img_name}
        
        if self.split == "train" and self.args.use_albu:
            # print('sample', sample)
            sample = self.transform_train1(sample)
            img = np.array(sample['image'])
            seg_mask = np.array(sample['label'])        
            seg_loss_mask = np.array(sample['seg_loss_mask'])        

            img = self.pixel_trans(image=img)['image']

            #to tensor
            if self.grayscale:
                img = np.expand_dims(img, axis=0)
            else:
                img = img.transpose( (2, 0, 1) )
            # print('_img', _img.shape, '_target', _target.shape)
            img = torch.from_numpy(img).float()
            seg_mask = torch.from_numpy(seg_mask).float()            
            seg_loss_mask = torch.from_numpy(seg_loss_mask).float()            
            
            return img, seg_mask, seg_loss_mask, is_segmented, img_name

        else:
            # return image, seg_mask, seg_loss_mask, is_segmented, sample_name
        
            if self.split == 'train':
                sample = self.transform_train(sample)
                img = sample['image']
                seg_mask = sample['label']
                seg_loss_mask = sample['seg_loss_mask']
                return img, seg_mask, seg_loss_mask, is_segmented, img_name
            
            elif self.split == 'val':
                sample =  self.transform_val(sample)
                img = sample['image']
                seg_mask = sample['label']
                seg_loss_mask = sample['seg_loss_mask']
                return img, seg_mask, seg_loss_mask, is_segmented, img_name
            
            elif self.split == 'test' or self.args.testset_dir:
                sample =  self.transform_test(sample)
                img = sample['image']
                seg_mask = sample['label']
                seg_loss_mask = sample['seg_loss_mask']
                return img, seg_mask, seg_loss_mask, is_segmented, img_name
            
    def encode_segmap(self, mask, img = None):
        '''
        label_dict = {'lasi_heavy': 11, 'lasi_medium':12, 'lasi_slight':13,
        'gengshang_heavy':21, 'gengshang_medium':22, 'gengshang_slight':23,  
        'gengshi_heavy':31, 'gengshi_medium':32, 'gengshi_slight':33,
        'shayan_heavy':41, 'shayan_medium':42, 'shayan_medium':43,
        'huahen_heavy':51, 'huahen_medium':52, 'huahen_medium':53,
        'zhoubian_heavy':61, 'zhoubian_medium':62, 'zhoubian_medium':63,
        'bowen_heavy':71, 'bowen_medium':72, 'bowen_medium':73,
        'youwu_heavy':81, 'youwu_medium':82, 'youwu_medium':83,
        }

        '''        
        if mask.any() > 0:
            mask_bk = mask.copy()
            mask[mask_bk == 13] = 0
            mask[mask_bk == 23] = 0
            mask[mask_bk == 33] = 0
            mask[mask_bk == 43] = 0
            mask[mask_bk == 53] = 0
            if self.args.pot_train_mode == 1: #???????????????
                mask[mask_bk >= 61] = 0
                mask[mask_bk>0] = 1
            elif self.args.pot_train_mode == 2: #???????????????,??????????????????
                mask[mask_bk >= 41] = 0
                mask[mask_bk>0] = 1
            else:
                print('please specify pot_train_mode')
                return None, None
            
            if self.args.de_ignore_index:  ## make sure ignore_index have already been defaulted
                mask[mask_bk==self.args.ignore_index] = 0 #255
            else:
                mask[mask_bk==self.args.ignore_index] = self.args.ignore_index #255
            
            
        return mask, img.astype(np.uint8)

# ignore_loss_index
# ignore_index
# rotate_degree
# base_size
# crop_size

    def transform_train(self, sample):
        composed_transforms = transforms.Compose([
            # tr.RandomCutPostives(size=self.args.base_size, args = self.args, split = self.split),
            tr.ShortEdgePad(size=self.args.base_size, args = self.args),
            tr.RandomCrop(args=self.args),
            tr.RandomScaleRemainSize(args=self.args),
            # tr.ShortEdgeCrop(hw_ratio= self.args.hw_ratio, args = self.args),

            # tr.RandomAddNegSample(args = self.args),
            tr.RandomHorizontalFlip(self.args),
            tr.RandomVerticalFlip(self.args),            
            # tr.RandomRotate(degree = self.args.rotate_degree, args = self.args),
            tr.RandomGaussianBlur(),
            # tr.FixScaleCrop(crop_size=self.args.crop_size, args = self.args),
            # tr.RandomHorizontalFlipImageMask(self.args),
            # tr.RandomShadows(args = self.args),
            tr.FixedResize(size=self.args.base_size, args = self.args, cfg = self.cfg),
            tr.DeIgnoreIndex(self.args),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):
        composed_transforms = transforms.Compose([
            # tr.RandomCutPostives(size=self.args.base_size, args = self.args, split = self.split),
            tr.ShortEdgePad(size=self.args.base_size, args = self.args),
            tr.RandomCrop(args=self.args),
            tr.RandomScaleRemainSize(args=self.args),
            # tr.ShortEdgeCrop(hw_ratio= self.args.hw_ratio, args = self.args),
            # tr.RandomAddNegSample(args = self.args),
            # tr.FixScaleCrop(crop_size=self.args.crop_size, args = self.args),
            # tr.LimitResize(size=self.args.max_size, args = self.args),
            tr.FixedResize(size=self.args.base_size, args = self.args, cfg = self.cfg),
            tr.DeIgnoreIndex(self.args),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])
        return composed_transforms(sample)

    def transform_test(self, sample):
        composed_transforms = transforms.Compose([
            # tr.RandomCutPostives(size=self.args.base_size, args = self.args, split = self.split),
            tr.ShortEdgePad(size=self.args.base_size, args = self.args),
            tr.RandomCrop(args=self.args),
            # tr.RandomScaleRemainSize(args=self.args),
            # tr.RandomAddNegSample(args = self.args),
            # tr.CenterPadAndCrop(size=self.args.base_size, args = self.args),
            # tr.LimitResize(size=self.args.max_size),
            tr.FixedResize(size=self.args.base_size, args = self.args, cfg = self.cfg),
            tr.DeIgnoreIndex(self.args),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])
        return composed_transforms(sample)

    def transform_train1(self, sample):
        composed_transforms = transforms.Compose([
            # tr.RandomCutPostives(size=self.args.base_size, args = self.args, split = self.split),
            tr.ShortEdgePad(size=self.args.base_size, args = self.args),
            # tr.ShortEdgeCrop(hw_ratio= self.args.hw_ratio, args = self.args),
            tr.RandomCrop(args=self.args),
            tr.RandomScaleRemainSize(args=self.args),
            tr.RandomHorizontalFlip(self.args),
            tr.RandomVerticalFlip(self.args),
            # tr.RandomAddNegSample(args = self.args),
            # tr.RandomRotate(degree = self.args.rotate_degree, args = self.args),
            # tr.RandomShadows(args = self.args),
            # tr.RandomHorizontalFlipImageMask(self.args),
            tr.RandomGaussianBlur(),
            # tr.FixScaleCrop(crop_size=self.args.crop_size, args = self.args),
            tr.FixedResize(size=self.args.base_size, args = self.args, cfg = self.cfg),
            tr.DeIgnoreIndex(self.args),
            ])

        return composed_transforms(sample)

    def albumentations_aug(self):
        args = self.args

        spatial_trans = albu.Compose([
            #albu.RandomSizedCrop(args.base_size, scale=(0.6, 1.0), ratio=(0.75, 1.3333333333333333)),
            albu.SmallestMaxSize(args.base_size, p=1.),
            albu.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=20, interpolation=1, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=args.ignore_index, p=0.5),
            #albu.ShiftScaleRotate(shift_limit=0.0, scale_limit=0.0, rotate_limit=90, interpolation=1, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=args.ignore_index, p=0.5),
            albu.PadIfNeeded(min_height=args.base_size, min_width=args.base_size, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=args.ignore_index, p=1.),
            #albu.RandomCrop(args.base_size, args.base_size, p=1.),
            albu.CenterCrop(args.base_size, args.base_size, p=1.),
            albu.Flip(p=0.5),              
            albu.OneOf([
                #albu.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
                albu.GridDistortion(p=0.5),
                albu.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=0.5)                  
                ], p=0.3),
            #albu.CoarseDropout (max_holes=8, max_height=int(args.base_size*0.1), max_width=int(args.base_size*0.1), fill_value=0, mask_fill_value=args.ignore_index, p=0.3)
            albu.CoarseDropout (max_holes=32, max_height=20, max_width=20, fill_value=255, mask_fill_value=0, p=0.3)
            ], p=1.)


        pixel_trans = albu.Compose([
            #albu.OneOf([
            #    albu.CLAHE(clip_limit=2, p=.5),
            #    albu.Sharpen(p=.25),
            #    ], p=0.35),
            albu.RandomBrightnessContrast(p=.4),
            albu.OneOf([
                # albu.GaussNoise(p=.2),
                albu.ISONoise(p=.2),
                albu.ImageCompression(quality_lower=75, quality_upper=100, p=.4)
                ], p=.4),
            albu.RGBShift(p=.4),
            albu.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=12, val_shift_limit=8, p=.4),
            #albu.ToGray(p=.2),
            albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1.0),
            ], p=1.)
        return spatial_trans, pixel_trans

    # skip boundary pixels to handle nosiy annotation
    def skip_boundary(self, mask):
        mat = mask >= 1
        mat = mat.astype(np.uint8)*255        
        edges = cv2.Canny(mat,240,240)
        edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)), iterations=1) 
        indices = edges == 255 
        mask[indices] = self.ignore_index

        return mask


if __name__ == '__main__':
    # from dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import argparse
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

    parser = argparse.ArgumentParser()
    parser.add_argument('-im', '--input_dir', type=str, default=None)
    parser.add_argument('-om', '--output_dir', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=16,
                    metavar='N', help='input batch size for \
                            training (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=8,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--hw_ratio', type=float, default=1.25)
    parser.add_argument('--ignore_index', type=int, default=255)

    parser.add_argument('--base_size', type=int, default=640)
    parser.add_argument('--crop_size', type=int, default=640)
    parser.add_argument('--max_size', type=int, default=1080)


    parser.add_argument('--rotate_degree', type=int, default=15)
    parser.add_argument('--dataset', type=str, default='basicDataset')
    parser.add_argument('--dataset_dir', type=str, default=None, help='dataset dir')
    parser.add_argument('--testValTrain', type=int, default=-2, help='-1: infer, 0: test, 1: testval, 2: train, 3: trainval, 4: trainvaltest')
    parser.add_argument('--testset_dir', type=str, default=None, help='input test image dir')
    parser.add_argument('--testOut_dir', type=str, default=None, help='test image output dir')
    parser.add_argument('--distinguish_left_right_semantic', action='store_true', default=True, help='distinguish left and right rail semantic segmentation')
    parser.add_argument('--skip_boundary', action='store_true', default=False, help="skip boundary pixel to handle annotation noise")

    parser.add_argument('--use_albu', action='store_true', default=False, help="indicate wheather to use albumentation in training phase for data augmentation")


    args = parser.parse_args()

    output_dir = args.output_dir
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    root = args.input_dir

    # basicDataset_train = BasicDataset(args, root, split="train")
    # basicDataset_test = BasicDataset(args, root, split="test")

    # train_loader = DataLoader(basicDataset_train, batch_size=2, shuffle=False, num_workers=2)
    # test_loader = DataLoader(basicDataset_test, batch_size=2, shuffle=False, num_workers=2)

    # def save_img_mask(loader):
    #     for ii, sample in enumerate(loader):
    #         if ii == 3:
    #             break
    #         batch_size = sample["image"].size()[0]
    #         # print('batch_size: ', batch_size)
    #         for jj in range(batch_size):

    #             img = sample['image'].numpy()
    #             gt = sample['label'].numpy()
    #             img_name =  sample['img_name']
    #             img_name_perfix = img_name.split('.')[0]
    #             segmap = np.array(gt[jj]).astype(np.uint8)
    #             # segmap = decode_segmap(segmap, dataset='cityscapes')
    #             img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
    #             img_tmp *= (0.229, 0.224, 0.225)
    #             img_tmp += (0.485, 0.456, 0.406)
    #             img_tmp *= 255.0
    #             img_tmp = img_tmp.astype(np.uint8)
                
    #             # plt.figure()
    #             # plt.title('display')
    #             # plt.subplot(211)
    #             # plt.imshow(img_tmp)
    #             # plt.subplot(212)
    #             # plt.imshow(segmap)
    #             # ax = plt.subplot(4, batch_size*2, ii*batch_size*2 + 2*jj+1), plt.imshow(img_tmp), plt.title(f'img_{ii*batch_size + jj}'), plt.xticks([]), plt.yticks([])
    #             # ax = plt.subplot(4, batch_size*2, ii*batch_size*2 + 2*jj+2), plt.imshow(segmap*60), plt.title(f'mask_{ii*batch_size + jj}'), plt.xticks([]), plt.yticks([])
    #             # if segmap.ndim == 2:
    #             #     plt.gray()

    #             cv2.imwrite(os.path.join(output_dir, f'{img_name_perfix}.jpg'), img_tmp)
    #             cv2.imwrite(os.path.join(output_dir, f'{img_name_perfix}.png'), segmap*60)

    # save_img_mask(train_loader)
    # save_img_mask(test_loader)
    # # plt.show()
    # plt.show(block=True)

