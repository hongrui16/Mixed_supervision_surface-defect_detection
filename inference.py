import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from models import SegDecNet
import numpy as np
import os
from torch import nn as nn
import torch
import utils
import pandas as pd

import random
import cv2
from config import Config
from pathlib import Path
import contextlib
import glob
import logging
import math
import os
import platform
import random
import re
import shutil
import signal
import time
import sys
from tqdm import tqdm
from timeit import default_timer as timer
import argparse
from config import Config
from util.metrics import Evaluator, EvaluatorForeground

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


from torch.utils.tensorboard import SummaryWriter

LVL_ERROR = 10
LVL_INFO = 5
LVL_DEBUG = 1

LOG = 1  # Will log all mesages with lvl greater than this
SAVE_LOG = True

WRITE_TENSORBOARD = True


class End2EndInfer:
    def __init__(self, cfg: Config, args = None):
        self.args = args
        self.cfg: Config = cfg
        self.run_name = 'inferece'
        if self.cfg.REPRODUCIBLE_RUN:
            self._log("Reproducible run, fixing all seeds to:1337", LVL_DEBUG)
            np.random.seed(1337)
            torch.manual_seed(1337)
            random.seed(1337)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        self.device = self._get_device()
        self.model = self._get_model().to(self.device)
        self.resume_model(self.model, args.MODEL_PATH)
        self.set_dec_gradient_multiplier(self.model, 0.0)
        self.model.eval()
        # for k,v in self.model.state_dict().items():
        #     print(k, v.shape)

    def _log(self, message, lvl=LVL_INFO):
        n_msg = f"{self.run_name} {message}"
        if lvl >= LOG:
            print(n_msg)

    def forward(self, image, img_name=None, output_dir = None, label = None, resize = True, result = [], evaluator = None, thres = None):
        # self.model.eval()
        dsize = self.cfg.INPUT_WIDTH, self.cfg.INPUT_HEIGHT
        label[label > 0] = 1
        if not label is None:
            is_pos = True if label.any() > 0 else False    
            if resize:    
                label = cv2.resize(label, dsize, interpolation=cv2.INTER_NEAREST)
            
        image = self.preprocess_np(image, self.device, resize)

        with torch.no_grad():
            prediction, output_seg_mask = self.model(image)
        pred_seg = nn.Sigmoid()(output_seg_mask)
        prediction = nn.Sigmoid()(prediction)

        prediction = prediction.item()
        image = image.detach().cpu().numpy()
        pred_seg = pred_seg.detach().cpu().numpy()
        image = np.squeeze(image)
        pred_seg = np.squeeze(pred_seg)
        pre_seg_bk = pred_seg.copy()
        if not evaluator is None and not thres is None:
            # print('image.shape, label.shape, pred_seg.shape', image.shape, label.shape, pred_seg.shape)
            h, w = label.shape
            pred_seg = cv2.resize(pred_seg, [w, h], interpolation=cv2.INTER_NEAREST)
            pred_seg[pred_seg >= thres] = 1
            pred_seg[pred_seg < thres] = 0
            evaluator.add_batch(label.astype(int), pred_seg.astype(int))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        utils.plot_sample(img_name.split('.')[0], image, pre_seg_bk, seg_mask = label, decision=prediction, save_dir = output_dir, is_pos = is_pos)

        result.append((prediction, None, None, is_pos, img_name))

    def resume_model(self, model, model_path=None):
        if model_path:
            model.load_state_dict(torch.load(model_path))
            self._log(f"Loading model state from {model_path}")

        elif self.cfg.USE_BEST_MODEL:
            path = os.path.join(self.model_path, "best_state_dict.pth")
            model.load_state_dict(torch.load(path))
            self._log(f"Loading model state from {path}")          

        else:
            self._log("Keeping same model state")

    def _get_device(self):
        return f"cuda:{self.cfg.GPU}"
        
    def _get_model(self):
        seg_net = SegDecNet(self._get_device(), self.cfg.INPUT_WIDTH, self.cfg.INPUT_HEIGHT, self.cfg.INPUT_CHANNELS)
        return seg_net

    def preprocess_np(self, image, device, resize = True):
        assert image.ndim == 2
        if resize:
            t_h, t_w = self.cfg.INPUT_HEIGHT, self.cfg.INPUT_WIDTH
            image = cv2.resize(image, (t_w, t_h))
        image = image.astype("float32") / 255.0       
        image = np.expand_dims(image, 0)
        image = np.expand_dims(image, 0)
        image = torch.from_numpy(image).to(device)
        return image


    def set_dec_gradient_multiplier(self, model, multiplier):
        model.set_gradient_multipliers(multiplier)

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--GPU', type=int, required=True, help="ID of GPU used for training/evaluation.")
    parser.add_argument('--RUN_NAME', type=str, required=True, help="Name of the run, used as directory name for storing results.")
    parser.add_argument('--DATASET', type=str, required=True, help="Which dataset to use.")
    parser.add_argument('--DATASET_PATH', type=str, required=True, help="Path to the dataset.")

    parser.add_argument('--EPOCHS', type=int, required=True, help="Number of training epochs.")

    parser.add_argument('--LEARNING_RATE', type=float, required=True, help="Learning rate.")
    parser.add_argument('--DELTA_CLS_LOSS', type=float, required=True, help="Weight delta for classification loss.")

    parser.add_argument('--BATCH_SIZE', type=int, required=True, help="Batch size for training.")

    parser.add_argument('--WEIGHTED_SEG_LOSS', type=str2bool, required=True, help="Whether to use weighted segmentation loss.")
    parser.add_argument('--WEIGHTED_SEG_LOSS_P', type=float, required=False, default=None, help="Degree of polynomial for weighted segmentation loss.")
    parser.add_argument('--WEIGHTED_SEG_LOSS_MAX', type=float, required=False, default=None, help="Scaling factor for weighted segmentation loss.")
    parser.add_argument('--DYN_BALANCED_LOSS', type=str2bool, required=True, help="Whether to use dynamically balanced loss.")
    parser.add_argument('--GRADIENT_ADJUSTMENT', type=str2bool, required=True, help="Whether to use gradient adjustment.")
    parser.add_argument('--FREQUENCY_SAMPLING', type=str2bool, required=False, help="Whether to use frequency-of-use based sampling.")

    parser.add_argument('--DILATE', type=int, required=False, default=None, help="Size of dilation kernel for labels")

    parser.add_argument('--FOLD', type=int, default=None, help="Which fold (KSDD) or class (DAGM) to train.")
    parser.add_argument('--TRAIN_NUM', type=int, default=None, help="Number of positive training samples for KSDD or STEEL.")
    parser.add_argument('--NUM_SEGMENTED', type=int, required=True, default=None, help="Number of segmented positive  samples.")
    parser.add_argument('--RESULTS_PATH', type=str, default=None, help="Directory to which results are saved.")

    parser.add_argument('--VALIDATE', type=str2bool, default=None, help="Whether to validate during training.")
    parser.add_argument('--VALIDATE_ON_TEST', type=str2bool, default=None, help="Whether to validate on test set.")
    parser.add_argument('--VALIDATION_N_EPOCHS', type=int, default=None, help="Number of epochs between consecutive validation runs.")
    parser.add_argument('--USE_BEST_MODEL', type=str2bool, default=None, help="Whether to use the best model according to validation metrics for evaluation.")

    parser.add_argument('--ON_DEMAND_READ', type=str2bool, default=None, help="Whether to use on-demand read of data from disk instead of storing it in memory.")
    parser.add_argument('--REPRODUCIBLE_RUN', type=str2bool, default=None, help="Whether to fix seeds and disable CUDA benchmark mode.")

    parser.add_argument('--MEMORY_FIT', type=int, default=None, help="How many images can be fitted in GPU memory.")
    parser.add_argument('--SAVE_IMAGES', type=str2bool, default=None, help="Save test images or not.")


    parser.add_argument('--DEBUG', type=str2bool, default=False, help="debug code flag.")
    parser.add_argument('--TEST_N_EPOCHS', type=int, default=None, help="inteval for test")
    parser.add_argument('--testValTrain', type=int, default=-1, help="infer: 0, test:1, testval:2, train:3, trainval:4, trainvaltest:5")
    parser.add_argument('--MODEL_PATH', type=str, default='runs/STEEL/train/exp8/models/best_state_dict.pth', help="model weight file path")

    
    parser.add_argument('-im', '--input_dir', type=str, default='/home/hongrui/project/metro_pro/dataset/pot/pot_20220108_cutblock', help="input image dir")
    parser.add_argument('--output_dir', type=str, default='temp2', help="output image dir")
    


    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir 

    configuration = Config()
    configuration.merge_from_args(args)
    configuration.init_extra()

    steel_mean_valule = 89.9580
    engine = End2EndInfer(cfg=configuration, args = args)
    evaluator = EvaluatorForeground(2)

    resize = False
    thres = 0.1

    output_dir = f'{output_dir}_resize_{resize}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    run_name = f'pot_{resize}'
    img_names = os.listdir(input_dir)
    result = []
    for i, img_name in enumerate(img_names):
        if not '.jpg' in img_name:
            continue
        print(f'processing {img_name} {i+1}/{len(img_names)}')
        img_filepath = os.path.join(input_dir, img_name)
        label_filepath = img_filepath.replace('.jpg', '.png')

        img = cv2.imread(img_filepath, 0)
        # img = img.astype(int)

        '''
        add the following, AP dropped from 0.7222 to 0.704351
        # mean_img = img.mean()
        # off_set = int(mean_img) - int(steel_mean_valule)
        # img -= off_set
        '''
        label = cv2.imread(label_filepath, 0)
        engine.forward(img, img_name=img_name, output_dir = output_dir, label = label, resize=resize, result = result, evaluator = evaluator, thres = thres)
        # if i > 10:
        #     break
    mIoU = evaluator.Mean_Intersection_over_Union()
    utils.evaluate_metrics(result, output_dir, run_name, mIoU = mIoU, thres = thres)
    
    print('mIoU', mIoU)