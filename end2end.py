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
from data.dataset_catalog import get_dataset
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
from util.metrics import Evaluator, EvaluatorForeground

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
from util.summaries import TensorboardSummary

from torch.utils.tensorboard import SummaryWriter

LVL_ERROR = 10
LVL_INFO = 5
LVL_DEBUG = 1

LOG = 1  # Will log all mesages with lvl greater than this
SAVE_LOG = True

WRITE_TENSORBOARD = True


class End2End:
    def __init__(self, cfg: Config, args = None):
        self.cfg: Config = cfg
        self.args = args
        self.storage_path: str = os.path.join(self.cfg.RESULTS_PATH, self.cfg.DATASET)
        self._set_results_path()
        self._create_results_dirs()
        
        self.train_loader = get_dataset("TRAIN", self.cfg, args = self.args)
        self.val_loader = get_dataset("VAL", self.cfg, args = self.args)
        self.test_loader = get_dataset("TEST", self.cfg, args = self.args)
        self.evaluator = EvaluatorForeground(self.cfg.N_CLS_SEG)
        self.device = self._get_device()
        self.model = self._get_model().to(self.device)
        self.optimizer = self._get_optimizer(self.model)
        self.criterion_seg, self.criterion_dec = self._get_loss(True), self._get_loss(False)
        self.num_epochs = self.cfg.EPOCHS
        # self.tensorboard_writer = TensorboardSummary(self.tensorboard_path) if WRITE_TENSORBOARD else None
        self.tensorboard_writer = SummaryWriter(log_dir=self.tensorboard_path) if WRITE_TENSORBOARD else None
        self.best_pred = 0.
        self.start_epoch = 0
        self.train_loader_len = len(self.train_loader)
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
            self.model.load_state_dict(checkpoint['state_dict'])
            
            if not args.ft:
                self.start_epoch = checkpoint['epoch']
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.best_pred = checkpoint['best_pred']
            self._log("=> loaded checkpoint '{}' (epoch {})\n".format(args.resume, checkpoint['epoch']))
            # self.model.load_state_dict(checkpoint)
            # print(f"=> loaded checkpoint {args.resume}")
        
        self.print_run_params()
        print(f'train: {len(self.train_loader)}, val: {len(self.val_loader)}, test: {len(self.test_loader)}')
        
    def _log(self, message, lvl=LVL_INFO):
        n_msg = f"{message}"
        if lvl >= LOG:
            print(n_msg)
        self.write_log_to_txt(n_msg)
        
    def run(self):
        
        if self.cfg.REPRODUCIBLE_RUN:
            self._log("Reproducible run, fixing all seeds to:1337", LVL_DEBUG)
            np.random.seed(1337)
            torch.manual_seed(1337)
            random.seed(1337)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        validation_data = []
        max_validation = -1
        validation_step = self.cfg.VALIDATION_N_EPOCHS
        test_step = self.cfg.TEST_N_EPOCHS
        losses = []
        
        
        for epoch in range(self.start_epoch, self.num_epochs):
            ##train  
            losses = self.train(epoch)
            
            ## self.test(model, device, self.cfg.SAVE_IMAGES, True, False)

            # if self.cfg.VALIDATE and (epoch % validation_step == 0 or epoch == self.num_epochs - 1):
                # validation_ap, validation_accuracy = self.eval_model(device, model, val_loader, None, False, True, False)
            validation_ap, validation_accuracy = self.val(epoch)
            
            validation_data.append((validation_ap, epoch))

            if validation_ap > self.best_pred:
                self.best_pred = validation_ap
                is_best = True
                # self._save_model(self.model, "best_state_dict.pth")
                
            is_best = self.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)
                
            _, _ = self.test(epoch)
            
            if epoch % test_step == 0 or epoch == self.num_epochs - 1:
                self.eval_test_set(self.cfg.SAVE_IMAGES, False, False, epoch)
                
            # self._save_model(self.model, "model.pth")

        self.eval_test_set(self.cfg.SAVE_IMAGES, False, False, epoch)
        self._save_train_results((losses, validation_data))

    def train(self, epoch):
        self.model.train()

        debug_flag = self.cfg.DEBUG
        losses = []        
        samples_per_epoch = len(self.train_loader) * self.cfg.BATCH_SIZE

        self.set_dec_gradient_multiplier(self.model, 0.0)

        weight_loss_seg, weight_loss_dec = self.get_loss_weights(epoch)
        dec_gradient_multiplier = self.get_dec_gradient_multiplier()
        self.set_dec_gradient_multiplier(self.model, dec_gradient_multiplier)

        epoch_loss_seg, epoch_loss_dec, epoch_loss = 0, 0, 0
        epoch_correct = 0

        mode = 'train'
        ## begining of training code 
        time_acc = 0
        start = timer()
        tbar = tqdm(self.train_loader)
        # debug_flag = True
        for iter_index, data in enumerate(tbar):
            if debug_flag and iter_index > 30:
                break
            start_1 = timer()
            curr_loss_seg, curr_loss_dec, curr_loss, correct = self.train_iter(data, weight_loss_seg,
                                                                        weight_loss_dec, iter_index, epoch = epoch)
            end_1 = timer()
            time_acc = time_acc + (end_1 - start_1)

            epoch_loss_seg += curr_loss_seg
            epoch_loss_dec += curr_loss_dec
            epoch_loss += curr_loss

            epoch_correct += correct

        end = timer()

        epoch_loss_seg = epoch_loss_seg / samples_per_epoch
        epoch_loss_dec = epoch_loss_dec / samples_per_epoch
        epoch_loss = epoch_loss / samples_per_epoch
        losses.append((epoch_loss_seg, epoch_loss_dec, epoch_loss, epoch))

        self._log(
            f"{mode:5s} Epoch {epoch + 1:3d}/{self.num_epochs:3d} ==> avg_loss_seg={epoch_loss_seg:.5f}, avg_loss_dec={epoch_loss_dec:.5f}, avg_loss={epoch_loss:.5f}, correct={epoch_correct:5d}/{samples_per_epoch:5d}, in {end - start:.2f}s/epoch (fwd/bck in {time_acc:.2f}s/epoch)")

        if self.tensorboard_writer is not None:
            self.tensorboard_writer.add_scalar(f"{mode}/Epoch_Loss/segmentation", epoch_loss_seg, epoch)
            self.tensorboard_writer.add_scalar(f"{mode}/Epoch_Loss/classification", epoch_loss_dec, epoch)
            self.tensorboard_writer.add_scalar(f"{mode}/Epoch_Loss/joined", epoch_loss, epoch)
            self.tensorboard_writer.add_scalar(f"{mode}/Correct/", epoch_correct / samples_per_epoch, epoch)

        ### end of training code
        return losses

    def train_iter(self, data, weight_loss_seg, weight_loss_dec, iter_index, epoch = None):
        images, seg_masks, seg_loss_masks, is_segmented, sample_name = data
        # print('images.size', images.size()) ##torch.Size([1, 1, 256, 1600])
        # print('seg_masks.size', seg_masks.size()) ##torch.Size([1, 1, 32, 200])
        # print('seg_loss_masks.size', seg_loss_masks.size()) ##torch.Size([1, 1, 32, 200])
        # print('is_segmented', is_segmented) ##tensor([False])
        # print('sample_name', sample_name) ##sample_name ('69cf6690b',)

        batch_size = self.cfg.BATCH_SIZE
        memory_fit = self.cfg.MEMORY_FIT  # Not supported yet for > 1

        num_subiters = int(batch_size / memory_fit)
        #
        total_loss = 0
        total_correct = 0

        self.optimizer.zero_grad()

        total_loss_seg = 0
        total_loss_dec = 0
        # print('images', images.size, images)
        mode = 'train'
        for sub_iter in range(num_subiters):            
            # print(f'{sub_iter * memory_fit}:{(sub_iter + 1) * memory_fit}') #0:1
            images_ = images[sub_iter * memory_fit:(sub_iter + 1) * memory_fit, :, :, :].to(self.device)
            seg_masks_ = seg_masks[sub_iter * memory_fit:(sub_iter + 1) * memory_fit, :, :, :].to(self.device)
            seg_loss_masks_ = seg_loss_masks[sub_iter * memory_fit:(sub_iter + 1) * memory_fit, :, :, :].to(self.device)
            image_lable = seg_masks_.detach().cpu().numpy()
            image_lable[image_lable>=2] = 0     
            if len(image_lable[image_lable > 0]) > 12:
                is_pos_ = 1.
            else:
                is_pos_ = 0.
            is_pos_ = torch.from_numpy(np.array(is_pos_).reshape((memory_fit, 1))).to(self.device)
            # print('is_pos_', is_pos_) #tensor([[0.]] or tensor([[1.]]
            # print('seg_loss_masks_', seg_loss_masks_)
            # 
            decision, output_seg_mask = self.model(images_)
            
            if self.tensorboard_writer is not None and iter_index % 50 == 0:
                pred_seg = nn.Sigmoid()(output_seg_mask)
                prediction = nn.Sigmoid()(decision)
                # image_label = is_pos_.detach().cpu().numpy()[0][0]
                # 
                # pred_seg = pred_seg.detach().cpu().numpy()
                # im_res = nn.Sigmoid()(decision)
                # im_res = im_res.detach().cpu().numpy()
                # im_res = round(np.squeeze(im_res), 2)
                # # _, _, h, w = pred_seg.shape
                # # imgs = images_.detach().cpu().numpy()
                # # imgs = cv2.resize(imgs, (w,h))
                # # self.tensorboard_writer.add_image(f"{mode}/image {image_label}", images_[0, :, :, :], (epoch+1)*iter_index)
                # # self.tensorboard_writer.add_image(f"{mode}/image", images_[0, :, :, :], (epoch+1)*iter_index)
                # # self.tensorboard_writer.add_image(f"{mode}/seg_mask", seg_masks_[0, :, :, :], (epoch+1)*iter_index)
                # # self.tensorboard_writer.add_image(f"{mode}/seg_loss_mask", seg_loss_masks_[0, :, :, :], (epoch+1)*iter_index)
                # # self.tensorboard_writer.add_image(f"{mode}/pred_seg", pred_seg[0, :, :, :], (epoch+1)*iter_index)
                
                # self.tensorboard_writer.add_image(f"{mode}/{(epoch+1)*iter_index}/image {image_label}", images_[0, :, :, :])
                # self.tensorboard_writer.add_image(f"{mode}/{(epoch+1)*iter_index}/seg_mask", seg_masks_[0, :, :, :])
                # self.tensorboard_writer.add_image(f"{mode}/{(epoch+1)*iter_index}/seg_loss_mask", seg_loss_masks_[0, :, :, :])
                # self.tensorboard_writer.add_image(f"{mode}/{(epoch+1)*iter_index}/pred_seg {im_res}", pred_seg[0, :, :, :])
                # self.tensorboard_writer.vis_image(imgs, seg_masks, seg_loss_masks_, pred_seg, (epoch+1)*iter_index, mode, img_label = image_label) ##image, seg_mask, seg_loss_mask, pre_seg, global_step, split = ''):
                # figure = self.get_plot_sample(images_[0, :, :, :], pred_seg)#, seg_masks_[0, :, :, :], 
                figure = self.get_plot_sample(images_[0, :, :, :], pred_seg, seg_masks_[0, :, :, :], 
                                                  seg_loss_masks_[0, :, :, :], prediction, is_pos_)
                self.tensorboard_writer.add_figure(f"{mode}/{(epoch+1)*iter_index}",figure)
            # image_cpu = images_.detach().cpu().numpy()
            # print('image_cpu', image_cpu.shape) ### image_cpu (1, 1, 256, 1600)
            

            if is_segmented[sub_iter]:
                if self.cfg.WEIGHTED_SEG_LOSS:
                    # print('1', self.criterion_seg(output_seg_mask, seg_masks_).size(), self.criterion_seg(output_seg_mask, seg_masks_))
                    loss_seg = torch.mean(self.criterion_seg(output_seg_mask, seg_masks_) * seg_loss_masks_)
                    # print('2', loss_seg)
                else:
                    loss_seg = self.criterion_seg(output_seg_mask, seg_masks_)
                loss_dec = self.criterion_dec(decision, is_pos_)

                total_loss_seg += loss_seg.item()
                total_loss_dec += loss_dec.item()

                total_correct += (decision > 0.0).item() == is_pos_.item()
                loss = weight_loss_seg * loss_seg + weight_loss_dec * loss_dec
            else:
                loss_dec = self.criterion_dec(decision, is_pos_)
                total_loss_dec += loss_dec.item()

                total_correct += (decision > 0.0).item() == is_pos_.item()
                loss = weight_loss_dec * loss_dec
            total_loss += loss.item()

            loss.backward()

        # Backward and optimize
        self.optimizer.step()
        self.optimizer.zero_grad()

        return total_loss_seg, total_loss_dec, total_loss, total_correct
    
    def val(self, epoch = 0):
        return self.eval(self.val_loader, epoch = epoch, mode = 'val')

    def test(self, epoch = 0):
        return self.eval(self.test_loader, epoch = epoch, mode = 'test')

    def eval(self, data_loader, epoch = 0, mode = 'val'):
        self.model.eval()
        
        stop = self.train_loader_len if len(data_loader) >= self.train_loader_len else len(data_loader)
        # samples_per_epoch = len(data_loader) * self.cfg.BATCH_SIZE
        samples_per_epoch = stop * self.cfg.BATCH_SIZE if len(data_loader) > stop else len(data_loader) * self.cfg.BATCH_SIZE

        weight_loss_seg, weight_loss_dec = self.get_loss_weights(epoch)
        epoch_loss_seg, epoch_loss_dec, epoch_loss = 0, 0, 0
        epoch_correct = 0

        batch_size = self.cfg.BATCH_SIZE
        memory_fit = self.cfg.MEMORY_FIT  # Not supported yet for > 1
        debug_flag = self.cfg.DEBUG
        num_subiters = int(batch_size / memory_fit)
        res = []
        predictions, ground_truths = [], []
        tbar = tqdm(data_loader)
        for iter_index, data_point in enumerate(tbar):
            if debug_flag and iter_index > 30:
                break
            if iter_index >= stop:
                break
            images, seg_masks, seg_loss_masks, is_segmented, sample_name = data_point
            total_loss = 0
            total_correct = 0
            total_loss_seg = 0
            total_loss_dec = 0
            # res_ = []
            # predictions_, ground_truths_ = [], []
            for sub_iter in range(num_subiters):
                if debug_flag and not sub_iter % 80 == 0:
                    continue
                images_ = images[sub_iter * memory_fit:(sub_iter + 1) * memory_fit, :, :, :].to(self.device)
                seg_masks_ = seg_masks[sub_iter * memory_fit:(sub_iter + 1) * memory_fit, :, :, :].to(self.device)
                seg_loss_masks_ = seg_loss_masks[sub_iter * memory_fit:(sub_iter + 1) * memory_fit, :, :, :].to(self.device)  
                seg_mask = seg_masks_.detach().cpu().numpy()
                seg_mask_ = seg_mask.copy()
                seg_mask_[seg_mask_>=2] = 0              
                # is_pos_ = torch.from_numpy(seg_mask_.max().reshape((memory_fit, 1))).to(self.device)
                if len(seg_mask_[seg_mask_ > 0]) > 12:
                    is_pos_ = 1.
                else:
                    is_pos_ = 0.
                is_pos_ = torch.from_numpy(np.array(is_pos_).reshape((memory_fit, 1))).to(self.device)
                ### print('seg_loss_masks_', seg_loss_masks_.min(), '-->', seg_loss_masks_.max()) 
                ### tensor(1., device='cuda:5') --> tensor(1.6686, device='cuda:5') if reverse_distance_transform
                with torch.no_grad():
                    decision, output_seg_mask = self.model(images_)
                    
                pred_seg = nn.Sigmoid()(output_seg_mask)
                prediction = nn.Sigmoid()(decision)
                
                
                
                if self.tensorboard_writer is not None and iter_index % 40 == 0:
                    # pred_seg = pred_seg.detach().cpu().numpy()
                    # is_pos = is_pos_.detach().cpu().numpy()[0][0]
                    # _, _, h, w = pred_seg.shape
                    # # imgs = images_.detach().cpu().numpy()
                    # # imgs = cv2.resize(imgs, (w,h))
                    # im_res = prediction.detach().cpu().numpy()
                    # im_res = round(np.squeeze(im_res), 2)
                    
                    # self.tensorboard_writer.add_image(f"{mode}/{(epoch+1)*iter_index}/image {is_pos}", images_[0, :, :, :])
                    # self.tensorboard_writer.add_image(f"{mode}/{(epoch+1)*iter_index}/seg_mask", seg_masks_[0, :, :, :])
                    # self.tensorboard_writer.add_image(f"{mode}/{(epoch+1)*iter_index}/seg_loss_mask", seg_loss_masks_[0, :, :, :])
                    # self.tensorboard_writer.add_image(f"{mode}/{(epoch+1)*iter_index}/pred_seg {im_res}", pred_seg[0, :, :, :])
                    
                    # self.tensorboard_writer.add_image(f"{mode}/image {is_pos}", images_[0, :, :, :], (epoch+1)*iter_index)
                    # self.tensorboard_writer.add_image(f"{mode}/image", images_[0, :, :, :], (epoch+1)*iter_index)
                    # self.tensorboard_writer.add_image(f"{mode}/seg_mask", seg_masks_[0, :, :, :], (epoch+1)*iter_index)
                    # self.tensorboard_writer.add_image(f"{mode}/seg_loss_mask", seg_loss_masks_[0, :, :, :], (epoch+1)*iter_index)
                    # self.tensorboard_writer.add_image(f"{mode}/pred_seg", pred_seg[0, :, :, :], (epoch+1)*iter_index)
                    # self.tensorboard_writer.vis_image(imgs, seg_masks, seg_loss_masks_, pred_seg, (epoch+1)*iter_index, mode, img_label = image_label) ##image, seg_mask, seg_loss_mask, pre_seg, global_step, split = ''):
                    # image, segmentation, seg_mask = None, seg_loss_mask = None, decision=None, is_pos
                    figure = self.get_plot_sample(images_[0, :, :, :], pred_seg, seg_masks_[0, :, :, :], 
                                                  seg_loss_masks_[0, :, :, :], prediction, is_pos_)
                    self.tensorboard_writer.add_figure(f"{mode}/{(epoch+1)*iter_index}",figure)

                
                prediction = prediction.item()
                # image = images_.detach().cpu().numpy()
                

                predictions.append(prediction)
                ground_truths.append(is_pos_.item())
                res.append((prediction, None, None, is_pos_.item(), sample_name[0]))
                
                if is_segmented[sub_iter]:
                    if self.cfg.WEIGHTED_SEG_LOSS:
                        loss_seg = torch.mean(self.criterion_seg(output_seg_mask, seg_masks_) * seg_loss_masks_)
                    else:
                        loss_seg = self.criterion_seg(output_seg_mask, seg_masks_)
                    loss_dec = self.criterion_dec(decision, is_pos_)

                    total_loss_seg += loss_seg.item()
                    total_loss_dec += loss_dec.item()

                    total_correct += (decision > 0.0).item() == is_pos_.item()
                    loss = weight_loss_seg * loss_seg + weight_loss_dec * loss_dec

                    #### Add batch sample into evaluator
                    seg_mask_int = seg_mask.copy()
                    seg_mask_int[seg_mask_int>0] = 1
                    seg_mask_int = seg_mask_int.astype(np.int32)
                    # if seg_mask_int.any() > 0:
                    #     pred = np.argmax(output_seg_mask.detach().cpu().numpy(), axis=1).astype(np.int32)  
                    #     # print('seg_mask_int.shape', seg_mask_int.shape)                  
                    #     # print('pred.shape', pred.shape)                  
                    #     self.evaluator.add_batch(np.squeeze(seg_mask_int), np.squeeze(pred))
                else:
                    loss_dec = self.criterion_dec(decision, is_pos_)
                    total_loss_dec += loss_dec.item()

                    total_correct += (decision > 0.0).item() == is_pos_.item()
                    loss = weight_loss_dec * loss_dec
                total_loss += loss.item()

            epoch_loss_seg += total_loss_seg
            epoch_loss_dec += total_loss_dec
            epoch_loss += total_loss
            epoch_correct += total_correct

            # res.append(res_)
            # predictions.append(predictions_)
            # ground_truths.append(ground_truths_)
        
        epoch_loss_seg = epoch_loss_seg / samples_per_epoch
        epoch_loss_dec = epoch_loss_dec / samples_per_epoch
        epoch_loss = epoch_loss / samples_per_epoch

        ground_truths = np.array(ground_truths)
        predictions = np.array(predictions).astype(np.int64)
        # print(f'val ground_truths', ground_truths.shape, ground_truths)
        # print(f'val predictions', predictions.shape, predictions)
        # print()
        metrics = utils.get_metrics(ground_truths.astype(np.int64), predictions)
        # metrics = utils.get_metrics(np.array(ground_truths), np.array(predictions))
        FP, FN, TP, TN = list(map(sum, [metrics["FP"], metrics["FN"], metrics["TP"], metrics["TN"]]))
        self._log(f"{mode:5s} Epoch {epoch + 1:3d}/{self.num_epochs:3d} || AUC={metrics['AUC']:f}, and AP={metrics['AP']:f}, with best thr={metrics['best_thr']:f} "
                    f"at f-measure={metrics['best_f_measure']:.3f} and FP={FP:d}, FN={FN:d}, TOTAL SAMPLES={FP + FN + TP + TN:d}")

        if mode == 'test':
            self._log('\n')
            
        # mIoU = self.evaluator.Mean_Intersection_over_Union()
        validation_accuracy = metrics["accuracy"]
        validation_ap = metrics["AP"]
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.add_scalar(f"{mode}/Epoch_Loss/segmentation", epoch_loss_seg, epoch)
            self.tensorboard_writer.add_scalar(f"{mode}/Epoch_Loss/classification", epoch_loss_dec, epoch)
            self.tensorboard_writer.add_scalar(f"{mode}/Epoch_Loss/joined", epoch_loss, epoch)
            self.tensorboard_writer.add_scalar(f"{mode}/Correct/", epoch_correct / samples_per_epoch, epoch)
            self.tensorboard_writer.add_scalar(f"{mode}/Accuracy", validation_accuracy, epoch)
            self.tensorboard_writer.add_scalar(f"{mode}/AP", validation_ap, epoch)
            # tensorboard_writer.add_scalar(f"{mode}/mIoU", mIoU, epoch)

        return metrics["AP"], metrics["accuracy"]



    def eval_test_set(self, save_images, plot_seg, reload_final, epoch = None):
        if reload_final:
            self.reload_model(self.model, reload_final)
        self.eval_model(self.test_loader, save_folder=self.outputs_path, save_images=save_images, is_validation=False, plot_seg=plot_seg, epoch = epoch)



    def eval_model(self, eval_loader, save_folder, save_images, is_validation, plot_seg, epoch = None, mode = 'test'):
        self.model.eval()

        dsize = self.cfg.INPUT_WIDTH, self.cfg.INPUT_HEIGHT
        debug_flag = self.cfg.DEBUG
        res = []
        predictions, ground_truths = [], []
        
        tbar = tqdm(eval_loader)
        for iter_index, (data_point) in enumerate(tbar):
            if debug_flag and iter_index > 5:
                break
            if iter_index > 600:
                break
            image, seg_mask, seg_loss_mask, _, sample_name = data_point
            image, seg_mask = image.to(self.device), seg_mask.to(self.device)
            is_pos = (seg_mask.max() > 0).reshape((1, 1)).to(self.device).item()
            with torch.no_grad():
                prediction, pred_seg = self.model(image)
            pred_seg = nn.Sigmoid()(pred_seg)
            prediction = nn.Sigmoid()(prediction)

            prediction = prediction.item()
            image = image.detach().cpu().numpy()
            pred_seg = pred_seg.detach().cpu().numpy()
            seg_mask = seg_mask.detach().cpu().numpy()

            predictions.append(prediction)
            ground_truths.append(is_pos)
            res.append((prediction, None, None, is_pos, sample_name[0]))
            if not is_validation:
                if save_images:
                    if not iter_index % 50 == 0:
                        continue
                    image = cv2.resize(np.transpose(image[0, :, :, :], (1, 2, 0)), dsize)
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    pred_seg = cv2.resize(pred_seg[0, 0, :, :], dsize) if len(pred_seg.shape) == 4 else cv2.resize(pred_seg[0, :, :], dsize)
                    seg_mask = cv2.resize(seg_mask[0, 0, :, :], dsize)
                    img_name = sample_name[0]
                    if self.cfg.WEIGHTED_SEG_LOSS: #True
                        seg_loss_mask = cv2.resize(seg_loss_mask.numpy()[0, 0, :, :], dsize)
                        utils.plot_sample(sample_name[0], image, pred_seg, seg_mask = seg_mask, seg_loss_mask = seg_loss_mask,
                                             save_dir = save_folder, decision=prediction, plot_seg=plot_seg, epoch = epoch, is_pos = is_pos)
                    else:
                        # print(f'epoch {epoch}, {img_name}, image: {image.min()}->{image.max()}; pred_seg: {pred_seg.min()}->{pred_seg.max()}; seg_mask: {seg_mask.min()}->{seg_mask.max()}')
                        utils.plot_sample(sample_name[0], image, pred_seg, seg_mask = seg_mask, 
                                             save_dir = save_folder, decision=prediction, plot_seg=plot_seg, epoch = epoch)

        if is_validation:
            ground_truths = np.array(ground_truths)
            predictions = np.array(predictions)
            # print(f'test ground_truths', ground_truths)
            # print(f'test predictions', predictions)
            # print()
            metrics = utils.get_metrics(ground_truths.astype(np.int64), predictions)
            FP, FN, TP, TN = list(map(sum, [metrics["FP"], metrics["FN"], metrics["TP"], metrics["TN"]]))
            self._log(f"{mode:5s} Epoch {epoch + 1:3d}/{self.num_epochs:3d}  || AUC={metrics['AUC']:f}, and AP={metrics['AP']:f}, with best thr={metrics['best_thr']:f} "
                      f"at f-measure={metrics['best_f_measure']:.3f} and FP={FP:d}, FN={FN:d}, TOTAL SAMPLES={FP + FN + TP + TN:d}")

            return metrics["AP"], metrics["accuracy"]
        else:
            utils.evaluate_metrics(res, self.run_path, self.run_name)

    def get_dec_gradient_multiplier(self):
        if self.cfg.GRADIENT_ADJUSTMENT:
            grad_m = 0
        else:
            grad_m = 1

        self._log(f"Returning dec_gradient_multiplier {grad_m}", LVL_DEBUG)
        return grad_m

    def set_dec_gradient_multiplier(self, model, multiplier):
        model.set_gradient_multipliers(multiplier)

    def get_loss_weights(self, epoch):
        total_epochs = float(self.cfg.EPOCHS)

        if self.cfg.DYN_BALANCED_LOSS:
            seg_loss_weight = 1 - (epoch / total_epochs)
            dec_loss_weight = self.cfg.DELTA_CLS_LOSS * (epoch / total_epochs)
        else:
            seg_loss_weight = 1
            dec_loss_weight = self.cfg.DELTA_CLS_LOSS

        self._log(f"Returning seg_loss_weight {seg_loss_weight} and dec_loss_weight {dec_loss_weight}", LVL_DEBUG)
        return seg_loss_weight, dec_loss_weight

    def reload_model(self, model, load_final=False):
        if self.cfg.USE_BEST_MODEL:
            path = os.path.join(self.model_path, "best_state_dict.pth")
            model.load_state_dict(torch.load(path))
            self._log(f"Loading model state from {path}")
        elif load_final:
            path = os.path.join(self.model_path, "model.pth")
            model.load_state_dict(torch.load(path))
            self._log(f"Loading model state from {path}")
        else:
            self._log("Keeping same model state")

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

    def _save_params(self):
        params = self.cfg.get_as_dict()
        params_lines = sorted(map(lambda e: e[0] + ":" + str(e[1]) + "\n", params.items()))
        fname = os.path.join(self.run_path, "run_params.txt")
        with open(fname, "w+") as f:
            f.writelines(params_lines)

    def _save_train_results(self, results):
        losses, validation_data = results
        ls, ld, l, le = map(list, zip(*losses))
        plt.plot(le, l, label="Loss", color="red")
        plt.plot(le, ls, label="Loss seg")
        plt.plot(le, ld, label="Loss dec")
        plt.ylim(bottom=0)
        plt.grid()
        plt.xlabel("Epochs")
        if self.cfg.VALIDATE:
            v, ve = map(list, zip(*validation_data))
            plt.twinx()
            plt.plot(ve, v, label="Validation AP", color="Green")
            plt.ylim((0, 1))
        plt.legend()
        plt.savefig(os.path.join(self.run_path, "loss_val"), dpi=200)

        df_loss = pd.DataFrame(data={"loss_seg": ls, "loss_dec": ld, "loss": l, "epoch": le})
        df_loss.to_csv(os.path.join(self.run_path, "losses.csv"), index=False)

        if self.cfg.VALIDATE:
            df_loss = pd.DataFrame(data={"validation_data": ls, "loss_dec": ld, "loss": l, "epoch": le})
            df_loss.to_csv(os.path.join(self.run_path, "losses.csv"), index=False)

    def _save_model(self, model, name="final_state_dict.pth"):
        output_name = os.path.join(self.model_path, name)
        self._log(f"Saving current model state to {output_name}")
        if os.path.exists(output_name):
            os.remove(output_name)
        
        torch.save(model.state_dict(), output_name)

    def _get_optimizer(self, model):
        return torch.optim.SGD(model.parameters(), self.cfg.LEARNING_RATE)

    def _get_loss(self, is_seg):
        reduction = "none" if self.cfg.WEIGHTED_SEG_LOSS and is_seg else "mean"
        return nn.BCEWithLogitsLoss(reduction=reduction).to(self._get_device())

    def _get_device(self):
        return f"cuda:{self.cfg.GPU}"

    def _set_results_path(self):
        self.run_name = f"{self.cfg.RUN_NAME}_FOLD_{self.cfg.FOLD}" if self.cfg.DATASET in ["KSDD", "DAGM"] else self.cfg.RUN_NAME

        # results_path = os.path.join(self.cfg.RESULTS_PATH, self.cfg.DATASET, self.run_name)
        project_path = str(ROOT /'runs')
        results_path = str(self.increment_path(Path(project_path)/self.cfg.DATASET/self.run_name/'exp', exist_ok=False))
        self.tensorboard_path = os.path.join(results_path, "tensorboard" )
        # run_path = os.path.join(results_path, self.cfg.RUN_NAME)
        run_path = results_path
        if self.cfg.DATASET in ["KSDD", "DAGM"]:
            run_path = os.path.join(run_path, f"FOLD_{self.cfg.FOLD}")
            
        self.run_path = run_path
        self.model_path = os.path.join(run_path, "models")
        self.outputs_path = os.path.join(run_path, "test_outputs")
        self.logfile_path = os.path.join(run_path, "log.txt")

        print(f"Executing run with path {self.run_path}")
        

    def _create_results_dirs(self):
        list(map(utils.create_folder, [self.run_path, self.model_path, self.outputs_path, ]))
        self._log(f"Executing run with path {self.run_path}")
        
    def _get_model(self):
        seg_net = SegDecNet(self._get_device(), self.cfg.INPUT_WIDTH, self.cfg.INPUT_HEIGHT, self.cfg.INPUT_CHANNELS)
        return seg_net

    def print_run_params(self):
        parameter_txt = os.path.join(self.run_path, 'parameter.txt')
        f = open(parameter_txt, "a+")
    
        for l in sorted(map(lambda e: e[0] + ":" + str(e[1]) + "\n", self.cfg.get_as_dict().items())):
            k, v = l.split(":")
            self._log(f"{k:25s} : {str(v.strip())}")
            line = f"{k:25s} : {str(v.strip())}"
            f.write(line +'\n')
        f.write('\n')
        f.write(f"dataset split file: splits/{self.cfg.DATASET}/split_{self.cfg.TRAIN_NUM}_{self.cfg.NUM_SEGMENTED}.pyb" +'\n')
        f.write(f'train: {len(self.train_loader)}, val: {len(self.val_loader)}, test: {len(self.test_loader)}' +'\n')
        f.close()#
            
     
    def increment_path(self, path, exist_ok=False, sep='', mkdir=False):
        # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
        path = Path(path)  # os-agnostic
        if path.exists() and not exist_ok:
            path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')
            dirs = glob.glob(f"{path}{sep}*")  # similar paths
            matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
            i = [int(m.groups()[0]) for m in matches if m]  # indices
            n = max(i) + 1 if i else 2  # increment number
            path = Path(f"{path}{sep}{n}{suffix}")  # increment path
        if mkdir:
            path.mkdir(parents=True, exist_ok=True)  # make directory
        return path


    def preprocess_np(self, image, device):
        assert image.ndim == 2
        t_h, t_w = self.cfg.INPUT_HEIGHT, self.cfg.INPUT_WIDTH
        image = cv2.resize(image, (t_w, t_h))
        image = image.astype("float32") / 255.0       
        image = np.expand_dims(image, 0)
        image = np.expand_dims(image, 0)
        image = torch.from_numpy(image).to(device)
        return image
    
    def write_log_to_txt(self, data):
        assert isinstance(data, str)
        log_file = open(self.logfile_path, "a+")
        if data.endswith('\n'):
            log_file.write(data)
        else:
            log_file.write(data+'\n')
        log_file.close()# 
        
    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        """Saves checkpoint to disk"""

        filename = os.path.join(self.model_path, filename)
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, os.path.join(self.model_path, 'model_best.pth.tar'))
        return False
    
    def get_plot_sample(self, image, seg_pre, seg_mask = None, seg_loss_mask = None, decision=None, is_pos = None,  blur=True):
        
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        if isinstance(seg_pre, torch.Tensor):
            seg_pre = seg_pre.detach().cpu().numpy()
        
        if not seg_mask is None and isinstance(seg_mask, torch.Tensor):
            seg_mask = seg_mask.detach().cpu().numpy()
            
        if not seg_loss_mask is None and isinstance(seg_loss_mask, torch.Tensor):
            seg_loss_mask = seg_loss_mask.detach().cpu().numpy()
        
        if not decision is None and isinstance(decision, torch.Tensor):
            decision = decision.detach().cpu().numpy()
            
        if not is_pos is None and isinstance(is_pos, torch.Tensor):
            is_pos = is_pos.detach().cpu().numpy()
            
        image = np.squeeze(image)
        seg_pre = np.squeeze(seg_pre)
        
        if not seg_mask is None and not seg_loss_mask is None:
            n_col = 5
        elif not seg_mask is None or not seg_loss_mask is None:
            n_col = 4
        else:
            n_col = 3
        
        ## figure = plt.figure(figsize=(1,n_col)) 
        figure = plt.figure(figsize=(2.4*n_col, 3))
        
        
        pos = 1
        plt.subplot(1, n_col, pos)
        plt.xticks([])
        plt.yticks([])
        if not is_pos is None:
            is_pos = np.squeeze(is_pos)
            plt.title(f'InputImage\n{is_pos}', verticalalignment = 'bottom', fontsize = 'small')
        else:
            plt.title('InputImage', verticalalignment = 'bottom', fontsize = 'small')
        # plt.ylabel('Input image', multialignment='center')
        if image.shape[0] < image.shape[1]:
            trans_flag = True
            if image.ndim == 3:
                image = np.transpose(image, axes=[1, 0, 2])
            else:
                image = np.transpose(image) 
            seg_pre = np.transpose(seg_pre)
        else:
            trans_flag = False
            
        if image.ndim == 2:
            plt.imshow(image, cmap="gray")
        else:
            plt.imshow(image)
        pos += 1

        if not seg_mask is None:
            seg_mask = np.squeeze(seg_mask)
            label = seg_mask.copy()
            label = np.transpose(label) if trans_flag else label
            label_min = label.min()
            label_max = label.max()
            plt.subplot(1, n_col, pos)
            plt.xticks([])
            plt.yticks([])
            # plt.title('Groundtruth')
            plt.title(f'segMask\n{label_min:.2f}->{label_max:.2f}', verticalalignment = 'bottom', fontsize = 'small')
            plt.imshow(label, cmap="gray")
            pos += 1

        if not seg_loss_mask is None:
            seg_loss_mask = np.squeeze(seg_loss_mask)
            label = seg_loss_mask.copy()
            label = np.transpose(label) if trans_flag else label
            label_min = label.min()
            label_max = label.max()
            plt.subplot(1, n_col, pos)
            plt.xticks([])
            plt.yticks([])
            # plt.title('Groundtruth')
            plt.title(f'segLossMask\n{label_min:.2f}->{label_max:.2f}', verticalalignment = 'bottom', fontsize = 'small')
            plt.imshow(label, cmap="gray")
            pos += 1

        plt.subplot(1, n_col, pos)
        plt.xticks([])
        plt.yticks([])
        if decision is None:
            plt.title('Output', verticalalignment = 'bottom')
            # plt.ylabel('Output', multialignment='center')
        else:
            decision = np.squeeze(decision)
            plt.title(f"Output\nConf:{decision:.2f}", verticalalignment = 'bottom', fontsize = 'small')
            # plt.ylabel(f"Output:{decision:.2f}", multialignment='center')
        # display max
        vmax_value = max(1, np.max(seg_pre))
        plt.imshow(seg_pre, cmap="jet", vmax=vmax_value)
        pos += 1

        plt.subplot(1, n_col, pos)
        plt.xticks([])
        plt.yticks([])
        plt.title(f'OutputScaled\nmax:{seg_pre.max():.2f}', verticalalignment = 'bottom', fontsize = 'small')
        # plt.ylabel('OutputScaled', multialignment='center')
        if blur:
            normed = seg_pre / seg_pre.max()
            blured = cv2.blur(normed, (32, 32))
            plt.imshow((blured / blured.max() * 255).astype(np.uint8), cmap="jet")
        else:
            plt.imshow((seg_pre / seg_pre.max() * 255).astype(np.uint8), cmap="jet")
        # plt.show()

        return figure