import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import pandas as pd
import os
import errno
import pickle
import cv2
from PIL import Image, ImageOps, ImageFilter


def create_folder(folder, exist_ok=True):
    try:
        os.makedirs(folder)
    except OSError as e:
        if e.errno != errno.EEXIST or not exist_ok:
            raise


def calc_confusion_mat(D, Y):
    FP = (D != Y) & (Y.astype(np.bool) == False)
    FN = (D != Y) & (Y.astype(np.bool) == True)
    TN = (D == Y) & (Y.astype(np.bool) == False)
    TP = (D == Y) & (Y.astype(np.bool) == True)

    return FP, FN, TN, TP


def plot_sample(image_name, image, segmentation, seg_mask = None, seg_loss_mask = None, save_dir = None, decision=None, blur=True, plot_seg=False, epoch = None, is_pos = None):
    
    
    if not seg_mask is None and not seg_loss_mask is None:
        n_col = 5
    elif not seg_mask is None or not seg_loss_mask is None:
        n_col = 4
    else:
        n_col = 3
    pos = 1
    plt.figure()
    plt.clf()
    plt.subplot(1, n_col, pos)
    plt.xticks([])
    plt.yticks([])
    if not is_pos is None:
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
        segmentation = np.transpose(segmentation)
    else:
        trans_flag = False
        
    if image.shape[2] == 1:
        plt.imshow(image, cmap="gray")
    else:
        plt.imshow(image)
    pos += 1

    if not seg_mask is None:
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
        plt.title(f"Output\nConf:{decision:.2f}", verticalalignment = 'bottom', fontsize = 'small')
        # plt.ylabel(f"Output:{decision:.2f}", multialignment='center')
    # display max
    vmax_value = max(1, np.max(segmentation))
    plt.imshow(segmentation, cmap="jet", vmax=vmax_value)
    pos += 1

    plt.subplot(1, n_col, pos)
    plt.xticks([])
    plt.yticks([])
    plt.title(f'OutputScaled\nmax:{segmentation.max():.2f}', verticalalignment = 'bottom', fontsize = 'small')
    # plt.ylabel('OutputScaled', multialignment='center')
    if blur:
        normed = segmentation / segmentation.max()
        blured = cv2.blur(normed, (32, 32))
        plt.imshow((blured / blured.max() * 255).astype(np.uint8), cmap="jet")
    else:
        plt.imshow((segmentation / segmentation.max() * 255).astype(np.uint8), cmap="jet")

    out_prefix = '{:.3f}_'.format(decision) if decision is not None else ''
    # if not epoch is None:
    #     out_img_name = "epoch%03d" % epoch + f'_{image_name}_conf_{out_prefix}.jpg'
    # else:
    #     out_img_name = f'{image_name}_conf_{out_prefix}.jpg'
    
    if not epoch is None:
        out_img_name = "epoch%03d" % epoch + f'_{image_name}.jpg'
    else:
        out_img_name = f'{image_name}.jpg'

    plt.savefig(f"{save_dir}/{out_img_name}", bbox_inches='tight', dpi=300)
    plt.close()

    if plot_seg:
        jet_seg = cv2.applyColorMap((segmentation * 255).astype(np.uint8), cv2.COLORMAP_JET)
        cv2.imwrite(f"{save_dir}/{image_name}_segmentation_{out_prefix}.png", jet_seg)


def evaluate_metrics(samples, results_path, run_name, mIoU = None, thres = None):
    samples = np.array(samples)

    img_names = samples[:, 4]
    predictions = samples[:, 0]
    labels = samples[:, 3].astype(np.float32)

    metrics = get_metrics(labels, predictions)

    df = pd.DataFrame(
        data={'prediction': predictions,
              'decision': metrics['decisions'],
              'ground_truth': labels,
              'img_name': img_names})
    df.to_csv(os.path.join(results_path, 'results.csv'), index=False)

    print(
        f'{run_name} EVAL AUC={metrics["AUC"]:f}, and AP={metrics["AP"]:f}, w/ best thr={metrics["best_thr"]:f} at f-m={metrics["best_f_measure"]:.3f} and FP={sum(metrics["FP"]):d}, FN={sum(metrics["FN"]):d}')

    with open(os.path.join(results_path, 'metrics.pkl'), 'wb') as f:
        pickle.dump(metrics, f)
        f.close()

    plt.figure(1)
    plt.clf()
    plt.plot(metrics['recall'], metrics['precision'])
    if not mIoU is None:
        plt.title('Average Precision=%.4f, mIoU=%.4f@thres%.2f' %(metrics['AP'], mIoU, thres))
    else:
        plt.title('Average Precision=%.4f' % metrics['AP'])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.savefig(f"{results_path}/precision-recall.pdf", bbox_inches='tight')

    plt.figure(1)
    plt.clf()
    plt.plot(metrics['FPR'], metrics['TPR'])
    plt.title('AUC=%.4f' % metrics['AUC'])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.savefig(f"{results_path}/ROC.pdf", bbox_inches='tight')
    plt.close()


def get_metrics(labels, predictions):
    metrics = {}
    precision, recall, thresholds = precision_recall_curve(labels, predictions)
    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['thresholds'] = thresholds
    f_measures = 2 * np.multiply(recall, precision) / (recall + precision + 1e-8)
    metrics['f_measures'] = f_measures
    ix_best = np.argmax(f_measures)
    metrics['ix_best'] = ix_best
    best_f_measure = f_measures[ix_best]
    metrics['best_f_measure'] = best_f_measure
    best_thr = thresholds[ix_best]
    metrics['best_thr'] = best_thr
    FPR, TPR, _ = roc_curve(labels, predictions)
    metrics['FPR'] = FPR
    metrics['TPR'] = TPR
    AUC = auc(FPR, TPR)
    metrics['AUC'] = AUC
    AP = auc(recall, precision)
    metrics['AP'] = AP
    decisions = predictions >= best_thr
    metrics['decisions'] = decisions
    FP, FN, TN, TP = calc_confusion_mat(decisions, labels)
    metrics['FP'] = FP
    metrics['FN'] = FN
    metrics['TN'] = TN
    metrics['TP'] = TP
    metrics['accuracy'] = (sum(TP) + sum(TN)) / (sum(TP) + sum(TN) + sum(FP) + sum(FN))
    return metrics


# ab = {'a': 1, "b":2}
# a, b = ab
# print(a, b)
# if 'a' in ab:
#     print('yes')
# a = 1
# b = np.array(a)
# c = b.reshape((1,1))
# print(a, b, c)

# lists = [[0.1, 0.3], [3,2], [10, 0.4]]
# lists = [[[[1]]]]
# ar = np.array(lists)
# ar = np.squeeze(ar)
# print(ar[0])
# print(ar)
# ar = np.random.random((100, 100))
# ar_pil = Image.fromarray(ar)
# ar_pil = ar_pil.rotate(30, Image.NEAREST, fillcolor = 0)
# ar_pil.show()

# ar_pil = ar_pil.transpose(Image.FLIP_LEFT_RIGHT)
# ar_1 = np.array(ar_pil)

# print('1', ar_1)

