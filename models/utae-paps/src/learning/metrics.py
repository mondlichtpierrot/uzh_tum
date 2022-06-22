import os
import sys
import torch
import numpy as np
import pandas as pd

from src.learning.miou import Metric
sys.path.append(os.path.dirname(os.getcwd()))
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
#print(os.path.dirname(os.path.dirname(os.getcwd())))
#exit()
from util import pytorch_ssim

def mIou(y_true, y_pred, n_classes):
    """
    Mean Intersect over Union metric.
    Computes the one versus all IoU for each class and returns the average.
    Classes that do not appear in the provided set are not counted in the average.
    Args:
        y_true (1D-array): True labels
        y_pred (1D-array): Predicted labels
        n_classes (int): Total number of classes
    Returns:
        mean Iou (float)
    """
    iou = 0
    n_observed = n_classes
    for i in range(n_classes):
        y_t = (np.array(y_true) == i).astype(int)
        y_p = (np.array(y_pred) == i).astype(int)

        inter = np.sum(y_t * y_p)
        union = np.sum((y_t + y_p > 0).astype(int))

        if union == 0:
            n_observed -= 1
        else:
            iou += inter / union

    return iou / n_observed



def confusion_matrix_analysis(mat):
    """
    This method computes all the performance metrics from the confusion matrix. In addition to overall accuracy, the
    precision, recall, f-score and IoU for each class is computed.
    The class-wise metrics are averaged to provide overall indicators in two ways (MICRO and MACRO average)
    Args:
        mat (array): confusion matrix

    Returns:
        per_class (dict) : per class metrics
        overall (dict): overall metrics

    """
    TP = 0
    FP = 0
    FN = 0

    per_class = {}

    for j in range(mat.shape[0]):
        d = {}
        tp = np.sum(mat[j, j])
        fp = np.sum(mat[:, j]) - tp
        fn = np.sum(mat[j, :]) - tp

        d['IoU'] = tp / (tp + fp + fn)
        d['Precision'] = tp / (tp + fp)
        d['Recall'] = tp / (tp + fn)
        d['F1-score'] = 2 * tp / (2 * tp + fp + fn)

        per_class[str(j)] = d

        TP += tp
        FP += fp
        FN += fn

    overall = {}
    overall['micro_IoU'] = TP / (TP + FP + FN)
    overall['micro_Precision'] = TP / (TP + FP)
    overall['micro_Recall'] = TP / (TP + FN)
    overall['micro_F1-score'] = 2 * TP / (2 * TP + FP + FN)

    macro = pd.DataFrame(per_class).transpose().mean()
    overall['MACRO_IoU'] = macro.loc['IoU']
    overall['MACRO_Precision'] = macro.loc['Precision']
    overall['MACRO_Recall'] = macro.loc['Recall']
    overall['MACRO_F1-score'] = macro.loc['F1-score']

    overall['Accuracy'] = np.sum(np.diag(mat)) / np.sum(mat)

    return per_class, overall

def img_metrics(target, pred, masks):
    rmse = torch.sqrt(torch.mean(torch.square(target - pred)))
    psnr = 20 * torch.log10(1 / rmse)
    mae = torch.mean(torch.abs(target - pred))
    
    # spectral angle mapper
    mat = target * pred
    mat = torch.sum(mat, 1)
    mat = torch.div(mat, torch.sqrt(torch.sum(target * target, 1)))
    mat = torch.div(mat, torch.sqrt(torch.sum(pred * pred, 1)))
    sam = torch.mean(torch.acos(torch.clamp(mat, -1, 1)))

    ssim = pytorch_ssim.ssim(target, pred)
    
    # get an aggregated cloud mask over all time points and compute metrics over (non-)cloudy px
    tileTo = target.shape[1]
    mask   = torch.clamp(torch.sum(masks, dim=0, keepdim=True), 0, 1)
    mask   = mask.repeat(1, tileTo, 1, 1)
    target, pred, mask = target.cpu().numpy(), pred.cpu().numpy(), mask.cpu().numpy()

    rmse_cloudy = np.sqrt(np.nanmean(np.square(target[mask==1] - pred[mask==1])))
    rmse_cloudfree = np.sqrt(np.nanmean(np.square(target[mask==0] - pred[mask==0])))
    mae_cloudy = np.nanmean(np.abs(target[mask==1] - pred[mask==1]))
    mae_cloudfree = np.nanmean(np.abs(target[mask==0] - pred[mask==0]))

    return {'RMSE': rmse.cpu().numpy().item(),
            'RMSE_cloudy': rmse_cloudy, 
            'RMSE_cloudfree': rmse_cloudfree, 
            'MAE': mae.cpu().numpy().item(),
            'MAE_cloudy': mae_cloudy, 
            'MAE_cloudfree': mae_cloudfree, 
            'PSNR': psnr.cpu().numpy().item(),
            'SAM': sam.cpu().numpy().item(),
            'SSIM': ssim.cpu().numpy().item()}

class avg_img_metrics(Metric):
    def __init__(self):
        super().__init__()
        self.n_samples = 0
        self.metrics   = ['RMSE','RMSE_cloudy','RMSE_cloudfree','MAE', 'MAE_cloudy','MAE_cloudfree','PSNR','SAM','SSIM']
        self.running_img_metrics = {}
        self.running_nonan_count = {}
        self.reset()

    def reset(self):
        for metric in self.metrics: 
            self.running_nonan_count[metric] = 0
            self.running_img_metrics[metric] = np.nan

    def add(self, metrics_dict):
        for key, val in metrics_dict.items():
            # only keep a running mean of non-nan values
            if np.isnan(val): continue

            if not self.running_nonan_count[key]: 
                self.running_nonan_count[key] = 1
                self.running_img_metrics[key] = val
            else: 
                self.running_nonan_count[key]+= 1
                self.running_img_metrics[key] = (self.running_nonan_count[key]-1)/self.running_nonan_count[key] * self.running_img_metrics[key] \
                                                + 1/self.running_nonan_count[key] * val

    def value(self):
        return self.running_img_metrics