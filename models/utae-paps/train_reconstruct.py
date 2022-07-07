"""
Main script for image reconstruction experiments
Author: Patrick Ebel (github/PatrickTUM), based on the scripts of
        Vivien Sainte Fare Garnot (github/VSainteuf)
License: MIT
"""

""" TODO: HACKY HARD-CODED CHANGES TO THE SCRIPT FOR DEBUGGING, DEVELOPMENT ETC.

    changed range:
    out = model(x, batch_positions=dates).unsqueeze(1)
    out = (1+model(2*x-1, batch_positions=dates).unsqueeze(1))/2

    flags:
    changed --epochs to 200 and --val_after to 100000
    changed --model and --loss and --use_sar (so we can just copy y to x)
    changed --n_head or --d_model

    commented out: 
    if step%config.display_step==0: 
    indexing [:5 in data loader], shuffle=False in wrapper of loader

    call on ScienceCluster via:
    python train_reconstruct.py --root1 /net/cephfs/home/pebel/scratch/SEN12MSCRTS --root2 /net/cephfs/home/pebel/scratch/SEN12MSCRTS_val_test
"""

import cProfile as profile
import pstats

import argparse
import json
import os
import sys
import pprint
import time

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.utils.data as data
import torchnet as tnt
from matplotlib import pyplot as plt

from src import utils, model_utils
from src.learning.weight_init import weight_init

import torchgeometry as tgm
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
from data.dataLoader import SEN12MSCRTS
from util.utils import LossNetwork, get_perceptual_loss
from src.learning.metrics import img_metrics, avg_img_metrics

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), "util", "covweighting"))
from util.covweighting.losses.simplecovweighting_loss import SimpleCoVWeightingLoss
from util.covweighting.losses.simpleuncertainty_loss import SimpleUncertaintyLoss

parser = argparse.ArgumentParser()
# Model parameters
parser.add_argument(
    "--model",
    default="ensemble_utae",
    type=str,
    help="Type of architecture to use. Can be one of: (utae/unet3d/fpn/convlstm/convgru/uconvlstm/buconvlstm/ensemble_utae)",
)
parser.add_argument("--ensemble_m", default="1", type=int, help="numer of models m in the ensemble")


## U-TAE Hyperparameters
parser.add_argument("--encoder_widths", default="[64,64,64,128]", type=str)
parser.add_argument("--decoder_widths", default="[32,32,64,128]", type=str)
parser.add_argument("--out_conv", default="[32, 13]") # changed from [32, 20]
parser.add_argument("--out_nonlin", dest="out_nonlin", action="store_false", help="whether to apply an output nonlinearity (sigmoidal or sigmoidal (mean) & ReLU (var))") 
parser.add_argument("--use_sar", dest="use_sar", action="store_false", help="whether to use SAR or not") 
parser.add_argument("--str_conv_k", default=4, type=int)
parser.add_argument("--str_conv_s", default=2, type=int)
parser.add_argument("--str_conv_p", default=1, type=int)
parser.add_argument("--agg_mode", default="att_group", type=str)
parser.add_argument("--encoder_norm", default="group", type=str)
parser.add_argument("--decoder_norm", default="group", type=str)
parser.add_argument("--n_head", default=16, type=int, help="default value of 16, 4 for debugging")
parser.add_argument("--d_model", default=256, type=int, help="default value of 256, 64 for debugging")
parser.add_argument("--d_k", default=4, type=int)

# Set-up parameters
parser.add_argument(
    "--res_dir",
    default="./results",
    help="Path to the folder where the results should be stored",
)
parser.add_argument(    
    "--experiment_name",
    default='test_ensemble',#'utae_S1S2_t4_covweightingNoPerceptual_europe', # #'utae_S1S2_t4_covweighting_europe', 'utae_S1S2_L1SSIM_perceptual1video_1000samples', #"utae_L1SSIM_perceptual01video",
    help="Name of the current experiment, store outcomes in a subdirectory of the results folder",
)
parser.add_argument(
    "--num_workers", default=8, type=int, help="Number of data loading workers"
)
parser.add_argument("--rdm_seed", default=1, type=int, help="Random seed")
parser.add_argument(
    "--device",
    default="cuda",
    type=str,
    help="Name of device to use for tensor computations (cuda/cpu)",
)
parser.add_argument(
    "--display_step",
    default=10,
    type=int,
    help="Interval in batches between display of training metrics",
)
parser.add_argument(
    "--cache",
    dest="cache",
    action="store_true",
    help="If specified, the whole dataset is kept in RAM",
)
# Training parameters
parser.add_argument("--epochs", default=100, type=int, help="Number of epochs per fold")
parser.add_argument("--batch_size", default=5, type=int, help="Batch size")
parser.add_argument("--lr", default=0.01, type=float, help="Learning rate, e.g. 0.001")
parser.add_argument("--mono_date", default=None, type=str)
parser.add_argument("--ref_date", default="2014-04-03", type=str)
parser.add_argument(
    "--fold",
    default=None,
    type=int,
    help="Do only one of the five fold (between 1 and 5)",
)
parser.add_argument("--num_classes", default=20, type=int) # TODO: not used for reconstruction
parser.add_argument("--ignore_index", default=-1, type=int)
parser.add_argument("--pad_value", default=0, type=float)
parser.add_argument("--padding_mode", default="reflect", type=str)
parser.add_argument(
    "--val_every",
    default=1,
    type=int,
    help="Interval in epochs between two validation steps.",
)
parser.add_argument(
    "--val_after",
    default=0,
    type=int,
    help="Do validation only after that many epochs.",
)

# flags specific to SEN12MS-CR-TS
parser.add_argument("--input_t", default=4, type=int, help="number of input time points to sample, unet3d needs at least 4 time points")
parser.add_argument("--sample_type", default="cloudy_cloudfree", type=str, help="type of samples returned [cloudy_cloudfree | generic]")
parser.add_argument("--root1", default='/media/DATA/SEN12MSCRTS', type=str, help="path to your copy of SEN12MS-CR-TS")
parser.add_argument("--root2", default='~/Data/SEN12MSCRTS_val_test', type=str, help="path to your copy of SEN12MS-CR-TS validation & test splits")
parser.add_argument("--region", default="europa", type=str, help="region to (sub-)sample ROI from [all | europa]")
parser.add_argument("--input_size", default=256, type=int, help="size of input patches to (sub-)sample")
parser.add_argument("--plot_every", default=-1, type=int, help="Interval (in items) of exporting plots at validation or test time. Set -1 to disable")
parser.add_argument("--loss", default="covweighting", type=str, help="Image reconstruction loss to utilize [l1|l2|ssim|combined|covweighting|uncertainty].")
parser.add_argument("--perceptual", default=os.path.expanduser("~/Documents/models/vgg16_13C.pth"), type=str, help="Path to VGG16 checkpoint, no perceptual loss if passing None")
parser.add_argument("--layers_perc", default="video", type=str, help="layers to compute perceptual loss over [dip|video|original|experimental]")
parser.add_argument("--debug", dest="debug", action="store_true", help="whether to debug and profile code or not") 

# flags specific to loss weighting
parser.add_argument("--mean_sort", default='full', type=str, help="cov weighting [full|decay]")
parser.add_argument("--mean_decay_param", default=1.0, type=float, help="What decay to use with mean decay")


list_args = ["encoder_widths", "decoder_widths", "out_conv"]
parser.set_defaults(cache=False)

config = parser.parse_args()
for k, v in vars(config).items():
    if k in list_args and v is not None:
        v = v.replace("[", "")
        v = v.replace("]", "")
        config.__setattr__(k, list(map(int, v.split(","))))

if 'ensemble' in config.model: 
    config.loss = 'NLL'         # negative log likelihood loss
    config.out_conv[-1] *= 2    # predict means and variances
pprint.pprint(config)

# validate input flags
#assert config.num_classes == config.out_conv[-1]
isPow2 = lambda x: (x & (x-1) == 0) and x != 0
assert isPow2(config.input_size) and 0<config.input_size<=256

# instantiate tensorboard logger
writer = SummaryWriter(os.path.join("runs", config.experiment_name))


def plot_img(imgs, mod, epoch, split, file_id=None):
    imgs = imgs.cpu().numpy()
    for tdx, img in enumerate(imgs): # iterate over temporal dimension
        if mod in ["pred", "in", "target"]:
            rgb = [3,2,1] if img.shape[0]==13 else [5,4,3]
            img = np.clip(img[rgb, ...], 0, 1)
        elif mod == "s1":
            img = np.clip(img[[0], ...], 0, 1)
        elif mod == "s2":
            img = np.clip(img[[3,2,1], ...], 0, 1)
        elif mod == "mask":
            img = np.clip(img[[0], ...], 0, 1)
        if file_id is not None: # export into file name
            plot_dir = os.path.join(config.res_dir, config.experiment_name, 'plots', f'epoch_{epoch}', f'{split}')
            if not os.path.exists(plot_dir): os.makedirs(plot_dir)
            plt.imsave(os.path.join(plot_dir, f'img-{file_id}_{mod}_t-{tdx}.png'), np.moveaxis(img,0,-1), dpi=10)
    return img

# TODO: IN: batch, device; OUT: x, y, in_m, dates
def prepare_data(batch, device, config):
    in_S2 = recursive_todevice(batch['input']['S2'], device)
    in_S2_td = recursive_todevice(batch['input']['S2 TD'], device)
    if config.batch_size>1: in_S2_td = torch.stack((in_S2_td)).T
    in_m  = torch.stack(recursive_todevice(batch['input']['masks'], device)).swapaxes(0,1)
    #target_S1 = recursive_todevice(batch['target']['S1'], device)
    target_S2 = recursive_todevice(batch['target']['S2'], device)
    #target_S1_td = recursive_todevice(batch['target']['S1 TD'], device)
    #target_S2_td = recursive_todevice(batch['target']['S2 TD'], device)
    #target_m  = recursive_todevice(batch['target']['masks'], device)
    y     = torch.cat(target_S2,dim=0).unsqueeze(1)

    if config.use_sar: 
        in_S1 = recursive_todevice(batch['input']['S1'], device)
        in_S1_td = recursive_todevice(batch['input']['S1 TD'], device)
        if config.batch_size>1: in_S1_td = torch.stack((in_S1_td)).T
        x     = torch.cat((torch.stack(in_S1,dim=1), torch.stack(in_S2,dim=1)),dim=2)
        dates = torch.stack((torch.tensor(in_S1_td),torch.tensor(in_S2_td))).float().mean(dim=0).to(device)
    else:
        x     = torch.stack(in_S2,dim=1)
        dates = torch.tensor(in_S2_td).float().to(device)

    if config.input_size < 256: # batch sub-samples if mosaicing patches
        x_mosaic = x.unfold(4, config.input_size, config.input_size).unfold(3, config.input_size, config.input_size)
        x_batch  = x_mosaic.reshape(-1, config.input_t, x.shape[2], config.input_size, config.input_size).swapaxes(-1,-2)
        y_mosaic = y.unfold(4, config.input_size, config.input_size).unfold(3, config.input_size, config.input_size)
        y_batch  = y_mosaic.reshape(-1, 1, y.shape[2], config.input_size, config.input_size).swapaxes(-1,-2)
        m_mosaic = in_m.unfold(3, config.input_size, config.input_size).unfold(2, config.input_size, config.input_size)
        m_batch  = m_mosaic.reshape(-1, config.input_t, config.input_size, config.input_size).swapaxes(-1,-2)

        x, y, in_m, dates = x_batch, y_batch, m_batch, dates.expand(x_batch.shape[0],-1)
    return x, y, in_m, dates

def get_loss(config):
    if 'ensemble' in config.model:
        criterion = nn.GaussianNLLLoss(reduction='mean')
    else:
        if config.loss=="l1":
            criterion = nn.L1Loss()
        elif config.loss=="l2":
            criterion = nn.MSELoss()
        elif config.loss=="ssim": #  SSIM loss is SDSIM: (1-SSIM)/2
            criterion1 = tgm.losses.SSIM(5, reduction='mean')
            # note: ssim can currently only handle 3D (unbatched) or 4D (batched)
            criterion = lambda pred, targ: criterion1(pred[:,0,...], targ[:,0,...])
        elif config.loss=="combined": #  SSIM loss is SDSIM: (1-SSIM)/2
            # naive 1:1 weighting
            criterion1 = nn.L1Loss()
            criterion2 = tgm.losses.SSIM(5, reduction='mean')
            criterion = lambda pred, targ: criterion1(pred, targ) + criterion2(pred[:,0,...], targ[:,0,...])
        elif config.loss=="covweighting":
            # coefficient of variations weighted loss
            criterion = SimpleCoVWeightingLoss(config)
        elif config.loss=="uncertainty":
            # Kendall et al's uncertainty weighting
            criterion = SimpleUncertaintyLoss(config)
        else: raise NotImplementedError
    return criterion


def log_train(writer, config, criterion, loss, model, step, x, out, y, name=''):
    if name != '': name == 'model_{name}/' 
    if step%config.display_step==0: 
        writer.add_scalar(f'Loss/train/{name}{config.loss}', loss, step)
        if config.loss in ["covweighting", "uncertainty"]: 
            for idx, alpha in enumerate(criterion.alphas): # individual loss weightings
                loss_n  = ["L1","SSIM"]#, "perceptual"]
                writer.add_scalar(f'Loss/train/{name}alpha_{loss_n[idx]}', alpha, step)
                writer.add_scalar(f'Loss/train/{name}{loss_n[idx]}', criterion.weighted_l[idx], step)
            #criterion.running_mean_l, criterion.running_mean_L
    # separately evaluate perceptual loss, if included
    if config.perceptual and config.perceptual not in ["none", "None"] and config.loss not in ["covweighting", "uncertainty"] and not 'ensemble' in config.model: 
        perceptual = get_perceptual_loss(model.perceptual, out, y)
        if step%config.display_step==0: 
            writer.add_scalar(f'Loss/train/{name}perceptual', perceptual, step)
        loss += 1 *perceptual # add perceptual loss, eventually rescale
    if step%config.display_step==0: 
        writer.add_scalar(f'Loss/train/{name}total', loss, step)
        # use add_images for batch-wise adding across temporal dimension
        if config.use_sar:
            writer.add_image(f'Img/train/{name}in_s1', x[0,:,[0], ...], step, dataformats='NCHW')
            writer.add_image(f'Img/train/{name}in_s2', x[0,:,[5,4,3], ...], step, dataformats='NCHW')
        else:
            writer.add_image(f'Img/train/{name}in_s2', x[0,:,[3,2,1], ...], step, dataformats='NCHW')
        writer.add_image(f'Img/train/{name}out', out[0,0,[3,2,1], ...], step, dataformats='CHW')
        writer.add_image(f'Img/train/{name}y', y[0,0,[3,2,1], ...], step, dataformats='CHW')


def iterate(
    model, data_loader, criterion, config, optim=None, mode="train", epoch=None, device=None):
    loss_meter = tnt.meter.AverageValueMeter()
    img_meter  = avg_img_metrics()
    if not optim:
        optimizer, scheduler = (None, None)
    else:
        optimizer, scheduler = optim[0], optim[1]

    t_start = time.time()
    for i, batch in enumerate(tqdm(data_loader)):
        step = (epoch-1)*len(data_loader)+i

        if config.sample_type == 'cloudy_cloudfree':
            # TODO: IN: batch, device; OUT: x, y, in_m, dates
            x, y, in_m, dates = prepare_data(batch, device, config)
        else: raise NotImplementedError

        if mode != "train": # val or test
            with torch.no_grad():
                if 'ensemble' in config.model:
                    out = [model_m(x, batch_positions=dates) for model_m in model]
                    var = [out_m[:, :, 13:, ...] for out_m in out]
                    out = [out_m[:, :, :13, ...] for out_m in out]
                    # approximate 1 Gaussian by mixture parameter ensembling
                    mean_ensemble = 1/config.ensemble_m * torch.sum(torch.stack(out), dim=0)
                    var_ensemble  = 1/config.ensemble_m * torch.sum(torch.stack(var) + torch.stack(out)**2, dim=0) - mean_ensemble**2
                    out, var = mean_ensemble, var_ensemble
                else:
                    out = model(x, batch_positions=dates)
                batch_size = y.size()[0]
                for bdx in range(batch_size):
                    # note: for ensembles, models are averaged by now
                    #       so we can evaluate a single prediction as usual
                    extended_metrics = img_metrics(y[bdx], out[bdx], in_m[bdx])
                    img_meter.add(extended_metrics)
                    idx = (i*batch_size+bdx) # plot and export every k-th item
                    if config.plot_every>0 and idx % config.plot_every == 0:
                        plot_img(x[bdx], 'in', epoch, mode, file_id=idx)
                        plot_img(out[bdx], 'pred', epoch, mode, file_id=idx)
                        plot_img(y[bdx], 'target', epoch, mode, file_id=idx)
        else: # training
            if isinstance(optimizer, list):
                for optim in optimizer: optim.zero_grad()
            else:
                optimizer.zero_grad()
            
            if torch.isnan(x).sum(): print("Warning: NaN encountered") # TODO

            if 'ensemble' in config.model:
                out = [model_m(x, batch_positions=dates) for model_m in model]
                var = [out_m[:, :, 13:, ...] for out_m in out]
                out = [out_m[:, :, :13, ...] for out_m in out]
            else: out = model(x, batch_positions=dates)
            #"""" #TODO plotting even at train time
            if config.plot_every>0:
                plot_out = out.detach()
                batch_size = y.size()[0]
                for bdx in range(batch_size):
                    idx = (i*batch_size+bdx) # plot and export every k-th item
                    if idx % config.plot_every == 0:
                    #if config.plot_every>0 and idx % config.plot_every == 0:
                        plot_img(x[bdx], 'in', epoch, mode, file_id=i)
                        plot_img(plot_out[bdx], 'pred', epoch, mode, file_id=i)
                        plot_img(y[bdx], 'target', epoch, mode, file_id=i)
            #""""

        # compute loss
        if 'ensemble' in config.model:
            if mode != "train": # val or test
                # evaluate the ensembled predictions
                loss = criterion(out, y, var)
            else: # train
                # train each ensemble member individually
                loss = [criterion(out_m, y, var[m]) for m, out_m in enumerate(out)]
        else:
            if config.loss in ["covweighting", "uncertainty"]: 
                # handle SimpleCoVWeightingLoss separately
                loss = criterion.forward(out, y)
            else: loss = criterion(out, y)

        if mode == "train":
            # pass: writer, config, criterion, loss, model, step, x, out, y
            if isinstance(model, list):
                for m in range(len(model)):
                    log_train(writer, config, criterion, loss[m], model[m], step, x, out[m], y, name=m)
                    loss[m].backward()
                    optimizer[m].step()
            else:
                log_train(writer, config, criterion, loss, model, step, x, out, y)
                loss.backward()
                optimizer.step()

        # log the loss
        if isinstance(loss, list):
            for loss_m in loss: loss_meter.add(loss_m.item())
        else: loss_meter.add(loss.item())

    if mode == "train": # after each epoch, update lr acc. to scheduler
        if isinstance(optimizer, list):
            for m, optim in enumerate(optimizer):
                current_lr = optim.state_dict()['param_groups'][0]['lr']
                writer.add_scalar(f'Etc/train/model_{m}/lr', current_lr, step)
                scheduler[m].step()
        else:
            current_lr = optimizer.state_dict()['param_groups'][0]['lr']
            writer.add_scalar('Etc/train/lr', current_lr, step)
            scheduler.step()

    t_end = time.time()
    total_time = t_end - t_start
    print("Epoch time : {:.1f}s".format(total_time))
    metrics = {
        "{}_loss".format(mode): loss_meter.value()[0],
        "{}_epoch_time".format(mode): total_time,
    }

    if mode == "test" or mode == "val":
        for key, val in img_meter.value().items():
            writer.add_scalar(f'Loss/{mode}/{key}', val, step)
        # use add_images for batch-wise adding across temporal dimension
        if config.use_sar:
            writer.add_image(f'Img/{mode}/in_s1', x[0,:,[0], ...], step, dataformats='NCHW')
            writer.add_image(f'Img/{mode}/in_s2', x[0,:,[5,4,3], ...], step, dataformats='NCHW')
        else:
            writer.add_image(f'Img/{mode}/in_s2', x[0,:,[3,2,1], ...], step, dataformats='NCHW')
        writer.add_image(f'Img/{mode}/out', out[0,0,[3,2,1], ...], step, dataformats='CHW')
        writer.add_image(f'Img/{mode}/y', y[0,0,[3,2,1], ...], step, dataformats='CHW')
        return metrics, img_meter.value()
    else:
        return metrics


def recursive_todevice(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, dict):
        return {k: recursive_todevice(v, device) for k, v in x.items()}
    else:
        return [recursive_todevice(c, device) for c in x]


def prepare_output(config):
    os.makedirs(os.path.join(config.res_dir, config.experiment_name), exist_ok=True)
    #for fold in range(1, 6):
    #    os.makedirs(os.path.join(config.res_dir, config.experiment_name, "Fold_{}".format(fold)), exist_ok=True)


def checkpoint(fold, log, config):
    with open(
        #os.path.join(config.res_dir, config.experiment_name, "Fold_{}".format(fold), "trainlog.json"), "w"
        os.path.join(config.res_dir, config.experiment_name, "trainlog.json"), "w"
    ) as outfile:
        json.dump(log, outfile, indent=4)


def save_results(metrics, config):
    with open(
        #os.path.join(config.res_dir, config.experiment_name, "Fold_{}".format(fold), "test_metrics.json"), "w"
        os.path.join(config.res_dir, config.experiment_name, "test_metrics.json"), "w"
    ) as outfile:
        json.dump(metrics, outfile, indent=4)

def save_model(config, epoch, model, optimizer, scheduler, name):
    torch.save(
        {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
        os.path.join(
            config.res_dir, config.experiment_name, f"{name}.pth.tar"
        ),
    )

def load_model(config, model, name):
    model.load_state_dict(
        torch.load(
            os.path.join(
                config.res_dir, config.experiment_name, f"{name}.pth.tar"
            )
        )["state_dict"]
    )

def main(config):
    # fix all RNG seeds
    np.random.seed(config.rdm_seed)
    torch.manual_seed(config.rdm_seed)
    prepare_output(config)
    device = torch.device(config.device)

    #for fold, (train_folds, val_fold, test_fold) in enumerate(fold_sequence):

    # define data sets
    fold        = 0 # n-fold cross-val may be to cumbersome for SEN12MSCRTS, hardcoding this var for now to avoid breaking downstream code
    dt_train    = SEN12MSCRTS(os.path.expanduser(config.root1), split='train', region=config.region, sample_type=config.sample_type , n_input_samples=config.input_t, import_data_path=None)
    dt_val      = SEN12MSCRTS(os.path.expanduser(config.root2), split='val', region=config.region, sample_type=config.sample_type , n_input_samples=config.input_t, import_data_path=None) 
    dt_test     = SEN12MSCRTS(os.path.expanduser(config.root2), split='test', region=config.region, sample_type=config.sample_type , n_input_samples=config.input_t, import_data_path=None)

    collate_fn = lambda x: utils.pad_collate(x, pad_value=config.pad_value) # padding across batch elements
    train_loader = data.DataLoader(
        dt_train,
        batch_size=config.batch_size,
        shuffle=True,
        #num_workers=config.num_workers,
        #drop_last=True,
        #collate_fn=collate_fn,
    )
    val_loader = data.DataLoader(
        dt_val,
        batch_size=config.batch_size,
        shuffle=False,
        #num_workers=config.num_workers,
        #drop_last=True,
        #collate_fn=collate_fn,
    )
    test_loader = data.DataLoader(
        dt_test,
        batch_size=config.batch_size,
        shuffle=False,
        #num_workers=config.num_workers,
        #drop_last=True,
        #collate_fn=collate_fn,
    )

    print(
        "Train {}, Val {}, Test {}".format(len(dt_train), len(dt_val), len(dt_test))
    )

    # model definition
    model = model_utils.get_model(config, mode="reconstruct")
    if isinstance(model, list):
        config.N_params = [utils.get_ntrainparams(model_m) for model_m in model]
        for mdx, model_m in enumerate(model):
            print(f"\n\nTrainable layers model {mdx+1}:")
            for name, p in model_m.named_parameters():
                if p.requires_grad: print(f"\t{name}")
        model = [model_m.to(device) for model_m in model]
        # note: this results in different initial weights, as required by deep ensembles
        # print(list(model[0].parameters())[0])
        # TODO: FiLM ensemble may require copying // sharing weights
        for model_m in model: model_m.apply(weight_init)
    else:
        config.N_params = utils.get_ntrainparams(model)
        print("\n\nTrainable layers:")
        for name, p in model.named_parameters():
            if p.requires_grad: print(f"\t{name}")
        model = model.to(device)
        model.apply(weight_init)
    with open(os.path.join(config.res_dir, config.experiment_name, "conf.json"), "w") as file:
        file.write(json.dumps(vars(config), indent=4))
    print(model)
    print("TOTAL TRAINABLE PARAMETERS :", config.N_params)

    # perceptual loss
    perceptual_layers = {   'dip': [11, 20, 29],
                            'video': [3, 8, 15],
                            'original': [3, 8, 15, 22],
                            'experimental': [8, 15, 22, 29]
                        }

    if config.perceptual and config.perceptual not in ["none", "None"] and config.loss not in ["NLL", "covweighting", "uncertainty"]: 
        # make the perceptual network a property of the main model, this is a bit ad-hoc
        model.perceptual = LossNetwork(config.perceptual, perceptual_layers[config.layers_perc], config.device)

    # Optimizer and Loss
    # TODO: pick a nicer schedule plx, see https://pytorch.org/docs/stable/optim.html
    if isinstance(model, list):
        optimizer = [torch.optim.Adam(model_m.parameters(), lr=config.lr) for model_m in model]
        scheduler = [torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.9) for optim in optimizer]
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    criterion = get_loss(config)
    if config.loss == 'NLL': # maximize loss function
        is_better, best_loss = lambda new, prev: new >= prev, float("-inf")
    else:                    # minimize loss function
        is_better, best_loss = lambda new, prev: new <= prev, float("inf")

    # Training loop
    trainlog = {}
    for epoch in range(1, config.epochs + 1):
        print("EPOCH {}/{}".format(epoch, config.epochs))

        if config.debug:
            prof = profile.Profile()
            prof.enable()

        if isinstance(model, list):
            for model_m in model: model_m.train()
        else: model.train()
        if config.loss in ["covweighting", "uncertainty"]: criterion.to_train()
        train_metrics = iterate(
            model,
            data_loader=train_loader,
            criterion=criterion,
            config=config,
            optim=(optimizer, scheduler),
            mode="train",
            epoch=epoch,
            device=device,
        )

        if config.debug:
            prof.disable()
            stats = pstats.Stats(prof).strip_dirs().sort_stats("cumtime")
            stats.print_stats(25) # top k rows
        
        # do regular validation steps at the end of each training epoch
        if epoch % config.val_every == 0 and epoch > config.val_after:
            print("Validation . . . ")
            if isinstance(model, list):
                for model_m in model: model_m.eval()
            else: model.eval()
            if config.loss in ["covweighting", "uncertainty"]: criterion.to_eval()
            val_metrics, val_img_metrics = iterate(
                                            model,
                                            data_loader=val_loader,
                                            criterion=criterion,
                                            config=config,
                                            optim=(optimizer, scheduler),
                                            mode="val",
                                            epoch=epoch,
                                            device=device,
                                        )

            print(f'Loss {val_metrics["val_loss"]}')
            print(f'validation image metrics: {val_img_metrics}')

            trainlog[epoch] = {**train_metrics, **val_metrics}
            checkpoint(fold+1, trainlog, config)
            # checkpoint best model
            if is_better(val_metrics["val_loss"], best_loss):
                best_loss = val_metrics["val_loss"]
                if isinstance(model, list):
                    for m, model_m in enumerate(model): save_model(config, epoch, model_m, optimizer[m], scheduler[m], f"model_{m}")
                else:
                    save_model(config, epoch, model, optimizer, scheduler, "model")
        else:
            trainlog[epoch] = {**train_metrics}
            checkpoint(fold+1, trainlog, config)

        # always checkpoint the current epoch's model
        if isinstance(model, list):
            for m, model_m in enumerate(model): save_model(config, epoch, model_m, optimizer[m], scheduler[m], f"model_{m}_epoch_{epoch}")
        else:
            save_model(config, epoch, model, optimizer, scheduler, f"model_epoch_{epoch}")

    # following training, test on hold-out data
    print("Testing best epoch . . .")
    if isinstance(model, list):
        for m, model_m in enumerate(model): load_model(config, model_m, f"model_{m}")
    else:
        load_model(config, model, "model")
    
    if isinstance(model, list):
        for model_m in model: model_m.eval()
    else: model.eval()
    if config.loss in ["covweighting", "uncertainty"]: criterion.to_eval()

    test_metrics, test_img_metrics = iterate(
                                    model,
                                    data_loader=test_loader,
                                    criterion=criterion,
                                    config=config,
                                    optim=(optimizer, scheduler),
                                    mode="test",
                                    epoch=epoch,
                                    device=device,
                                )
    print(f'Loss {test_metrics["test_loss"]}')
    print(f' test image metrics: {test_img_metrics}')
    save_results(test_metrics, config)

    # close tensorboard logging
    writer.close()

if __name__ == "__main__":
    main(config)