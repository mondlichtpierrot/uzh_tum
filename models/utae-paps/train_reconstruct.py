"""
Main script for image reconstruction experiments
Author: Patrick Ebel (github/PatrickTUM), based on the scripts of
        Vivien Sainte Fare Garnot (github/VSainteuf)
License: MIT
"""

"""
out = model(x, batch_positions=dates).unsqueeze(1)
out = (1+model(2*x-1, batch_positions=dates).unsqueeze(1))/2

changed --epochs to 200 and --val_after to 100000
changed --model and --loss and --use_sar (so we can just copy y to x)
commented out: if step%config.display_step==0: 

indexing [:5 in data loader], shuffle=False in wrapper of loader
"""
import argparse
import json
import os
import sys
import pickle as pkl
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
#from src.dataset import PASTIS_Dataset
from src.learning.metrics import confusion_matrix_analysis
#from src.learning.miou import IoU
from src.learning.weight_init import weight_init

import torchgeometry as tgm
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
from data.dataLoader import SEN12MSCRTS
from util.util import LossNetwork, get_perceptual_loss
from src.learning.metrics import img_metrics, avg_img_metrics

parser = argparse.ArgumentParser()
# Model parameters
parser.add_argument(
    "--model",
    default="utae",
    type=str,
    help="Type of architecture to use. Can be one of: (utae/unet3d/fpn/convlstm/convgru/uconvlstm/buconvlstm)",
)
## U-TAE Hyperparameters
parser.add_argument("--encoder_widths", default="[64,64,64,128]", type=str)
parser.add_argument("--decoder_widths", default="[32,32,64,128]", type=str)
parser.add_argument("--out_conv", default="[32, 13]") # changed from [32, 20]
parser.add_argument("--out_sigm", dest="out_sigm", action="store_false", help="whether to apply an output nonlinearity") 
parser.add_argument("--use_sar", dest="use_sar", action="store_true", help="whether to use SAR or not") 
parser.add_argument("--str_conv_k", default=4, type=int)
parser.add_argument("--str_conv_s", default=2, type=int)
parser.add_argument("--str_conv_p", default=1, type=int)
parser.add_argument("--agg_mode", default="att_group", type=str)
parser.add_argument("--encoder_norm", default="group", type=str)
parser.add_argument("--decoder_norm", default="group", type=str)
parser.add_argument("--n_head", default=16, type=int, help="default value of 16, 4 for debugging") # TODO
parser.add_argument("--d_model", default=256, type=int, help="default value of 256, 64 for debugging") # TODO
parser.add_argument("--d_k", default=4, type=int)

# Set-up parameters
parser.add_argument(
    "--dataset_folder",
    default="",
    type=str,
    help="Path to the folder where the results are saved.",
)
parser.add_argument(
    "--res_dir",
    default="./results",
    help="Path to the folder where the results should be stored",
)
parser.add_argument(
    "--experiment_name",
    default='dbg', #"utae_L1SSIM_perceptual01video",
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
parser.add_argument("--epochs", default=500, type=int, help="Number of epochs per fold") ############### TODO
parser.add_argument("--batch_size", default=5, type=int, help="Batch size")
parser.add_argument("--lr", default=0.01, type=float, help="Learning rate, e.g. 0.001") # TODO
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
    default=1000000000000000000000000000000000000000000000000000000,
    type=int,
    help="Do validation only after that many epochs.",
)

# flags specific to SEN12MS-CR-TS
parser.add_argument("--input_t", default=4, type=int, help="number of input time points to sample, unet3d needs at least 4 time points")
parser.add_argument("--sample_type", default="cloudy_cloudfree", type=str, help="type of samples returned [cloudy_cloudfree | generic]")
parser.add_argument("--root", default='/media/DATA/SEN12MSCRTS', type=str, help="path to your copy of SEN12MS-CR-TS")
parser.add_argument("--region", default="europa", type=str, help="region to (sub-)sample ROI from [all | europa]")
parser.add_argument("--input_size", default=256, type=int, help="size of input patches to (sub-)sample")
parser.add_argument("--plot_every", default=1, type=int, help="Interval (in items) of exporting plots at validation or test time.")
parser.add_argument("--loss", default="combined", type=str, help="Image reconstruction loss to utilize [l1|l2|ssim|combined].")
parser.add_argument("--perceptual", default=os.path.expanduser('~/Documents/models/vgg16_13C.pth'), type=str, help="Path to VGG16 checkpoint, no perceptual loss if passing None")
parser.add_argument("--layers_perc", default="video", type=str, help="layers to compute perceptual loss over [dip|video|original|experimental]")

list_args = ["encoder_widths", "decoder_widths", "out_conv"]
parser.set_defaults(cache=False)

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

def iterate(
    model, data_loader, criterion, config, optimizer=None, mode="train", epoch=None, device=None):
    loss_meter = tnt.meter.AverageValueMeter()
    img_meter = avg_img_metrics()

    t_start = time.time()
    for i, batch in enumerate(tqdm(data_loader)):
        step = (epoch-1)*len(data_loader)+i
        if config.sample_type == 'cloudy_cloudfree':
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

        else: raise NotImplementedError

        if mode != "train":
            with torch.no_grad():
                out = model(x, batch_positions=dates).unsqueeze(1)
                batch_size = y.size()[0]
                for bdx in range(batch_size):
                    extended_metrics = img_metrics(y[bdx], out[bdx], in_m[bdx])
                    img_meter.add(extended_metrics)
                    idx = (i*batch_size+bdx) # plot and export every k-th item
                    if idx % config.plot_every == 0:
                        plot_img(x[bdx], 'in', epoch, mode, file_id=idx)
                        plot_img(out[bdx], 'pred', epoch, mode, file_id=idx)
                        plot_img(y[bdx], 'target', epoch, mode, file_id=idx)
                        
        else:
            optimizer.zero_grad()
            if torch.isnan(x).sum(): print("DOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO") # TODO
            #x = y.repeat((1,config.input_t,1,1,1)) # TODO setting where input=target patch, just learn identity mapping
            #x = 0.725*torch.ones_like(x, device=device) # TODO setting where we set input to a constant
            #y = 0.725*torch.ones_like(y, device=device) # TODO setting where we set target to a constant
            out = model(x, batch_positions=dates).unsqueeze(1)
            #"""" #TODO plotting even at train time
            plot_out = out.detach()
            batch_size = y.size()[0]
            for bdx in range(batch_size):
                idx = (i*batch_size+bdx) # plot and export every k-th item
                if idx % config.plot_every == 0:
                    plot_img(x[bdx], 'in', epoch, mode, file_id=i)
                    plot_img(plot_out[bdx], 'pred', epoch, mode, file_id=i)
                    plot_img(y[bdx], 'target', epoch, mode, file_id=i)
            #""""

        loss = criterion(out, y)
        if mode == "train":
            #if step%config.display_step==0: 
            writer.add_scalar(f'Loss/train/{config.loss}', loss, step)
            if config.perceptual and config.perceptual!="none": 
                perceptual = get_perceptual_loss(model.perceptual, out, y) # get_perceptual_loss(net, fake, real)
                #if step%config.display_step==0: 
                writer.add_scalar('Loss/train/perceptual', perceptual, step)
                # rescale perceptual loss prop. to lr
                loss += 0.1*config.lr*perceptual
            #if step%config.display_step==0: 
            writer.add_scalar('Loss/train/total', loss, step)
            # use add_images for batch-wise adding across temporal dimension
            writer.add_image('Img/train/in', x[0,:,[3,2,1], ...], step, dataformats='NCHW')
            writer.add_image('Img/train/out', out[0,0,[3,2,1], ...], step, dataformats='CHW')
            writer.add_image('Img/train/y', y[0,0,[3,2,1], ...], step, dataformats='CHW')
            loss.backward()
            optimizer.step()

        #with torch.no_grad():
        #    pred = out.argmax(dim=1)
        #iou_meter.add(pred, y)
        loss_meter.add(loss.item())

        print(f'Prediction min {out.min()}, max {out.max()}, mean {out.mean()}, std {out.std()}. Loss {loss}.') # TODO

        # report training metrics on terminal
        if (i + 1) % config.display_step == 0:
            #miou, acc = iou_meter.get_miou_acc()
            #print("Step [{}/{}], Loss: {:.4f}, Acc : {:.2f}, mIoU {:.2f}".format(
            #        i + 1, len(data_loader), loss_meter.value()[0], acc, miou))
            print("Step [{}/{}], Loss: {:.4f}".format(
                    i + 1, len(data_loader), loss_meter.value()[0]))

    t_end = time.time()
    total_time = t_end - t_start
    print("Epoch time : {:.1f}s".format(total_time))
    #miou, acc = iou_meter.get_miou_acc()
    metrics = {
        #"{}_accuracy".format(mode): acc,
        "{}_loss".format(mode): loss_meter.value()[0],
        #"{}_IoU".format(mode): miou,
        "{}_epoch_time".format(mode): total_time,
    }

    if mode == "test" or mode == "val":
        #for keys, vals in extended_metrics.items():
        #    metrics[f"{mode}_{keys}"] = vals
        return metrics, img_meter.value() #, iou_meter.conf_metric.value()  # confusion matrix
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
    for fold in range(1, 6):
        os.makedirs(os.path.join(config.res_dir, config.experiment_name, "Fold_{}".format(fold)), exist_ok=True)


def checkpoint(fold, log, config):
    with open(
        os.path.join(config.res_dir, config.experiment_name, "Fold_{}".format(fold), "trainlog.json"), "w"
    ) as outfile:
        json.dump(log, outfile, indent=4)


def save_results(fold, metrics, config):
    with open(
        os.path.join(config.res_dir, config.experiment_name, "Fold_{}".format(fold), "test_metrics.json"), "w"
    ) as outfile:
        json.dump(metrics, outfile, indent=4)


def overall_performance(config):
    
    cm = np.zeros((config.num_classes, config.num_classes))
    for fold in range(1, 6):
        cm += pkl.load(
            open(
                os.path.join(config.res_dir, config.experiment_name, "Fold_{}".format(fold), "conf_mat.pkl"),
                "rb",
            )
        )

    if config.ignore_index is not None:
        cm = np.delete(cm, config.ignore_index, axis=0)
        cm = np.delete(cm, config.ignore_index, axis=1)

    _, perf = confusion_matrix_analysis(cm)

    print("Overall performance:")
    print("Acc: {},  IoU: {}".format(perf["Accuracy"], perf["MACRO_IoU"]))

    with open(os.path.join(config.res_dir, config.experiment_name, "overall.json"), "w") as file:
        file.write(json.dumps(perf, indent=4))


def main(config):
    # fix all RNG seeds
    np.random.seed(config.rdm_seed)
    torch.manual_seed(config.rdm_seed)
    prepare_output(config)
    device = torch.device(config.device)

    #for fold, (train_folds, val_fold, test_fold) in enumerate(fold_sequence):

    # define data sets
    fold        = 0 # n-fold cross-val may be to cumbersome for SEN12MSCRTS, hardcoding this var for now to avoid breaking downstream code
    dt_train    = SEN12MSCRTS(os.path.expanduser(config.root), split='train', region=config.region, sample_type=config.sample_type , n_input_samples=config.input_t, import_data_path=None)
    dt_val      = SEN12MSCRTS(os.path.expanduser('~/Data/SEN12MSCR_val_test'), split='val', region=config.region, sample_type=config.sample_type , n_input_samples=config.input_t, import_data_path=None) 
    dt_test     = SEN12MSCRTS(os.path.expanduser('~/Data/SEN12MSCR_val_test'), split='test', region=config.region, sample_type=config.sample_type , n_input_samples=config.input_t, import_data_path=None)

    collate_fn = lambda x: utils.pad_collate(x, pad_value=config.pad_value) # padding across batch elements
    train_loader = data.DataLoader(
        dt_train,
        batch_size=config.batch_size,
        shuffle=False,############################################################True,
        #drop_last=True,
        #collate_fn=collate_fn,
    )
    val_loader = data.DataLoader(
        dt_val,
        batch_size=config.batch_size,
        shuffle=False,
        #drop_last=True,
        #collate_fn=collate_fn,
    )
    test_loader = data.DataLoader(
        dt_test,
        batch_size=config.batch_size,
        shuffle=False,
        #drop_last=True,
        #collate_fn=collate_fn,
    )

    print(
        "Train {}, Val {}, Test {}".format(len(dt_train), len(dt_val), len(dt_test))
    )

    # Model definition
    model = model_utils.get_model(config, mode="reconstruct")
    config.N_params = utils.get_ntrainparams(model)
    with open(os.path.join(config.res_dir, config.experiment_name, "conf.json"), "w") as file:
        file.write(json.dumps(vars(config), indent=4))
    print(model)
    print("TOTAL TRAINABLE PARAMETERS :", config.N_params)
    print("Trainable layers:")
    for name, p in model.named_parameters():
        if p.requires_grad:
            print(name)
    model = model.to(device)
    model.apply(weight_init)

    # perceptual loss
    perceptual_layers = {   'dip': [11, 20, 29],
                            'video': [3, 8, 15],
                            'original': [3, 8, 15, 22],
                            'experimental': [8, 15, 22, 29]
                        }

    if config.perceptual and config.perceptual!="none": 
        # make the perceptual network a property of the main model, this is a bit ad-hoc
        model.perceptual = LossNetwork(config.perceptual, perceptual_layers[config.layers_perc], config.device)

    # Optimizer and Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    if config.loss=="l1":
        criterion = nn.L1Loss()
        best_loss = float("inf")
        is_better = lambda new, prev: new <= prev
    elif config.loss=="l2":
        criterion = nn.MSELoss()
        best_loss = float("inf")
        is_better = lambda new, prev: new <= prev
    elif config.loss=="ssim": #  SSIM loss is SDSIM: (1-SSIM)/2
        criterion = tgm.losses.SSIM(5, reduction='mean')
        # note: ssim can currently only handle 3D (unbatched) or 4D (batched)
        criterion = lambda pred, targ: criterion(pred[:,0,...], targ[:,0,...])
        best_loss = float("inf")
        is_better = lambda new, prev: new <= prev
    if config.loss=="combined": #  SSIM loss is SDSIM: (1-SSIM)/2
        # naive 1:1 weighting
        criterion1 = nn.L1Loss()
        criterion2 = tgm.losses.SSIM(5, reduction='mean')
        criterion = lambda pred, targ: criterion1(pred, targ) + criterion2(pred[:,0,...], targ[:,0,...])
        best_loss = float("inf")
        is_better = lambda new, prev: new <= prev

    # Training loop
    trainlog = {}
    for epoch in range(1, config.epochs + 1):
        print("EPOCH {}/{}".format(epoch, config.epochs))

        model.train()
        train_metrics = iterate(
            model,
            data_loader=train_loader,
            criterion=criterion,
            config=config,
            optimizer=optimizer,
            mode="train",
            epoch=epoch,
            device=device,
        )
        # do regular validation steps
        if epoch % config.val_every == 0 and epoch > config.val_after:
            print("Validation . . . ")
            model.eval()
            val_metrics, val_img_metrics = iterate(
                                            model,
                                            data_loader=val_loader,
                                            criterion=criterion,
                                            config=config,
                                            optimizer=optimizer,
                                            mode="val",
                                            epoch=epoch,
                                            device=device,
                                        )

            print(f'Loss {val_metrics["val_loss"]}')
            print(f' validation image metrics: {val_img_metrics}')
            #val_metrics["val_loss"],
            #val_metrics["val_accuracy"],
            #val_metrics["best_loss"],

            trainlog[epoch] = {**train_metrics, **val_metrics}
            checkpoint(fold+1, trainlog, config)
            if is_better(val_metrics["val_loss"], best_loss):
                best_loss = val_metrics["val_loss"]
                torch.save(
                    {
                        "epoch": epoch,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    },
                    os.path.join(
                        config.res_dir, config.experiment_name, "Fold_{}".format(fold + 1), "model.pth.tar"
                    ),
                )
        else:
            trainlog[epoch] = {**train_metrics}
            checkpoint(fold+1, trainlog, config)

    # following training, test on hold-out data
    print("Testing best epoch . . .")
    model.load_state_dict(
        torch.load(
            os.path.join(
                config.res_dir, config.experiment_name, "Fold_{}".format(fold + 1), "model.pth.tar"
            )
        )["state_dict"]
    )
    model.eval()

    test_metrics, test_img_metrics = iterate(
                                    model,
                                    data_loader=test_loader,
                                    criterion=criterion,
                                    config=config,
                                    optimizer=optimizer,
                                    mode="test",
                                    epoch=epoch,
                                    device=device,
                                )
    print(f'Loss {test_metrics["test_loss"]}')
    print(f' test image metrics: {test_img_metrics}')
    #"Loss {:.4f},  Acc {:.2f},  IoU {:.4f}".format(
        #test_metrics["test_loss"],
        #test_metrics["test_accuracy"],
        #test_metrics["test_IoU"],
    #))
    save_results(fold + 1, test_metrics, config)

    #if config.fold is None:
    #overall_performance(config)

    # close tensorboard logging
    writer.close()

if __name__ == "__main__":
    config = parser.parse_args()
    for k, v in vars(config).items():
        if k in list_args and v is not None:
            v = v.replace("[", "")
            v = v.replace("]", "")
            config.__setattr__(k, list(map(int, v.split(","))))

    # validate input flags
    #assert config.num_classes == config.out_conv[-1]
    isPow2 = lambda x: (x & (x-1) == 0) and x != 0
    assert isPow2(config.input_size) and 0<config.input_size<=256

    pprint.pprint(config)
    main(config)
