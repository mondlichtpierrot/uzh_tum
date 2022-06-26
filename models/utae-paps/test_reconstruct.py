"""
Script for semantic inference with pre-trained models
Author: Vivien Sainte Fare Garnot (github/VSainteuf)
License: MIT
"""
import argparse
import json
import os
import sys
import pprint

import numpy as np
import torch
import torch.utils.data as data

from src import utils, model_utils
from train_reconstruct import iterate, save_results, prepare_output, get_loss

sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
from data.dataLoader import SEN12MSCRTS

parser = argparse.ArgumentParser()
# Model parameters
parser.add_argument(
    "--weight_folder",
    type=str,
    default="./results",
    help="Path to the main folder containing the pre-trained weights",
)
parser.add_argument(
    "--res_dir",
    default="./inference_utae",
    type=str,
    help="Path to directory where results are written."
)
parser.add_argument(    
    "--experiment_name",
    default='utae_S1S2_t3_L1SSIM_all', # #'utae_S1S2_t4_covweighting_europe', 'utae_S1S2_L1SSIM_perceptual1video_1000samples', #"utae_L1SSIM_perceptual01video",
    help="Name of the current experiment, store outcomes in a subdirectory of the results folder",
)
parser.add_argument(
    "--num_workers", default=8, type=int, help="Number of data loading workers"
)
parser.add_argument(
    "--fold",
    default=None,
    type=int,
    help="Do only one of the five fold (between 1 and 5)",
)
parser.add_argument(
    "--device",
    default="cuda",
    type=str,
    help="Name of device to use for tensor computations (cuda/cpu)",
)
parser.add_argument(
    "--display_step",
    default=50,
    type=int,
    help="Interval in batches between display of training metrics",
)

# flags specific to SEN12MS-CR-TS
parser.add_argument("--input_t", default=4, type=int, help="number of input time points to sample, unet3d needs at least 4 time points")
parser.add_argument("--sample_type", default="cloudy_cloudfree", type=str, help="type of samples returned [cloudy_cloudfree | generic]")
parser.add_argument("--root1", default='/media/DATA/SEN12MSCRTS', type=str, help="path to your copy of SEN12MS-CR-TS")
parser.add_argument("--root2", default='~/Data/SEN12MSCRTS_val_test', type=str, help="path to your copy of SEN12MS-CR-TS validation & test splits")
parser.add_argument("--region", default="all", type=str, help="region to (sub-)sample ROI from [all | europa]")
parser.add_argument("--input_size", default=256, type=int, help="size of input patches to (sub-)sample")
parser.add_argument("--plot_every", default=-1, type=int, help="Interval (in items) of exporting plots at validation or test time. Set -1 to disable")
parser.add_argument("--loss", default="combined", type=str, help="Image reconstruction loss to utilize [l1|l2|ssim|combined|covweighting|uncertainty].")
parser.add_argument("--perceptual", default=os.path.expanduser("~/Documents/models/vgg16_13C.pth"), type=str, help="Path to VGG16 checkpoint, no perceptual loss if passing None")
parser.add_argument("--layers_perc", default="video", type=str, help="layers to compute perceptual loss over [dip|video|original|experimental]")
parser.add_argument("--debug", dest="debug", action="store_true", help="whether to debug and profile code or not") 

# flags specific to loss weighting
#parser.add_argument("--mean_sort", default='full', type=str, help="cov weighting [full|decay]")
#parser.add_argument("--mean_decay_param", default=1.0, type=float, help="What decay to use with mean decay")

list_args = ["encoder_widths", "decoder_widths", "out_conv"]
parser.set_defaults(cache=False)

test_config = parser.parse_args()
for k, v in vars(test_config).items():
    if k in list_args and v is not None:
        v = v.replace("[", "")
        v = v.replace("]", "")
        test_config.__setattr__(k, list(map(int, v.split(","))))
pprint.pprint(test_config)

with open(os.path.join(test_config.weight_folder, test_config.experiment_name, "conf.json")) as file:
    model_config = json.loads(file.read())

config      = {**model_config, **vars(test_config)}
config      = argparse.Namespace(**config)
config.fold = test_config.fold
pprint.pprint(config)

def main(config):
    np.random.seed(config.rdm_seed)
    torch.manual_seed(config.rdm_seed)
    device = torch.device(config.device)
    prepare_output(config)

    model = model_utils.get_model(config, mode="reconstruct")
    model = model.to(device)

    config.N_params = utils.get_ntrainparams(model)
    print(model)
    print("TOTAL TRAINABLE PARAMETERS :", config.N_params)

    """
    fold_sequence = (
        fold_sequence if config.fold is None else [fold_sequence[config.fold - 1]]
    )
    for fold, (train_folds, val_fold, test_fold) in enumerate(fold_sequence):
        if config.fold is not None:
            fold = config.fold - 1
    """

    fold        = 0 # n-fold cross-val may be to cumbersome for SEN12MSCRTS, hardcoding this var for now to avoid breaking downstream code
    dt_train    = SEN12MSCRTS(os.path.expanduser(config.root1), split='train', region=config.region, sample_type=config.sample_type , n_input_samples=config.input_t, import_data_path=None)
    dt_val      = SEN12MSCRTS(os.path.expanduser(config.root2), split='val', region=config.region, sample_type=config.sample_type , n_input_samples=config.input_t, import_data_path=None) 
    dt_test     = SEN12MSCRTS(os.path.expanduser(config.root2), split='test', region=config.region, sample_type=config.sample_type , n_input_samples=config.input_t, import_data_path=None)
    
    test_loader = data.DataLoader(
        dt_test,
        batch_size=config.batch_size,
        shuffle=False,
        #num_workers=config.num_workers,
        #drop_last=True,
        #collate_fn=collate_fn,
    )

    # Load weights
    sd = torch.load(
        #os.path.join(config.weight_folder, config.experiment_name, "Fold_{}".format(fold+1), "model.pth.tar"),
        os.path.join(config.weight_folder, config.experiment_name, "model.pth.tar"),
        map_location=device,
    )
    model.load_state_dict(sd["state_dict"])

    # Loss
    criterion = get_loss(config)

    # Inference
    print("Testing . . .")
    model.eval()

    test_metrics, test_img_metrics = iterate(
                                model,
                                data_loader=test_loader,
                                criterion=criterion,
                                config=config,
                                optim=None,
                                mode="test",
                                epoch=0,
                                device=device,
                            )
    print(f'Test image metrics: {test_img_metrics}')
    save_results(fold + 1, test_img_metrics, config)
    """
    print(
        "Loss {:.4f},  Acc {:.2f},  IoU {:.4f}".format(
            test_metrics["test_loss"],
            test_metrics["test_accuracy"],
            test_metrics["test_IoU"],
        )
    )
    save_results(fold + 1, test_metrics, conf_mat.cpu().numpy(), config)
    if config.fold is None: overall_performance(config)
    """


if __name__ == "__main__":
    main(config)