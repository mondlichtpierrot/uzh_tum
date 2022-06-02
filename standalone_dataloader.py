# Python script to demonstrate utilizing the pyTorch data loader for SEN12MS-CR-TS

import os
import numpy as np
import torch
from data.dataLoader import SEN12MSCRTS

if __name__ == '__main__':

    # main parameters for instantiating SEN12MS-CR-TS
    root        = os.path.expanduser('~/Data')  # path to your copy of SEN12MS-CR-TS
    split       = 'all'                         # ROI to sample from, belonging to splits [all | train | val | test]
    input_t     = 3                             # number of input time points to sample
    import_path = None                          # path to importing the suppl. file specifying what time points to load for input and output
    sample_type = 'cloudy_cloudfree'            # type of samples returned [cloudy_cloudfree | generic]
    sen12mscrts = SEN12MSCRTS(root, split=split, sample_type=sample_type, n_input_samples=input_t, import_data_path=import_path)
    dataloader  = torch.utils.data.DataLoader(sen12mscrts, batch_size=1, shuffle=False, num_workers=10)
    
    # iterate over split and do some data accessing for demonstration
    for pdx, patch in enumerate(dataloader):
        print(f'Fetching {pdx}. batch of data.')

        input_s1  = patch['input']['S1']
        input_s2  = patch['input']['S2']
        input_c   = np.sum(patch['input']['coverage'])/len(patch['input']['coverage'])
        output_s2 = patch['target']['S2']
        dates_s1  = patch['input']['S1 TD']
        dates_s2  = patch['input']['S2 TD']
