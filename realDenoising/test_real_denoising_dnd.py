import numpy as np
import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import utils

from basicsr.models.archs.mairunet_arch import MaIRUNet

from skimage import img_as_ubyte
import h5py
import scipy.io as sio
from pdb import set_trace as stx

parser = argparse.ArgumentParser(description='Real Image Denoising')

parser.add_argument('--input_dir', default='/xlearning/boyun/datasets/RealDN/DND/', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='/xlearning/boyun/codes/MaIR/realDenoising/results/Real_Denoising/test/', type=str, help='Directory for results')
parser.add_argument('--weights', default='/xlearning/boyun/codes/MaIR/realDenoising/experiments/trainMaIR_RealDN/models/MaIR_RealDN.pth', type=str, help='Path to weights')
parser.add_argument('--save_images', action='store_true', help='Save denoised images in result directory')

args = parser.parse_args()

####### Load model options #######

opt_str = r"""
  type: MaIRUNet
  inp_channels: 3
  out_channels: 3
  dim: 48
  num_blocks: [4, 6, 6, 8]
  num_refinement_blocks: 4

  ssm_ratio: 2.0 
  flp_ratio: 4.0
  mlp_ratio: 1.5 
  bias: False
  dual_pixel_task: False

  img_size: 128
  scan_len: 4
  batch_size: 8
  dynamic_ids: False
"""

import yaml
opt = yaml.safe_load(opt_str)
network_type = opt.pop('type')
##########################################

result_dir_mat = os.path.join(args.result_dir, 'mat')
os.makedirs(result_dir_mat, exist_ok=True)

if args.save_images:
    result_dir_png = os.path.join(args.result_dir, 'png')
    os.makedirs(result_dir_png, exist_ok=True)

model_restoration = MaIRUNet(**opt)

device = torch.device('cuda:4')
# torch.cuda.set_device(7)
# checkpoint = torch.load(args.weights, map_location=device)

weights = '/xlearning/boyun/codes/MaIR/realDenoising/experiments/MaIR_RealDN/models/MaIR_RealDN.pth'
checkpoint = torch.load(args.weights, map_location=device)
model_restoration.load_state_dict(checkpoint['params'])
print("===>Testing using weights: ",args.weights)
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

israw = False
eval_version="1.0"

# Load info
infos = h5py.File(os.path.join(args.input_dir, 'info.mat'), 'r')
info = infos['info']
bb = info['boundingboxes']

# Process data
with torch.no_grad():
    for i in tqdm(range(50)):
        Idenoised = np.zeros((20,), dtype=np.object)
        filename = '%04d.mat'%(i+1)
        filepath = os.path.join(args.input_dir, 'images_srgb', filename)
        img = h5py.File(filepath, 'r')
        Inoisy = np.float32(np.array(img['InoisySRGB']).T)

        # bounding box
        ref = bb[0][i]
        boxes = np.array(info[ref]).T

        for k in range(20):
            idx = [int(boxes[k,0]-1),int(boxes[k,2]),int(boxes[k,1]-1),int(boxes[k,3])]
            noisy_patch = torch.from_numpy(Inoisy[idx[0]:idx[1],idx[2]:idx[3],:]).unsqueeze(0).permute(0,3,1,2).cuda()
            restored_patch = model_restoration(noisy_patch)
            restored_patch = torch.clamp(restored_patch,0,1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
            Idenoised[k] = restored_patch

            if args.save_images:
                save_file = os.path.join(result_dir_png, '%04d_%02d.png'%(i+1,k+1))
                denoised_img = img_as_ubyte(restored_patch)
                utils.save_img(save_file, denoised_img)

        # save denoised data
        sio.savemat(os.path.join(result_dir_mat, filename),
                    {"Idenoised": Idenoised,
                     "israw": israw,
                     "eval_version": eval_version},
                    )
