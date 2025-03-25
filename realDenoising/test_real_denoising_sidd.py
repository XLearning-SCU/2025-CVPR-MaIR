import numpy as np
import os
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
from skimage import img_as_ubyte
import utils

from basicsr.models.archs.mairunet_arch import MaIRUNet
import scipy.io as sio

parser = argparse.ArgumentParser(description='Real Image Denoising')

parser.add_argument('--input_dir', default='/xlearning/boyun/datasets/RealDN/val/', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='/xlearning/boyun/codes/MaIR/realDenoising/results/Real_Denoising/SIDD/', type=str, help='Directory for results')
parser.add_argument('--weights', default='/xlearning/boyun/codes/MaIR/realDenoising/experiments/MaIR_RealDN/models/MaIR_RealDN.pth', type=str, help='Path to weights')
parser.add_argument('--save_images', action='store_true', help='Save denoised images in result directory')

args = parser.parse_args()

####### Load yaml #######
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
x = yaml.safe_load(opt_str)

s = x.pop('type')
##########################

result_dir_mat = os.path.join(args.result_dir, 'mat')
os.makedirs(result_dir_mat, exist_ok=True)

if args.save_images:
  result_dir_png = os.path.join(args.result_dir, 'png')
  os.makedirs(result_dir_png, exist_ok=True)

model_restoration = MaIRUNet(**x)

device = torch.device('cuda:7')
# torch.cuda.set_device(7)


checkpoint = torch.load(args.weights, map_location=device)
model_restoration.load_state_dict(checkpoint['params'])
print("===>Testing using weights: ",args.weights)
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

# Process data
filepath = os.path.join(args.input_dir, 'ValidationNoisyBlocksSrgb.mat')
img = sio.loadmat(filepath)
Inoisy = np.float32(np.array(img['ValidationNoisyBlocksSrgb']))
Inoisy /=255.
restored = np.zeros_like(Inoisy)
with torch.no_grad():
    for i in tqdm(range(40)):
        for k in range(32):
            noisy_patch = torch.from_numpy(Inoisy[i,k,:,:,:]).unsqueeze(0).permute(0,3,1,2).cuda()
            restored_patch = model_restoration(noisy_patch)
            restored_patch = torch.clamp(restored_patch,0,1).cpu().detach().permute(0, 2, 3, 1).squeeze(0)
            restored[i,k,:,:,:] = restored_patch

            if args.save_images:
                save_file = os.path.join(result_dir_png, '%04d_%02d.png'%(i+1,k+1))
                utils.save_img(save_file, img_as_ubyte(restored_patch))

# save denoised data
sio.savemat(os.path.join(result_dir_mat, 'Idenoised.mat'), {"Idenoised": restored,})
