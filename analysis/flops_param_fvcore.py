import torch
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)
from model_zoo.mambaIR import buildMambaIR_Small, buildMambaIR
from model_zoo.mair import buildMaIR_Small, buildMaIR, buildMaIR_Tiny

from analysis.utils_fvcore import FLOPs
fvcore_flop_count = FLOPs.fvcore_flop_count

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    H=720
    W=1280
    scale=2
    # init_model = buildMambaIR_Small(upscale=scale).to(device)
    init_model = buildMaIR_Small(upscale=scale).to(device)
    # init_model = buildMaIR_Tiny(upscale=scale).to(device)
    # init_model = buildMaIR(upscale=scale).to(device)
    # init_model = buildMambaIR(upscale=scale).to(device)
    with torch.no_grad():
        FLOPs.fvcore_flop_count(init_model, input_shape=(3, H//scale,W//scale))
