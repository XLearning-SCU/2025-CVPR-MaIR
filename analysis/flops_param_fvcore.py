import torch
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)
from model_zoo.mair import buildMaIR_Small, buildMaIR_Tiny, buildMaIR_SR
from model_zoo.mairu import buildMaIRU, buildMaIRU_motiondeblur

from analysis.utils_fvcore import FLOPs
fvcore_flop_count = FLOPs.fvcore_flop_count

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num/1e6, 'Trainable': trainable_num/1e6}

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # SR_x2 | SR_x3 | SR_x4 | lightSR_S_x2 | lightSR_S_x3 | lightSR_S_x4 | lightSR_T_x2 | lightSR_T_x3 | lightSR_T_x4 | md | dh 
    task = 'lightSR_S_x2' 
    
    if task.startswith('SR') or task.startswith('lightSR'):
        H=720
        W=1280
        if task.endswith('x2'):
            scale=2
        elif task.endswith('x3'):
            scale=3
        elif task.endswith('x4'):
            scale=4 
        if task.startswith('SR'):
            init_model = buildMaIR_SR(upscale=scale).to(device)
        elif task.startswith('lightSR_S'):
            init_model = buildMaIR_Small(upscale=scale).to(device)
        elif task.startswith('lightSR_T'):
            init_model = buildMaIR_Tiny(upscale=scale).to(device)
    elif task.startswith('md'): # motion deblur
        H=128
        W=128
        scale=1
        init_model = buildMaIRU_motiondeblur().to(device)
    elif task.startswith('dh'): # image dehazing
        H=256
        W=256
        scale=1
        init_model = buildMaIRU().to(device)

    print(get_parameter_number(init_model))
    with torch.no_grad():
        FLOPs.fvcore_flop_count(init_model, input_shape=(3, H//scale,W//scale))
