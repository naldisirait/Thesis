import torch
from .ModelSimVP import SimVP_Model

def create_model_prediction(config_class, in_shape, device):
    #create model
    model = SimVP_Model(in_shape, hid_S=64, hid_T=256, N_S=4, N_T=8, model_type='gSTA',
                     mlp_ratio=8., drop=0.0, drop_path=0.0, spatio_kernel_enc=3,
                     spatio_kernel_dec=3, act_inplace=True)
    model.to(device=device)
    return model