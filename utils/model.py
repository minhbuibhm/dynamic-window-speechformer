"""
Created on Tue Sep 28 16:53:37 CST 2021
@author: lab-chen.weidong
"""

import torch
from torch import distributed as dist

def load_model(model_type, device='cpu', **kwargs):
    if model_type == 'Transformer':
        from model.transformer import build_vanilla_transformer
        model = build_vanilla_transformer(**kwargs)
    elif model_type == 'SpeechFormer':
        from model.speechformer import SpeechFormer
        model = SpeechFormer(**kwargs)
    elif model_type == 'SpeechFormer++' or model_type == 'SpeechFormer_v2':
        from model.speechformer_v2 import SpeechFormer_v2
        model = SpeechFormer_v2(**kwargs)
    elif model_type == 'SpeechDW':
        from model.speech_dw_former import SpeechDW_Former
        model = SpeechDW_Former(**kwargs)
    else:
        raise KeyError(f'Unknown model type: {model_type}')

    if device == 'cuda':
        model = model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[dist.get_rank()], find_unused_parameters=True)

    return model


