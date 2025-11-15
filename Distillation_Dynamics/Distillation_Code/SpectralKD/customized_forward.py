from types import MethodType

import torch

def register_forward(model, model_name):
    model.forward = MethodType(general_forward, model)

def general_forward(self, x, indices=None, require_feat: bool = True):
    if require_feat:
        x, block_outs = self.forward_intermediates(x, indices)
        x = self.forward_head(x)
        return x, block_outs
    else:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x



