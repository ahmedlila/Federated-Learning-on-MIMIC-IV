import torch

def get_model_weights(model):
    return {k: v.cpu() for k, v in model.state_dict().items()}

def set_model_weights(model, weights):
    model.load_state_dict(weights)
