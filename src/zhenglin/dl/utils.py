import torch

def summary(model):
    print(model)
    parameter_number = 0
    for layer in list(model.parameters()):
        parameter_number += torch.prod(torch.tensor(layer.size()))
    print('Total Parameter numbers:{:,}'.format(int(parameter_number)))


def fix_model(model):
    for param in model.parameters():
        param.requires_grad = False

    model.eval()
    
    return model