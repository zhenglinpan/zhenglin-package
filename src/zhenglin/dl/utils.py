import torch
from networks.cyclegan import ReplayBuffer as ReplayBuffer_

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

class ReplayBuffer(ReplayBuffer_):
    pass

class EasyReplayBuffer:
    def __init__(self, max_size=10):
        """
            An easier implementation of replaybuffer with peek()
            TODO: using list could be slow, use tensor concate instead
        """
        assert max_size > 0
        self.max_size = max_size
        self.data = []
    
    def push_and_pop(self, data):
        if len(self.data) < self.max_size:
            self.data.append(data)
        else:
            self.data.pop(0)
            self.data.append(data)
        return self.data[-1]
    
    def peek(self):
        assert len(self.data) != 0
        return self.data[0]