import torch
from callbacks.core import Callback


class DeviceSetupCallback(Callback):
    """
    Set the device to be use to run training, as well as responsible for all the `.to(device)` operations
    used to load model, cost function, metrics, data batches etc. to device.
    """
    _order=0  # this should run first before any other callbacks

    def before_train(self):
        if self.exp.device:
            if type(self.exp.device)==torch.device: 
                pass
            elif type(self.exp.device)==str:
                self.exp.device=torch.device(self.exp.device)
        # set device if not specified
        else:
            # defaults to cuda:0 if cuda is available
            if torch.cuda.is_available(): self.exp.device = torch.device('cuda:0') 
            else: self.exp.device = torch.device('cpu')
        
        # move model & cost function to device
        device = self.exp.device
        self.exp.model.to(device)
        self.exp.loss_func.to(device)
        
    
    def 
        
