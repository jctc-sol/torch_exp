from torch.optim.lr_scheduler import OneCycleLR
from torch_exp.utils.core import listify
from torch_exp.callbacks.core import Callback


class OneCycleSchedulerCallback(Callback):
    
    def __init__(self, max_lr, scheduler_kwargs):
        Callback.__init__(self)
        self.max_lr
        self.scheduler_kwargs = scheduler_kwargs
        

    def before_train(self):
        self.scheduler = OneCycleLR(self.exp.opt, max_lr=self.max_lr, **self.scheduler_kwargs)
    
    
    def after_step(self):
        self.scheduler.step()