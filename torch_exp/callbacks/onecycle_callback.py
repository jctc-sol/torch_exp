from torch.optim.lr_scheduler import OneCycleLR
from torch_exp.utils.core import listify
from torch_exp.utils.recorder import Recorder
from torch_exp.callbacks.core import Callback


class OneCycleCallback(Callback):
    
    def __init__(self, max_lr, scheduler_kwargs):
        Callback.__init__(self)
        self.max_lr = max_lr
        self.scheduler_kwargs = scheduler_kwargs
        self.lr_record   = Recorder('LearningRate')
        self.mom_record = Recorder('OptMomentum')

        
    def before_train(self):
        self.scheduler = OneCycleLR(self.exp.opt, max_lr=self.max_lr, **self.scheduler_kwargs)
    
    
    def after_step(self):
        self.scheduler.step()