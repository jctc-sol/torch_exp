from torch.optim.lr_scheduler import OneCycleLR
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
        # check to see if differential LR is applied, store relative lr_ratios if that's the case
        self.find_differential_lr()
    
    
    def after_step(self):
        # check if scheduler has been created or not, create one if not
        # this is a hack since often the exp obj has already been initialized
        # and therefore does not call the 'before_train' callback routine again
        if not hasattr(self, 'scheduler'):
            self.before_train()
        # step
        self.scheduler.step()
        # get the updated LR & momentum 
        lr  = self.exp.opt.param_groups[0]['lr']
        mom = self.exp.opt.param_groups[0]['momentum']
        # update trackers
        self.lr_record(lr)
        self.mom_record(mom)
        # set updated LR while maintaining any differential LR between param groups
        self.set_lr(lr)
        print('lr_scheduler step')