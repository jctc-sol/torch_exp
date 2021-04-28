import math
from torch_exp.utils.core import listify
from torch_exp.utils.recorder import Recorder
from torch_exp.callbacks.core import Callback


class LrFindCallback(Callback):
    
    def __init__(self, opt, n_batches, lr_start, lr_end, beta):
        Callback.__init__(self)
        self.opt = opt
        self.lr_start = lr_start
        self.beta = beta
        self.multiplier = (lr_end/lr_start) ** (1/n_batches)
        # recorders for lrs & losses
        self.lr_record   = Recorder('LrfLRs')
        self.loss_record = Recorder('LrfLoss', beta) # specify beta to compute exp-avg


    def find_differential_lr(self):
        # find the relative ratio of LR for all param groups wrt to the first 
        # param group; this is for differential learning scenarios where some 
        # params groups are updated at higher LR than others
        lr_pg0 = listify(self.opt.param_groups)[0]['lr']
        self.lr_ratios = []
        for pg in listify(self.opt.param_groups):
            self.lr_ratios.append(pg['lr']/lr_pg0)
        

    def set_lr(self, lr):
        # set current lr value; update all optimizer param group
        # learning rates to new value * their respective lr ratio
        self.lr = lr
        for i, pg in enumerate(listify(self.opt.param_groups)):
            pg['lr'] = lr * self.lr_ratios[i]
                    
    
    def before_epoch(self):        
        self.batch_num = 0
        self.best_loss = 0.
        # set stop flag to False to enable running exp for one-epoch
        self.exp.stop = False
        # find LR ratio for all param groups (expressed wrt to the 1st param group)
        self.find_differential_lr()
        # set all LR for all param groups at start value
        self.set_lr(self.lr_start)
        print('here1')
        
        
    def after_loss(self):
        # record loss
        # note: this computes the exp-avg internally & can be access via
        # self.loss_record.exp_avg
        loss = self.exp.loss.item()
        self.loss_record(loss)
        # record log(lr)
        self.lr_record(math.log10(self.lr))
        # compute the smoothed loss
        self.batch_num += 1
        smoothed_loss = self.loss_record.exp_avg / (1 - self.beta**self.batch_num)
        # stop if loss is exploding
        if self.batch_num > 1 and smoothed_loss > 4 * self.best_loss:
            # set stop flag to True to break from one_batch & one_epoch routine
            self.exp.stop = True
        # check if current loss is better
        if smoothed_loss < self.best_loss or self.batch_num==1:
            self.best_loss = smoothed_loss
        print(f'lr: {self.lr}  loss: {smoothed_loss}  best loss: {self.best_loss}')

            
    def after_batch(self):
        # update the lr for the next batch
        self.lr *= self.multiplier
        self.set_lr(self.lr)
        
        