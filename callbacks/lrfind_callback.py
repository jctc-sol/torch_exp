import math
import numpy as np
import matplotlib.pyplot as plt
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
        self.smooth_loss_record = Recorder('LrfSmoothLoss')

    
    def before_epoch(self):        
        self.batch_num = 0
        self.best_loss = 0.
        # set stop flag to False to enable running exp for one-epoch
        self.exp.stop = False
        # find LR ratio for all param groups (expressed wrt to the 1st param group)
        self.find_differential_lr()
        # set all LR for all param groups at start value
        self.set_lr(self.lr_start)
        
        
    def after_loss(self):
        # record log(lr)
        self.lr_record(math.log10(self.lr))
        # record loss
        loss = self.exp.loss.item()
        self.loss_record(loss)  # note: this computes the exp-avg internally
        # compute the smoothed loss
        self.batch_num += 1
        smoothed_loss = self.loss_record.exp_avg / (1 - self.beta**self.batch_num)
        # record smoothed loss
        self.smooth_loss_record(smoothed_loss)
        # stop if loss is exploding
        if self.batch_num > 1 and smoothed_loss > 4 * self.best_loss:
            # set stop flag to True to break from one_batch & one_epoch routine
            self.exp.stop = True
        # stop if loss is nan or inf
        if smoothed_loss == np.nan or smoothed_loss == float("inf"):
            self.exp.stop = True
        # check if current loss is better
        if smoothed_loss < self.best_loss or self.batch_num==1:
            self.best_loss = smoothed_loss

            
    def after_batch(self):
        # update the lr for the next batch
        self.lr *= self.multiplier
        self.set_lr(self.lr)
    
    
    def after_train(self):
        # reset exp state
        self.exp.n_epochs = 0
        self.exp.n_iter = 0
        self.exp.stop = False
        # plot out the log10(lr) vs smoothed loss
        lr = np.array(self.lr_record.value)
        losses = np.array(self.smooth_loss_record.value)
        # plot lr vs smooth losses
        plt.figure()
        plt.plot(lr[10:-2], losses[10:-2])
        plt.axhline(y=self.best_loss, color='green', linestyle='--')
        plt.xlabel('log_10(LR)')
        plt.ylabel('loss')
        plt.show()