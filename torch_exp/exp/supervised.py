import torch
from torch_exp.utils.core import listify
from torch_exp.exp.core import Exp
from torch_exp.callbacks import *


class SupervisedExp(Exp):
    
    def __init__(self, learner, cbs=None, cb_funcs=None, device=None, name=None, desc=None,
                 display_every_n_batch=25, eval_every_n_epoch=1, save_every_n_epoch=1,
                 save_dir='runs'):        
        # default callbacks for supervise experiments
        default_cbs = [SetupCallback(display_every_n_batch, eval_every_n_epoch, save_every_n_epoch, save_dir), 
                       RecorderCallback()]
        # init super
        Exp.__init__(self, learner, default_cbs + listify(cbs), cb_funcs, device, name, desc, 
                     display_every_n_batch, eval_every_n_epoch, save_every_n_epoch, save_dir)

    
    # single-batch rountine, called in the all_batches method
    def one_batch(self, xb, yb):
        try:
            self.opt.zero_grad()
            self.pred = self.model(self.xb)
            self('after_pred')
            # check whether to use a custom loss computation pattern
            if not (hasattr(self,'custom_loss') and self.custom_loss):
                self.loss = self.loss_func(self.pred, self.yb) # default loss compute pattern
            self('after_loss')
            if not self.in_train: return # stop here if in eval mode as you don't need to backprop & update
            self.loss.backward() # back propagation
            self('after_backward')
            self.opt.step() # update params
            print('opt step')
            self('after_step')
        # call after_cancel_batch callback if CancelBatchException occurs
        except CancelBatchException: 
            self('after_cancel_batch')  

            
    def one_epoch(self, dl):
        self.iters = len(dl)
        try:
            for batch in dl:
                self.batch = batch
                self('before_batch')
                # self.stop flag can be used for early stopping of training
                if not self.stop: self.one_batch(self.xb, self.yb)
                else: break
                self('after_batch')
                self.show_progress()
        # call after_cancel_epoch callback if CancelEpochException occurs
        except CancelEpochException:
            self('after_cancel_epoch')
     
        
    def run(self, epochs, optimizer, _eval=True):
        # initialization
        self.epochs, self.opt, self.loss = epochs, optimizer, torch.tensor(0.)
        try:
            if not (hasattr(self, 'initialized') and self.initialized):
                self('before_train')
            for epoch in range(epochs):
                # training routine
                self('before_epoch')
                if not self.stop: self.one_epoch(self.data.train_dl)
                else: break
                self('after_epoch')
        # call after_cancel_epoch callback if CancelTrainException occurs
        except CancelTrainException: 
            self('after_cancel_train')
        # cleanup
        finally:
            self('after_train')
            self.show_progress(True)

            
    def resume(self, epochs):
        self.run(epochs, self.opt)
        
    
    def show_progress(self, show=False):
        # display progress after every n batches
        if show or (self.n_iter % self.display_every_n_batch == 0):
            if self.in_train:
                print(f"[{self.n_epochs}/{self.epochs}] epochs | \
                [{self.n_iter}/{self.iters}] iterations | \
                avg. train loss: {self.train_loss.avg}")
            else:
                print(f"[{self.n_epochs}/{self.epochs}] epochs | \
                [{self.n_iter}/{self.iters}] iterations | \
                avg. eval loss: {self.eval_loss.avg}")                

            
    def evaluate(self, dl, train=False):
        # run evaluation routine after every n epochs
        if (self.n_epochs % self.eval_every_n_epoch == 0):
            with torch.no_grad():
                if train: self('before_train_eval')
                else: self('before_eval')
                self.one_epoch(dl)
                if train: self('after_train_eval')
                else: self('after_eval')


    def save(self, opt=None):
        state = {'id': self._id,
                 'trained_epochs': self.n_epochs,
                 'trained_iterations': self.n_iter,
                 'model': self.model}
        if hasattr(self,'opt'):
            state['optimizer'] = self.opt
        if opt:
            state['optimizer'] = opt
        filename = f'{self.save_dir}/{int(self.n_epochs)}epochs.pth.tar'
        torch.save(state, filename)


    def load(self, path):
        chkpt = torch.load(path)
        self._id = chkpt['id']
        self.n_epochs = chkpt['trained_epochs']
        self.n_iter = chkpt['trained_iterations']
        self.learner.model = chkpt['model']
        if 'optimizer' in chkpt:
            self.opt = chkpt['optimizer']
        print(f"checkpoint loaded; resume training from epoch {self.n_epochs}")