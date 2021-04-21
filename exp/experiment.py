import torch
from utils.core import listify
from callbacks  import *


class Exp():
    
    def __init__(self, learner, cbs=None, cb_funcs=None, device=None, name=None, desc=None):
        self.learner = learner
        default_cbs = [SetupCallback(), RecorderCallback()]
        cbs = default_cbs + listify(cbs)
        for cbf in listify(cb_funcs):
            # instantiate each callback function
            cb = cbf()
            cbs.append(cb)
        for cb in cbs:
            # add callback name to current Exp name-space
            setattr(self, cb.name, cb)
            # link each callback to this experiment
            cb.set_exp(self)
        self.stop, self.cbs = False, cbs
        # setup device
        if device:
            if type(device)==torch.device: 
                pass
            elif type(device)==str:
                self.device=torch.device(device)
        else:
            # defaults to cuda:0 if cuda is available
            if torch.cuda.is_available(): self.device = torch.device('cuda:0') 
            else: self.device = torch.device('cpu')
        # move model & cost function to device
        self.model.to(self.device)
        self.loss_func.to(self.device)
        # meta information
        self.name = name
        self.desc = desc
        self('before_train')
        
        
    def __call__(self, cb_name):
        # default to false
        res = False
        # sort the order of callbacks to execute 
        for cb in sorted(self.cbs, key=lambda x: x._order): 
            res = cb(cb_name) or res
        return res

    
    def __repr__(self):
        if hasattr(self,'_id'): return f'exp_{self._id}\n{self.name}; {self.desc}'
        else: return f'exp\n{self.name}; {self.desc}'
    
    
    # pass-along property methods that makes it easier to access learner properties from runner object
    @property
    def model(self):     return self.learner.model
    @property
    def loss_func(self): return self.learner.loss_func
    @property
    def data(self):      return self.learner.data    
    @property
    def metrics(self):   return self.learner.metrics

    
    # single-batch rountine, called in the all_batches method
    def one_batch(self, xb, yb):
        try:
            self.pred = self.model(self.xb)
            self('after_pred')
            # check whether to use a custom loss computation pattern
            if not (hasattr(self,'custom_loss') and self.custom_loss):
                self.loss = self.loss_func(self.pred, self.yb) # default loss compute pattern
            self('after_loss')
            if not self.in_train: return # stop here if in eval mode as you don't need to backprop & update
            self.loss.backward()
            self('after_backward')
            self.opt.step()
            self('after_step')
            self.opt.zero_grad()
        # call after_cancel_batch callback if CancelBatchException occurs
        except CancelBatchException: 
            self('after_cancel_batch')  

            
    def one_epoch(self, dl):
        self.iters = len(dl)
        try:
            for batch in dl:
                self.batch = batch
                self('before_batch')
                self.one_batch(self.xb, self.yb)
                self('after_batch')
                self.show_progress()                                
        # call after_cancel_epoch callback if CancelEpochException occurs
        except CancelEpochException:
            self('after_cancel_epoch')
     
        
    def run(self, epochs, optimizer):
        # initialization
        self.epochs, self.opt, self.loss = epochs, optimizer, torch.tensor(0.)
        try:
            if not (hasattr(self, 'initialized') and self.initialized):
                self('before_train')
            for epoch in range(epochs):
                # training routine
                self('before_epoch')
                self.one_epoch(self.data.train_dl)
                self('after_epoch')
                self.evaluate(self.data.train_dl)
                self.evaluate(self.data.valid_dl)
        # call after_cancel_epoch callback if CancelTrainException occurs
        except CancelTrainException: 
            self('after_cancel_train')
        # cleanup
        finally:
            self('after_train')
            self.show_progress(True)

            
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


    def save(self):
        state = {'id': self._id,
                 'trained_epochs': self.n_epochs,
                 'trained_iterations': self.n_iter,
                 'model': self.model,
                 'optimizer': self.opt}
        filename = f'{self.save_dir}/{self.n_epochs}epochs.pth.tar'
        torch.save(state, filename)


    def load(self, path):
        chkpt = torch.load(path)
        self._id = chkpt['id']
        self.n_epochs = chkpt['trained_epochs']
        self.n_iter = chkpt['trained_iterations']
        self.model = chkpt['model']
        self.opt = chkpt['optimizer']
        print(f"checkpoint loaded; resume training from epoch {self.epoch}")
