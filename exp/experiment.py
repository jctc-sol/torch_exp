from utils.core import listify
from callbacks  import *


class Exp():
    
    def listify(self, o):
        """
        Helper function to convert input o into list form
        """
        if o is None: return []
        if isinstance(o, list): return o
        if isinstance(o, str): return [o]
        if isinstance(o, Iterable): return list(o)
        return [o]
    
    
    def __init__(self, cbs=None, cb_funcs=None):
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

        
    def __call__(self, cb_name, *args, **kwargs):
        # default to false
        res = False
        # sort the order of callbacks to execute 
        for cb in sorted(self.cbs, key=lambda x: x._order): 
            res = cb(cb_name, *args, **kwargs) or res
        return res

    
    # pass-along property methods that makes it easier to access learner properties from runner object
    @property
    def opt(self):       return self.learner.opt
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
            self.xb, self.yb = xb, yb
            self('begin_batch')  # ex. call the "begin_batch" callback
            self.pred = self.model(self.xb)
            self('after_pred')
            self.loss = self.loss_func(self.pred, self.yb)
            self('after_loss')
            if not self.in_train: return # stop here if in eval mode as you don't need to backprop & update
            self.loss.backward()
            self('after_backward')
            self.opt.step()
            self('after_step')
            self.opt.zero_grad()
        # call after_cancel_batch callback if CancelBatchException occurs
        except CancelBatchException: self('after_cancel_batch')  
        finally: self('after_batch')

            
    def one_epoch(self, dl):
        self.iters = len(dl)
        try:
            for xb, yb in dl: self.one_batch(xb, yb)
        # call after_cancel_epoch callback if CancelEpochException occurs
        except CancelEpochException: self('after_cancel_epoch')

            
    def run(self, epochs, learn):
        # initialization
        self.epochs, self.learn, self.loss = epochs, learn, tensor(0.)
        try:
            self('before_train')
            for epoch in range(epochs):
                self.epoch = epoch
                if not self('before_epoch'): self.one_epoch(self.data.train_dl)
                with torch.no_grad(): 
                    if not self('before_eval'): self.one_epoch(self.data.valid_dl)
                self('after_epoch')
        # call after_cancel_epoch callback if CancelEpochException occurs
        except CancelTrainException: self('after_cancel_train')
        # cleanup
        finally:
            self('after_train')
            self.learn = None
