import os, uuid
from callbacks.core import Callback

class SetupCallback(Callback):
    """
    Performs setup procedures to setup an experiment (i.e. create local directory to store
    artifacts etc.) as well as updating training progress between batches and epochs.
    """
    def __init__(self, display_every_n_batch=100, eval_every_n_epoch=1, save_every_n_epoch=1,
                 save_dir='runs'):
        Callback.__init__(self)
        self.display_every_n_batch = display_every_n_batch
        self.eval_every_n_epoch = eval_every_n_epoch
        self.save_every_n_epoch = save_every_n_epoch
        self.save_dir = save_dir
        
    def before_train(self):
        self.exp._id = f"exp_{uuid.uuid4()}"
        os.makedirs(f'./{self.save_dir}/{self.exp._id}', exist_ok=True)
        self.exp.n_epochs=0.
        self.exp.n_iter=0
        return True
        
    def before_epoch(self):
        self.exp.n_epochs=self.epoch
        self.model.train()
        self.exp.in_train=True
        return True
    
    def before_batch(self):
        pass
    
    def after_batch(self):
        self.exp.n_iter+=1
        return True
        
    def after_epoch(self):
        self.exp.n_epochs+=1
        return True
    
    def begin_eval(self):
        self.model.eval()
        self.exp.in_train=False
        return True
        
    def after_eval(self):
        pass
    
    def after_train(self):
        pass
