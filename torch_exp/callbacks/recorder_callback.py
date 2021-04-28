import time
from torch_exp.callbacks.core import Callback
from torch_exp.utils import *


class RecorderCallback(Callback):
    """
    Instantiates a series of Tracker objects to track key progress and metrics
    of the experiment
    """ 
    def init_or_reset(self, name):
        attr_name = camel2snake(name)
        if hasattr(self.exp, attr_name): getattr(self.exp, attr_name).reset()
        else: setattr(self.exp, attr_name, Recorder(name))

    def before_train(self):
        self.init_or_reset('EpochTime')
        self.init_or_reset('BatchTime')
        self.init_or_reset('TrainEvalTime')
        self.init_or_reset('EvalTime')
        self.init_or_reset('TrainLoss')
        self.init_or_reset('EvalLoss')
        self.exp.train_metrics, self.exp.eval_metrics = [], []
        for i, m in enumerate(listify(self.exp.metrics)):
            self.exp.train_metrics.append(
                Recorder(m.name if hasattr(m,'name') else f'metric_{i}')
            )
            self.exp.eval_metrics.append(
                Recorder(m.name if hasattr(m,'name') else f'metric_{i}')
            )
    
    def before_epoch(self):
        self.epoch_start_time = time.time()
    
    def before_batch(self):
        self.batch_start_time = time.time()
        
    def after_batch(self):
        # update batch time tracker
        batch_time = time.time() - self.batch_start_time
        self.exp.batch_time(batch_time, 1)

    def after_loss(self):
        # update loss tracker
        if self.exp.in_train: 
            self.exp.train_loss(self.exp.loss.item()/1e-3, self.exp.xb.size()[0])
        else:
            self.exp.eval_loss(self.exp.loss.item()/1e-3, self.exp.xb.size()[0])

    def after_epoch(self):
        # update epoch time tracker
        epoch_time = time.time() - self.epoch_start_time
        self.exp.epoch_time(epoch_time, 1)
    
    def before_train_eval(self):
        self.eval_start_time = time.time()
        
    def before_eval(self):
        self.before_train_eval()
        
    def after_train_eval(self):
        # update eval time tracker
        eval_time = time.time() - self.eval_start_time
        self.exp.train_eval_time(eval_time, 1)
        # update metric trackers
#         for m, r in zip(listify(self.exp.metrics), self.exp.train_metrics):
#             recorder(m, 1)
