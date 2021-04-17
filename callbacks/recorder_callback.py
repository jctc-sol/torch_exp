import time
from callbacks.core import Callback
from utils import *


class RecorderCallback(Callback):
    """
    Instantiates a series of Tracker objects to track key progress and metrics
    of the experiment
    """ 
    def before_train(self):
        self.exp.epoch_time  = Recorder('EpochTime')
        self.exp.batch_time  = Recorder('BatchTime')
        self.exp.eval_time   = Recorder('EvalTime')
        self.exp.loss_record = Recorder('Loss')
        self.exp.metric_records = []
        for i, m in enumerate(listify(self.exp.metrics)):
            self.exp.metric_records.append(Recorder(f'{m}'))
        return True
    
    def before_epoch(self):
        self.epoch_start_time = time.time()
        return True
    
    def before_batch(self):
        self.batch_start_time = time.time()
        return True
        
    def after_batch(self, loss, n_samples):
        batch_time = time.time() - self.batch_start_time
        self.exp.batch_time(batch_time, 1)
        self.exp.loss_record(loss, n_samples)
        return True
    
    def after_epoch(self):
        epoch_time = time.time() - self.epoch_start_time
        self.exp.epoch_time(epoch_time, 1)
        return True
    
    def begin_eval(self):
        self.eval_start_time = time.time()
        return True
        
    def after_eval(self, metrics):
        eval_time = time.time() - self.eval_start_time
        self.exp.eval_time(eval_time, 1)
        for metric, recorder in zip(listify(metrics), self.exp.metric_records):
            recorder(metric, 1)
        return True
    
    def after_train(self):
        pass
