from callbacks.core import Callback


class CocoDataBatchCallback(Callback):
    
    def __init__(self):
        Callback.__init__(self)
        
    def before_batch(self):        
        self.exp.xb = self.exp.batch['images'].to(self.exp.device)
        self.exp.yb = (self.exp.batch['boxes'], self.exp.batch['cats'])
        
    def after_pred(self):
        pred_boxes  = self.exp.pred[0]
        pred_scores = self.exp.pred[1]
        true_boxes  = [b.to(self.exp.device) for b in self.exp.yb[0]]
        true_cats   = [c.to(self.exp.device) for c in self.exp.yb[1]]
        # compute loss here since the format is different from typical pattern
        self.exp.loss = self.loss_func(pred_boxes, pred_scores, true_boxes, true_cats)
        self.exp.custom_loss = True
