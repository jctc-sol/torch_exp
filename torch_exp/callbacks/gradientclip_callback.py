from torch_exp.callbacks.core import Callback


class GradientClipCallback(Callback):

    def __init__(self, clip):
        Callback.__init__(self)
        self.clip = clip
        
        
    def after_backward(self):
        for group in self.exp.opt.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    param.grad.data.clamp_(-self.clip, self.clip)
