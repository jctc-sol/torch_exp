from torch_exp.utils.core import camel2snake


class Recorder(object):
    """
    Keeps running record on max/min/avg/sum and cnt
    """
    fields = ['sum', 'cnt', 'min', 'max', 'avg']
    def __init__(self, name=''):
        self.reset()
        self.name = camel2snake(name + 'Recorder')
        
    def reset(self):
        for f in self.fields: setattr(self, f, 0)
        
    def update(self, val, n=1):
        # check for invalid input value
        if val == float("inf") or val == -1*float("inf"): return
        else:
            self.val = val
            self.sum += val*n
            self.cnt += n
            self.avg = self.sum/self.cnt
            if (val < self.min): self.min = val
            if (val > self.max): self.max = val
                
    def __call__(self, val, n=1):
        self.update(val, n)
            
    def __repr__(self):
        return self.name


