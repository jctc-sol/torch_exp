import re, uuid
from utils import *


class Callback():
    """
    Base template for all other callback functions to inhereit from.
    """
    _order=0 # used to sort order of callback to execute, 0 being the base layer parent Callback class
    _cb_names = ['before_train',
                 'before_epoch',
                 'before_batch',
                 'after_batch',
                 'after_epoch',
                 'begin_eval',
                 'after_eval',
                 'after_train'
                 ]

    def set_exp(self, exp): 
        self.exp=exp

    def __getattr__(self, k): 
        return getattr(self.exp, k)  # reaches into experiment to get attribute k
    
    @property
    def name(self):
        name = re.sub(r'Callback$', '', self.__class__.__name__)
        return camel2snake(name or 'callback')
    
    def __call__(self, cb_name):  # check if the callback specified by cb_name exists
        assert cb_name in self._cb_names, f'invalid callback name: {cb_name}'
        f = getattr(self, cb_name, None)
        if f and f(): return True
        return False
