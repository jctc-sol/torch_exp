import re, uuid
from torch_exp.utils import camel2snake


class Callback():
    """
    Base template for all other callback functions to inhereit from.
    """
    # _order filed will be used to sort order of callback to execute;
    # 0 being the base layer parent Callback class
    _order=0

    def set_exp(self, exp): 
        self.exp=exp
        
    def rmv_exp(self):
        if hasattr(self, 'exp'): delattr(self, 'exp')

    def __getattr__(self, k): 
        return getattr(self.exp, k)  # reaches into experiment to get attribute k
    
    @property
    def name(self):
        name = re.sub(r'Callback$', '', self.__class__.__name__)
        return camel2snake(name or 'callback')
    
    def __call__(self, cb_name):  # check if the callback specified by cb_name exists        
        f = getattr(self, cb_name, None)
        # below condition set to not f() to avoid writing "return True" in every callback function
        if f and not f(): return True
        return False


