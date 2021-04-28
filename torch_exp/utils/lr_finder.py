from torch_exp.callbacks import LrFindCallback


# +
class LrFinder():
    """
    learning rate finder routine based on FastAI's lr_find & Sylvain Gugger's 
    blog: https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html.
    This is a class wrapper around the LrFindCallback which specifies the rountine
    task to perform during each step of the batch-training process.
    """
    def __init__(self, exp, opt, lr_start=1e-8, lr_end=10., beta=0.98):
        # init values
        n_batches = len(exp.data.train_dl)
        # create lr find callback & add to experiment
        cb = LrFindCallback(opt, n_batches, lr_start, lr_end, beta)
        exp.add_callback(cb)
        self.exp = exp
        self.opt = opt
        
    
    def run(self):
        # save current experiment to preserve all param/opt states
        self.exp.save(opt=self.opt)
        # run one epoch through training dataset
        self.exp.run(epochs=1, optimizer=self.opt, _eval=False)

#         # show result
#         self.plot_lr()
        
    
#     def plot_lr(self):
        
        
