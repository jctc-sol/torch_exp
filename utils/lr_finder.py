import os
from torch_exp.callbacks import LrFindCallback


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
        self.cb = LrFindCallback(opt, n_batches, lr_start, lr_end, beta)        
        self.exp = exp
        self.opt = opt
        
    
    def run(self):
        # construct a filepath to save current state
        ckpt_path = f"{self.exp.save_dir}/lrfRountine.pth.tar"
        # save current experiment to preserve all param/opt states
        self.exp.save(opt=self.opt, save_path=ckpt_path)
        # attach lrf callback to experiment
        self.exp.add_callback(self.cb)
        # run one epoch routine through training dataset
        self.exp.run(epochs=1, optimizer=self.opt)
        # remove LrFindCallback from exp
        self.exp.rmv_callback(self.cb)
        # load back exp state before running lrf routine
        self.exp.load(ckpt_path)
        # remove ckpt state to clean up
        os.remove(ckpt_path)