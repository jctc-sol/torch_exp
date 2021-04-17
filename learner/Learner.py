class Learner():
    """
    Base learner class for all other learners to inherit from
    """
    def __init__(self, model, opt, loss_func, data):
        self.model, self.opt, self.loss_func, self.data = model, opt, loss_func, data
