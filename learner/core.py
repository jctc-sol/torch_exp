class Learner():
    """
    Base learner class for all other learners to inherit from
    """
    def __init__(self, model, loss_func, metrics, data):
        self.model, self.loss_func, self.metrics, self.data = model, loss_func, metrics, data
