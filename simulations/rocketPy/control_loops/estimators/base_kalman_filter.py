class BaseKalmanFilter:
    def __init__(self):
        self.state_dim = None
        self.measurement_dim = None

    def predict(self, *args, **kwargs):
        raise NotImplementedError("The predict method should be implemented by subclasses.")

    def update(self, *args, **kwargs):
        raise NotImplementedError("The update method should be implemented by subclasses.")

    def get_state_estimate(self):
        raise NotImplementedError("The get_state_estimate method should be implemented by subclasses.")
