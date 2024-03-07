import torch

from function_classes.function_class import FunctionClass, Function
    
class LinearRegressionClass(FunctionClass):
    def __init__(self, in_distribution, parameter_distribution):
        super().__init__(in_distribution, parameter_distribution)
        self.function = LinearRegression

    def sample_parameters(self) -> dict:
        return {'w': self.parameter_distribution.sample()}

class LinearRegression(Function):
    def __init__(self, in_distribution, parameters):
        super().__init__(in_distribution, parameters)

    def evaluate(self, xs):
        return (xs @ self.parameters['w'])[:, :, 0]