import torch
import torch.distributions

# question: why make Function an iterable?
class Function:
    def __init__(self, in_distribution, parameters):
        self.in_distribution = in_distribution
        self.parameters = parameters

    def __iter__(self) -> self:
        return self
    
    def __next__(self):
        # TODO: when to raise a StopIteration error?
        xs = self.sample_in()
        ys = self.evaluate(xs)
        return xs, ys
    
    def sample_in(self):
        # TODO: sample with what batch size? how to deal with this? should each __next__ be a single (x,y) pair or a batch?
        return self.in_distribution.sample()

    def evaluate(self, xs):
        raise NotImplementedError(f"Abstract class Function does not implement `.evaluate()`!")

class FunctionClass:
    def __init__(self, in_distribution, parameter_distribution):
        self.function = Function
        self.in_distribution = in_distribution
        self.parameter_distribution = parameter_distribution

    def sample_function(self) -> Function:
        return self.function(self.in_distribution, self.sample_parameters())
    
    def sample_parameters(self) -> dict:
        raise NotImplementedError(f"Abstract class FunctionClass does not implement `.sample_parameters()`!")

    def get_function_iter(self) -> iter(Function):
        # TODO: can this be implemented from sample_function?
        pass