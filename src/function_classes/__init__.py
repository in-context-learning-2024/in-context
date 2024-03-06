class Function:
    def __init__(self, in_distribution):
        pass

    def __iter__(self) -> self:
        pass
    
    def __next__(self):
        pass

class FunctionClass:
    def __init__(self, in_distribution, parameter_distribution):
        pass

    def sample_function(self) -> Function:
        pass 

    def get_function_iter(self) -> iter(Function):
        pass