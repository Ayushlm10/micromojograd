from .engine import Value


struct Neuron(Movable):
    var w: List[Value]
    var b: Value
    var nonlin: Bool

    def __init__(out self, var w: List[Value], b: Value, nonlin: Bool = True):
        self.w = w^
        self.b = b
        self.nonlin = nonlin

    def __call__(self, inputs: List[Value]) -> Value:
        var activation = self.b
        for i in range(len(self.w)):
            activation = activation + self.w[i] * inputs[i]
        return activation.relu() if self.nonlin else activation

    def parameters(self) -> List[Value]:
        var parameters = self.w.copy()
        parameters.append(self.b)
        return parameters^
