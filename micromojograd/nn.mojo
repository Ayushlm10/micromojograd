from .engine import Value
from std.random import random_float64


struct Neuron(Movable):
    var w: List[Value]
    var b: Value
    var nonlin: Bool

    def __init__(out self, var w: List[Value], b: Value, nonlin: Bool = True):
        self.w = w^
        self.b = b
        self.nonlin = nonlin

    def __init__(out self, nin: Int, nonlin: Bool = True):
        self.w = List[Value]()
        for _ in range(nin):
            self.w.append(Value(random_float64(-1.0, 1.0)))
        self.b = Value(random_float64(-1.0, 1.0))
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

    def zero_grad(self):
        for i in range(len(self.w)):
            self.w[i].zero_grad()
        self.b.zero_grad()


struct Layer(Movable):
    var neurons: List[Neuron]

    def __init__(out self, var neurons: List[Neuron]):
        self.neurons = neurons^

    def __init__(out self, nin: Int, nout: Int, nonlin: Bool = True):
        self.neurons = List[Neuron]()
        for _ in range(nout):
            self.neurons.append(Neuron(nin, nonlin=nonlin))

    def __call__(self, inputs: List[Value]) -> List[Value]:
        var outputs = List[Value]()
        for i in range(len(self.neurons)):
            outputs.append(self.neurons[i](inputs))
        return outputs^

    def parameters(self) -> List[Value]:
        var parameters = List[Value]()
        for i in range(len(self.neurons)):
            parameters.extend(self.neurons[i].parameters())
        return parameters^

    def zero_grad(self):
        for i in range(len(self.neurons)):
            self.neurons[i].zero_grad()


struct MLP(Movable):
    var layers: List[Layer]

    def __init__(out self, var layers: List[Layer]):
        self.layers = layers^

    def __init__(out self, nin: Int, nouts: List[Int]):
        self.layers = List[Layer]()
        var input_size = nin
        for i in range(len(nouts)):
            self.layers.append(
                Layer(input_size, nouts[i], nonlin=i != len(nouts) - 1)
            )
            input_size = nouts[i]

    def __call__(self, inputs: List[Value]) -> List[Value]:
        var activations = inputs.copy()
        for i in range(len(self.layers)):
            activations = self.layers[i](activations)
        return activations^

    def parameters(self) -> List[Value]:
        var parameters = List[Value]()
        for i in range(len(self.layers)):
            parameters.extend(self.layers[i].parameters())
        return parameters^

    def zero_grad(self):
        for i in range(len(self.layers)):
            self.layers[i].zero_grad()
