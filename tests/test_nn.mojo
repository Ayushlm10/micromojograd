from micromojograd.engine import Value
from micromojograd.nn import Neuron, Layer, MLP
from std.random import seed
from std.testing import assert_almost_equal, assert_equal, assert_true, TestSuite


def test_linear_neuron_forward_and_parameters() raises:
    var neuron = Neuron([Value(2.0), Value(-1.0)], Value(0.5), nonlin=False)
    var out = neuron([Value(3.0), Value(4.0)])
    var parameters = neuron.parameters()
    assert_almost_equal(out.data(), 2.5, atol=1e-12)
    assert_equal(len(parameters), 3)
    assert_almost_equal(parameters[0].data(), 2.0, atol=1e-12)
    assert_almost_equal(parameters[1].data(), -1.0, atol=1e-12)
    assert_almost_equal(parameters[2].data(), 0.5, atol=1e-12)


def test_relu_neuron_backward() raises:
    var w0 = Value(2.0)
    var w1 = Value(-1.0)
    var bias = Value(0.5)
    var x0 = Value(3.0)
    var x1 = Value(1.0)
    var neuron = Neuron([w0, w1], bias)
    var out = neuron([x0, x1])
    out.backward()
    assert_almost_equal(out.data(), 5.5, atol=1e-12)
    assert_almost_equal(w0.grad(), 3.0, atol=1e-12)
    assert_almost_equal(w1.grad(), 1.0, atol=1e-12)
    assert_almost_equal(bias.grad(), 1.0, atol=1e-12)
    assert_almost_equal(x0.grad(), 2.0, atol=1e-12)
    assert_almost_equal(x1.grad(), -1.0, atol=1e-12)


def test_layer_forward_and_parameters() raises:
    var first = Neuron([Value(1.0), Value(2.0)], Value(0.0), nonlin=False)
    var second = Neuron([Value(-1.0), Value(1.0)], Value(1.0), nonlin=False)
    var layer = Layer([first^, second^])
    var outputs = layer([Value(3.0), Value(4.0)])
    var parameters = layer.parameters()
    assert_equal(len(outputs), 2)
    assert_almost_equal(outputs[0].data(), 11.0, atol=1e-12)
    assert_almost_equal(outputs[1].data(), 2.0, atol=1e-12)
    assert_equal(len(parameters), 6)


def test_layer_shared_input_backward() raises:
    var x0 = Value(2.0)
    var x1 = Value(3.0)
    var first = Neuron([Value(1.0), Value(0.0)], Value(0.0), nonlin=False)
    var second = Neuron([Value(0.0), Value(2.0)], Value(0.0), nonlin=False)
    var layer = Layer([first^, second^])
    var outputs = layer([x0, x1])
    var out = outputs[0] + outputs[1]
    out.backward()
    assert_almost_equal(out.data(), 8.0, atol=1e-12)
    assert_almost_equal(x0.grad(), 1.0, atol=1e-12)
    assert_almost_equal(x1.grad(), 2.0, atol=1e-12)


def test_mlp_forward_and_parameters() raises:
    var hidden_first = Neuron([Value(1.0), Value(0.0)], Value(0.0))
    var hidden_second = Neuron([Value(0.0), Value(1.0)], Value(0.0))
    var hidden = Layer([hidden_first^, hidden_second^])
    var output_neuron = Neuron([Value(2.0), Value(-1.0)], Value(0.5), nonlin=False)
    var output = Layer([output_neuron^])
    var mlp = MLP([hidden^, output^])
    var outputs = mlp([Value(3.0), Value(4.0)])
    var parameters = mlp.parameters()
    assert_equal(len(outputs), 1)
    assert_almost_equal(outputs[0].data(), 2.5, atol=1e-12)
    assert_equal(len(parameters), 9)


def test_mlp_backward_through_layers() raises:
    var x0 = Value(3.0)
    var x1 = Value(4.0)
    var output_weight_first = Value(2.0)
    var output_weight_second = Value(-1.0)
    var hidden_first = Neuron([Value(1.0), Value(0.0)], Value(0.0))
    var hidden_second = Neuron([Value(0.0), Value(1.0)], Value(0.0))
    var hidden = Layer([hidden_first^, hidden_second^])
    var output_neuron = Neuron(
        [output_weight_first, output_weight_second], Value(0.5), nonlin=False
    )
    var output = Layer([output_neuron^])
    var mlp = MLP([hidden^, output^])
    var outputs = mlp([x0, x1])
    outputs[0].backward()
    assert_almost_equal(x0.grad(), 2.0, atol=1e-12)
    assert_almost_equal(x1.grad(), -1.0, atol=1e-12)
    assert_almost_equal(output_weight_first.grad(), 3.0, atol=1e-12)
    assert_almost_equal(output_weight_second.grad(), 4.0, atol=1e-12)


def test_mlp_parameter_update_and_zero_grad() raises:
    var neuron = Neuron([Value(2.0)], Value(0.0), nonlin=False)
    var layer = Layer([neuron^])
    var mlp = MLP([layer^])
    var output = mlp([Value(3.0)])
    output[0].backward()
    var parameters = mlp.parameters()
    assert_almost_equal(parameters[0].grad(), 3.0, atol=1e-12)
    assert_almost_equal(parameters[1].grad(), 1.0, atol=1e-12)
    for i in range(len(parameters)):
        parameters[i].set_data(parameters[i].data() - 0.1 * parameters[i].grad())
    mlp.zero_grad()
    assert_almost_equal(parameters[0].grad(), 0.0, atol=1e-12)
    assert_almost_equal(parameters[1].grad(), 0.0, atol=1e-12)
    var updated = mlp([Value(3.0)])
    assert_almost_equal(updated[0].data(), 5.0, atol=1e-12)


def test_random_initialization_constructors() raises:
    seed(42)
    var neuron = Neuron(2, nonlin=False)
    var neuron_parameters = neuron.parameters()
    assert_equal(len(neuron_parameters), 3)
    for i in range(len(neuron_parameters)):
        assert_true(neuron_parameters[i].data() >= -1.0)
        assert_true(neuron_parameters[i].data() < 1.0)
    var layer = Layer(2, 3, nonlin=False)
    assert_equal(len(layer([Value(1.0), Value(1.0)])), 3)
    assert_equal(len(layer.parameters()), 9)
    var mlp = MLP(2, [3, 1])
    assert_equal(len(mlp([Value(1.0), Value(1.0)])), 1)
    assert_equal(len(mlp.parameters()), 13)


def test_training_loop_reduces_loss() raises:
    var neuron = Neuron([Value(0.0)], Value(0.0), nonlin=False)
    var layer = Layer([neuron^])
    var mlp = MLP([layer^])
    for _ in range(10):
        mlp.zero_grad()
        var outputs = mlp([Value(2.0)])
        var loss = (outputs[0] - 4.0) ** 2.0
        loss.backward()
        var parameters = mlp.parameters()
        for i in range(len(parameters)):
            parameters[i].set_data(parameters[i].data() - 0.05 * parameters[i].grad())
    var trained = mlp([Value(2.0)])
    assert_almost_equal(trained[0].data(), 4.0, atol=0.01)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
