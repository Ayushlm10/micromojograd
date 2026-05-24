from micromojograd.engine import Value
from micromojograd.nn import Neuron
from std.testing import assert_almost_equal, assert_equal, TestSuite


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


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
