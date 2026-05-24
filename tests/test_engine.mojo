from micromojograd.engine import Value
from std.testing import assert_almost_equal, TestSuite


def test_shared_intermediate_backward() raises:
    var a = Value(2.0)
    var b = Value(3.0)
    var product = a * b
    var out = product + product
    out.backward()
    assert_almost_equal(out.data(), 12.0, atol=1e-12)
    assert_almost_equal(product.grad(), 2.0, atol=1e-12)
    assert_almost_equal(a.grad(), 6.0, atol=1e-12)
    assert_almost_equal(b.grad(), 4.0, atol=1e-12)


def test_scalar_operators() raises:
    var x = Value(2.0)
    var out = 1.0 + (3.0 * x) - (x / 2.0) + (8.0 / x) + (10.0 - x)
    out.backward()
    assert_almost_equal(out.data(), 18.0, atol=1e-12)
    assert_almost_equal(x.grad(), -0.5, atol=1e-12)


def test_power() raises:
    var x = Value(3.0)
    var out = x ** 3.0
    out.backward()
    assert_almost_equal(out.data(), 27.0, atol=1e-12)
    assert_almost_equal(x.grad(), 27.0, atol=1e-12)


def test_relu_positive() raises:
    var x = Value(3.0)
    var out = x.relu()
    out.backward()
    assert_almost_equal(out.data(), 3.0, atol=1e-12)
    assert_almost_equal(x.grad(), 1.0, atol=1e-12)


def test_relu_negative() raises:
    var x = Value(-2.0)
    var out = x.relu()
    out.backward()
    assert_almost_equal(out.data(), 0.0, atol=1e-12)
    assert_almost_equal(x.grad(), 0.0, atol=1e-12)


def test_compound_with_relu() raises:
    var a = Value(-4.0)
    var b = Value(2.0)
    var c = Value(2.0)
    var out = ((a * b + a ** 2.0) / c).relu()
    out.backward()
    assert_almost_equal(out.data(), 4.0, atol=1e-12)
    assert_almost_equal(a.grad(), -3.0, atol=1e-12)
    assert_almost_equal(b.grad(), -2.0, atol=1e-12)
    assert_almost_equal(c.grad(), -2.0, atol=1e-12)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
