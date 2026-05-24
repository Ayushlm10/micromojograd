from micromojograd.engine import Value


def main():
    var x = Value(2.0)
    var out = 1.0 + (3.0 * x) - (x / 2.0) + (8.0 / x) + (10.0 - x)
    out.backward()
    print(out, "x.grad=", x.grad())

    var y = Value(3.0)
    var scaled_square = (y * 2.0) ** 2.0
    scaled_square.backward()
    print(scaled_square, "y.grad=", y.grad())
