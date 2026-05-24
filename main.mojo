from micromojograd.engine import Value


def main():
    var p = Value(3.0)
    var square = p ** 2.0
    square.backward()
    print(square, "p.grad=", p.grad())

    var x = Value(6.0)
    var y = Value(2.0)
    var out = (-x) + (x / y) - y
    out.backward()
    print(out, "x.grad=", x.grad(), "y.grad=", y.grad())
