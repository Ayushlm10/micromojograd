from micromojograd.engine import Value


def main():
    var a = Value(2.0)
    var b = Value(-3.0)
    var c = a * b
    var out = c + a
    out.backward()
    print(out, "a.grad=", a.grad(), "b.grad=", b.grad(), "c.grad=", c.grad())

    var p = Value(2.0)
    var square = p * p
    var doubled_square = square + square
    doubled_square.backward()
    print(doubled_square, "p.grad=", p.grad(), "square.grad=", square.grad())
