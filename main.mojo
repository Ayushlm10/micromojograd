from micromojograd.engine import Value


def main():
    var a = Value(2.0)
    var b = Value(-3.0)
    var c = a + b
    var d = a * b
    print(a, "grad=", a.grad(), "parents=", a.previous_count())
    print(c, "grad=", c.grad(), "op=", c.operation(), "parents=", c.previous_count())
    print(d, "grad=", d.grad(), "op=", d.operation(), "parents=", d.previous_count())
