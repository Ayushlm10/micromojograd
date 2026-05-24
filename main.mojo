from micromojograd.engine import Value


def main():
    var a = Value(2.0)
    var b = Value(-3.0)
    var c = a + b
    c.set_grad(1.0)
    c._backward()
    print(c, "op=", c.operation(), "a.grad=", a.grad(), "b.grad=", b.grad())

    var x = Value(2.0)
    var y = Value(-3.0)
    var z = x * y
    z.set_grad(1.0)
    z._backward()
    print(z, "op=", z.operation(), "x.grad=", x.grad(), "y.grad=", y.grad())

    var p = Value(2.0)
    var q = p * p
    q.set_grad(1.0)
    q._backward()
    print(q, "op=", q.operation(), "p.grad=", p.grad())
