from micromojograd.engine import Value


def main():
    var positive = Value(3.0)
    var active = positive.relu()
    active.backward()
    print(active, "positive.grad=", positive.grad())

    var negative = Value(-3.0)
    var inactive = negative.relu()
    inactive.backward()
    print(inactive, "negative.grad=", negative.grad())

    var x = Value(-2.0)
    var out = (x * 3.0 + 10.0).relu()
    out.backward()
    print(out, "x.grad=", x.grad())
