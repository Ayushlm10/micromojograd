from std.memory import ArcPointer


struct _Node(Movable):
    var data: Float64
    var grad: Float64
    var previous: List[ArcPointer[Self]]
    var operation: String

    def __init__(out self, data: Float64):
        self.data = data
        self.grad = 0.0
        self.previous = []
        self.operation = ""


struct Value(Writable, ImplicitlyCopyable):
    var _node: ArcPointer[_Node]

    def __init__(out self, data: Float64):
        self._node = ArcPointer(_Node(data))

    def data(self) -> Float64:
        return self._node[].data

    def grad(self) -> Float64:
        return self._node[].grad

    def operation(self) -> String:
        return self._node[].operation

    def previous_count(self) -> Int:
        return len(self._node[].previous)

    def write_to(self, mut writer: Some[Writer]):
        writer.write("Value(data=", self.data(), ")")

    def __add__(self, other: Self) -> Self:
        var out = Self(self.data() + other.data())
        out._node[].previous.append(self._node)
        out._node[].previous.append(other._node)
        out._node[].operation = "+"
        return out

    def __mul__(self, other: Self) -> Self:
        var out = Self(self.data() * other.data())
        out._node[].previous.append(self._node)
        out._node[].previous.append(other._node)
        out._node[].operation = "*"
        return out
