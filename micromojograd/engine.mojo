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


def _was_visited(node: ArcPointer[_Node], visited: List[ArcPointer[_Node]]) -> Bool:
    for existing in visited:
        if node is existing:
            return True
    return False


def _build_topology(
    node: ArcPointer[_Node],
    mut topology: List[ArcPointer[_Node]],
    mut visited: List[ArcPointer[_Node]],
):
    if _was_visited(node, visited):
        return
    visited.append(node)
    for parent in node[].previous:
        _build_topology(parent, topology, visited)
    topology.append(node)


def _backward_node(node: ArcPointer[_Node]):
    if node[].operation == "+":
        var left = node[].previous[0]
        var right = node[].previous[1]
        left[].grad += node[].grad
        right[].grad += node[].grad
    elif node[].operation == "*":
        var left = node[].previous[0]
        var right = node[].previous[1]
        left[].grad += right[].data * node[].grad
        right[].grad += left[].data * node[].grad


struct Value(Writable, ImplicitlyCopyable):
    var _node: ArcPointer[_Node]

    def __init__(out self, data: Float64):
        self._node = ArcPointer(_Node(data))

    def data(self) -> Float64:
        return self._node[].data

    def grad(self) -> Float64:
        return self._node[].grad

    def set_grad(self, grad: Float64):
        self._node[].grad = grad

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

    def _backward(self):
        _backward_node(self._node)

    def backward(self):
        var topology = List[ArcPointer[_Node]]()
        var visited = List[ArcPointer[_Node]]()
        _build_topology(self._node, topology, visited)
        self.set_grad(1.0)
        for node in reversed(topology):
            _backward_node(node)
