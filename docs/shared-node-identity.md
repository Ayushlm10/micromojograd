---
summary: Explains the Mojo `Value` design: shared graph-node identity and typed scalar operator overloads replacing Python runtime coercion.
read_when:
  - Changing Value, _Node, graph construction, gradient accumulation, or arithmetic operators
  - Comparing the Mojo autodiff model to the Python scratchgrad implementation
  - Adding scalar-left or scalar-right Value expressions
---

# Shared Node Identity

## What

`Value` is the public scalar-autodiff handle, while `_Node` stores the mutable computation-graph state: numeric `data`, accumulated `grad`, parent links, and the operation that produced the node.

Each `Value` owns an `ArcPointer[_Node]`. When an operation records an operand as a parent, it copies the pointer handle rather than copying the node. All references to one mathematical graph node therefore reach the same future gradient accumulator.

## Why

Reverse-mode autodiff accumulates contributions into a node's single gradient slot. If a value is reused, such as `c = a * a` or `d = c + c`, both incoming gradient paths must update the same `a.grad` or `c.grad`.

Python class instances already have reference identity: placing `self` in `_prev` retains a reference to the original object. A plain Mojo `struct Value` has value semantics; storing `Value` operands as parents would copy their state rather than preserve the original node identity. Gradient updates through those copied parents would not update the user's original input handle.

## Key Files

- `micromojograd/engine.mojo`: defines `_Node`, the pointer-backed `Value` handle, graph/backward behavior, and typed arithmetic overloads.
- `main.mojo`: executable example exercising scalar-left and scalar-right autodiff expressions.

## How It Works

`_Node` is the heap-managed graph record:

```mojo
struct _Node(Movable):
    var data: Float64
    var grad: Float64
    var previous: List[ArcPointer[Self]]
    var operation: String
```

`Value` keeps the user-facing API small while holding shared identity:

```mojo
struct Value(Writable, ImplicitlyCopyable):
    var _node: ArcPointer[_Node]

    def __init__(out self, data: Float64):
        self._node = ArcPointer(_Node(data))
```

`ArcPointer` reference-counts a heap allocation. Copying a `Value` or appending `self._node` to a result node's `previous` list creates another handle to the same `_Node`; it does not duplicate `data` or `grad`.

For `var c = a + b`, addition performs the forward calculation, then records graph identity:

```mojo
var out = Self(self.data() + other.data())
out._node[].previous.append(self._node)
out._node[].previous.append(other._node)
out._node[].operation = "+"
```

The resulting relationship is:

```text
a Value ─────────────> Node A <──────────── c.previous[0]
b Value ─────────────> Node B <──────────── c.previous[1]
c Value ─────────────> Node C(data=A+B, operation="+")
```

Thus, when gradient propagation mutates `Node A.grad` through `c.previous[0]`, `a.grad()` reads the same storage.

### Scalar arithmetic: overloads instead of runtime coercion

The Python implementation can accept `Value | float` and normalize operands at runtime:

```python
other = other if isinstance(other, Value) else Value(other)
```

Mojo resolves argument types at compile time. `Value` therefore provides separate overloads for graph operands and `Float64` operands:

```mojo
def __add__(self, other: Self) -> Self:
    ...

def __add__(self, other: Float64) -> Self:
    return self + Self(other)
```

The `Float64` overload turns the scalar into a constant `Value` leaf, then delegates to the `Value`-to-`Value` implementation. This keeps graph creation and derivative rules in one place.

Operators also need reverse overloads when a scalar is on the left, because `Float64` does not know how to combine itself with `Value`:

```mojo
def __radd__(self, other: Float64) -> Self:
    return self + other

def __rtruediv__(self, other: Float64) -> Self:
    return Self(other) / self
```

This enables expressions such as `1.0 + x`, `3.0 * x`, `10.0 - x`, and `8.0 / x`. Scalar constant leaves participate in the graph but are not user parameters whose gradients normally need inspection.

## Gotchas

- Shared node identity is required for reused values and shared intermediates; a copied tree is not the original computation graph.
- `_Node` is `Movable` because `ArcPointer(_Node(data))` moves a constructed node into managed heap storage.
- `Value` is `ImplicitlyCopyable` because copying its `ArcPointer` handle should share one node; `_Node` itself is not implicitly copyable.
- Do not reproduce Python's `isinstance` coercion pattern in Mojo; add explicit typed overloads and reversed methods when operand order matters.
- `backward()` and power/derived arithmetic are implemented; activation functions such as `relu()` are not yet present.
