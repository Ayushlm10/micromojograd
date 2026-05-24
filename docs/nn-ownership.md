---
summary: Explains why `Layer` owns move-only `Neuron` values and accesses them by index instead of copying through list iteration.
read_when:
  - Changing Neuron, Layer, MLP, or parameter collection behavior
  - Debugging Mojo ownership or List iteration errors in neural network code
---

# Neural Network Ownership and Indexed Neurons

## What

`Neuron` and `Layer` are move-only model structures. A `Layer` owns a `List[Neuron]`, evaluates each stored neuron against the same input vector, and concatenates each neuron's shared `Value` parameter handles.

## Why

A `Neuron` owns its weight-list container (`w: List[Value]`). The implementation intentionally permits transferring a neuron into a layer but does not define what copying an entire neuron means. An implicit copy could be mistaken for an independent cloned model even though its `Value` elements are shared graph-node handles.

Mojo's `List` iteration currently requires its element type to be `Copyable`. Since `Neuron` conforms to `Movable`, not `Copyable`, iterating as `for neuron in self.neurons` fails compilation. Layer execution only needs temporary access to existing stored neurons, not copies of them.

## Key Files

- `micromojograd/nn.mojo`: defines move-only `Neuron` and `Layer`, including indexed layer traversal.
- `tests/test_nn.mojo`: verifies Layer output shape, flattened parameters, and gradient flow through shared inputs.

## How It Works

Neuron construction moves the supplied weight-list storage into the neuron:

```mojo
struct Neuron(Movable):
    var w: List[Value]

    def __init__(out self, var w: List[Value], b: Value, nonlin: Bool = True):
        self.w = w^
```

Likewise, a layer takes ownership of its neuron list:

```mojo
struct Layer(Movable):
    var neurons: List[Neuron]

    def __init__(out self, var neurons: List[Neuron]):
        self.neurons = neurons^
```

Forward evaluation indexes into `self.neurons`:

```mojo
for i in range(len(self.neurons)):
    outputs.append(self.neurons[i](inputs))
```

Index access allows the layer to borrow the neuron stored in its list for the duration of the call. The neuron stays owned by the layer before and after evaluation. This avoids requiring `Neuron(Copyable)` merely to run a forward pass.

Parameter collection uses the same pattern:

```mojo
for i in range(len(self.neurons)):
    parameters.extend(self.neurons[i].parameters())
```

Each `Neuron.parameters()` returns a new list container of `Value` handles. The handles point at the same underlying weight and bias `_Node`s, so eventual optimization through a flattened parameter list will affect the model's real parameters.

## Gotchas

- Do not change `for i in range(len(self.neurons))` to `for neuron in self.neurons` unless `Neuron` copy semantics are deliberately designed and tested.
- `first^` in `Layer([first^, second^])` transfers neuron ownership into the layer; `first` and `second` must not be used afterward.
- A copied `Value` is an intentional shared parameter handle; copying an entire `Neuron` is a separate model-design decision.
