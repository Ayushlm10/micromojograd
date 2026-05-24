# micromojograd

A small scalar reverse-mode autodiff engine and multilayer perceptron built from scratch in Mojo, following the learning progression of Karpathy's micrograd.

## Features

- Pointer-backed `Value` nodes with shared graph identity via `ArcPointer`
- Arithmetic operators, scalar overloads, power, and ReLU
- Reverse-mode `backward()` with topological traversal
- `Neuron`, `Layer`, and `MLP` model types
- Parameter updates, gradient reset, and random initialization
- Mojo `TestSuite` coverage for gradients and a tiny training loop

## Setup

This project uses Pixi and Mojo nightly.

```sh
pixi install
pixi run test
```

## Structure

- `micromojograd/engine.mojo`: scalar `Value` graph and autodiff rules
- `micromojograd/nn.mojo`: `Neuron`, `Layer`, and `MLP`
- `tests/test_engine.mojo`: scalar engine and gradient tests
- `tests/test_nn.mojo`: model composition and training tests
- `docs/`: notes on Mojo ownership, graph identity, and testing

## Training Example

Random constructors mirror micrograd's model-building ergonomics, while `parameters()`, `set_data()`, and `zero_grad()` enable gradient descent:

```mojo
from micromojograd.engine import Value
from micromojograd.nn import MLP
from std.random import seed


def main():
    seed(42)
    var model = MLP(1, [1])

    for _ in range(10):
        model.zero_grad()
        var prediction = model([Value(2.0)])
        var loss = (prediction[0] - 4.0) ** 2.0
        loss.backward()
        var parameters = model.parameters()
        for i in range(len(parameters)):
            parameters[i].set_data(
                parameters[i].data() - 0.05 * parameters[i].grad()
            )

    print(model([Value(2.0)])[0])
```

The final layer created by `MLP(nin, nouts)` is linear; preceding layers apply ReLU.

## Mojo Design Notes

Python micrograd nodes are objects with reference identity. In Mojo, `Value` is a lightweight handle around a shared `_Node`, ensuring all downstream paths accumulate gradients into the same node.

Mojo uses typed operator overloads rather than Python runtime `isinstance` coercion. For example, `Value + Value` builds the graph directly, while `Value + Float64` first converts the scalar into a constant `Value` leaf.

`Neuron`, `Layer`, and `MLP` are move-only owners of their internal lists. They access owned children by index to borrow existing model components rather than implicitly copying them.
