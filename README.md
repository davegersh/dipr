# Deep Intelligence in Pure Rust (DIPR)
Welcome to DIPR! A project focused on implementing a minimalistic framework for creating deep neural networks with *zero* external dependencies!

This is mostly a project to challenge myself with both Rust ML at it's most foundational.

## Features
Here is a list of some of the notable features in DIPR:

- Complete implementation of backpropagation with modular layers
- Simple model instancing system (PyTorch-esque)
- Custom n-dimensional tensor implementation with convenient ops
- Fully-functional integration test of an Iris classifier

Note that DIPR is currently tested to be functional with a simple Multilayer Perceptron. 
More models and layers will be supported as development continues!

## Running
At the moment, DIPR is not yet available on crates.io, so you'll need to download it by cloning this repo.

DIPR simply needs rust and cargo to work, and requires no external crates. 

Tests can be run with the command:
```
cargo test
```

## Next Steps 
Below is a non-comphrensive list of ideas or things to work on:
- [ ] Argmax
- [ ] Metrics module for calculating categorical accuracy, etc.
- [ ] Support for Mini-batch GD and Adam optimizers
- [ ] MNIST Integration Test
- [ ] Better randomness
