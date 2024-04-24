# Indirect Model Predictive Control with Sparse Nonlinear Regression on Erdös-Rényi-generated Bernoulli-SIR Network Models

This repository branch contains supporting source code for a [conference paper](https://www.sciencedirect.com/science/article/pii/S2405896323017676).
The branch is used to generate, simulate, model reduce and control simple epidemiological SIR Network models. 

The project utilize simulation and regression routines in C++, with optimization performed in Python.

## Cpp

### Package Requirements:
see `find_package` in root CMakeLists.txt

### Build
```
mkdir Cpp/build && cd Cpp/build
cmake ..
cmake --build .
```

## Python

### Package Requirements:
See `requirements.txt` in `./Cpp/Executables/Plot/` for plotting from the `./Cpp/`-side, and `./Python/requirements.txt` for Python-side simulation requirements.


