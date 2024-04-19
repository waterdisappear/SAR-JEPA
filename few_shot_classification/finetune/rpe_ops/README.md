# 2D RPE Operators

## Build iRPE operators implemented by CUDA.
Although iRPE can be implemented by PyTorch native functions, the backward speed of PyTorch index function is very slow. We implement CUDA operators for more efficient training and recommend to build it. `nvcc` is necessary to build CUDA operators.
```bash
cd rpe_ops/
python setup.py install --user
```
