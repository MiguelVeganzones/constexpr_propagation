# Introduction

# Compile time information
- On runtime vs compile time information

## Benefits of static information
## Costs of static specialization
## Compiler-side effects (VERY important)
inlining
unrolling
constant propagation
dead code elimination
vectorization enabling/disabling

# Compute kernel
Describe tesnor contraction amthematically and describe why it is an interesting
workload
## Computational complexity (naive impl)

# Design space
runtime vs compile-time parameters
flexibility vs performance
abstraction vs generated code simplicity

# Implementation
Describe the three levels of compile itme info (runtime rank, compile time rank,
compile time sizes) and how it affects v1-v3 tensor and contraction
implementations
## V1
### T1
### C1
## V2
### T2
### C2
## V3
### T3
### C3
## Compatibility

# Experimental setup

# Results
Introduce metrics
## Runtime 
## Compile time
## Code Generation
Constant propagation, code size, loop unrolling

# Discussion

# Conclusions



