# TensorFlow Aggregated Training
Batch Size is often an important hyper parameter required fro obtaining good results for training on larger models. Tensorflow builds support for large batch sizes through tools like Distributed training, but for small scale systems this is not viable. This library provides a set of Optimizers and Batch Normalization layers to allow for Large Batch Size Simulation on Small scale GPU's

# What this library provides
- Gradient Accumulated SGD
- Aggragated Standard Batch Normalization layer
- Aggragated Synchronized Batch Normalization layer 

# Optimizer Instantiation 

# Batch Norm Instantiation 

# What is being worked on
- Shuffle Batch Norm Support (used in MoCo model)
