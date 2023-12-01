---

# GPT-2 Model Implementation and Optimization Assignment

This repository contains the implementation and optimization of the GPT-2 model, covering three main tasks: Model Implementation and Checkpoints, Architectural Changes, and Distributed Training.

## Task 1: GPT-2 Model & Checkpoints (20 Points)

### GPT-2 Model Implementation
- Implemented the GPT-2 model with 125 million parameters using Python and PyTorch.
- Key aspects of the model, such as multi-head self-attention mechanism, feed-forward networks, and positional encoding, were considered.
- Followed the original GPT-2 design with token and positional embeddings.
- Abstained from using pre-built transformer libraries.

### Validation
- Loaded the original GPT-2 125M model checkpoints.
- Ran a sample prediction to verify the functioning of the implemented model.

## Task 2: Transformer Architectural Changes (40 Points)

### GPT-2 Model with Architectural Changes
- Extended the GPT-2 model to experiment with three architectural changes: Rotary Positional Embedding, Group Query Attention, and Sliding Window Attention.
- Implemented and commented on the model size, capabilities, potential pitfalls, and improvements after each change.

## Task 3: Training Loop Implementation (40 Points)

### Training Loop
- Created a training loop considering single GPU training, Distributed Data Parallel (DDP), and Fully Sharded Data Parallel (FSDP) options.
- Implemented a functional training loop compatible with various settings.

**Note**: Uncomment the relevant lines in the training loop for DDP and FSDP once a multi-GPU setup is available.

## Dependencies
- PyTorch

## Acknowledgments
- GPT-2 architecture inspired by the original paper and Andrej Karpathy's nanogpt repository.
- Additional insights from Su et. al. RoFormer, Ainslie et. al. GQA: Training Generalized Multi-Query Transformer, and Beltagy et. al. Longformer.

## Author
Kartikey Agarwal
