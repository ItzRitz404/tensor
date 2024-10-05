# Automatic differentiation with torch.autograd

"""Consider the simplest one-layer neural network, with input x, parameters w and b, and some loss function. It can be defined in PyTorch in the following manner:"""

import torch

x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w) + b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

print("Gradient function for z =", z.grad_fn)
print("Gradient function for loss =", loss.grad_fn)

# computing gradients

loss.backward()
print(w.grad)
print(b.grad)