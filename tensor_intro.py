import torch
import numpy as np

# initaialising a tensor directly from data

data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

"""Tensors can be created from NumPy arrays, and vice versa. Because numpy 'np_array' and tensor 'x_np' share the same memory location here, changing the value for one will change the other."""

# initialising a tensor from a numpy array
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

print(f"Numpy np_array value: \n {np_array} \n")
print(f"Tensor x_np value: \n {x_np} \n")

# The new tensor retains the properties (shape, data type) of the argument tensor, unless explicitly overridden
x_ones = torch.ones_like(x_data)  # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float)  # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")

"""Tensors can be created from NumPy arrays, and vice versa. Because numpy 'np_array' and tensor 'x_np' share the same memory location here, changing the value for one will change the other."""

shape = (2, 3)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")


"""Tensor attributes describe their shape, data type, and the device on which they're stored."""

tensor = torch.rand(3, 4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")


# We move our tensor to the GPU if available
if torch.cuda.is_available():
    tensor = tensor.to("cuda")

# Standard numpy-like indexing and slicing
tensor = torch.ones(4, 4)
print("First row: ", tensor[0])
print("First column: ", tensor[:, 0])
print("Last column:", tensor[..., -1])
tensor[:, 1] = 0
print(tensor)

# Joining tensors
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

"""Arithmetic operations"""

# This computes the matrix multiplication between two tensors. y1, y2, y3 will have the same value
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(tensor)
torch.matmul(tensor, tensor.T, out=y3)

# This computes the element-wise product. z1, z2, z3 will have the same value
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)
print(z3)

"""single element tensor"""

# If you have a one-element tensor—for example, by aggregating all values of a tensor into one value— you can convert it to a Python numerical value using item():

agg = tensor.sum()
agg_item = agg.item()
print("single element:: ", agg_item, type(agg_item))

"""In-place operations"""

# If you have a one-element tensor—for example, by aggregating all values of a tensor into one value— you can convert it to a Python numerical value using item():

print(tensor, "\n")
tensor.add_(5)
print(tensor)

"""bridge with NumPy"""

# Tensors on the CPU and NumPy arrays can share their underlying memory locations, and changing one will change the other.

# Tensor to NumPy array

t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

# A change in the tensor reflects in the NumPy array.

t.add_(1)
print(f"t: {t}")
print(f"n: {n}")

# NumPy array to Tensor
n = np.ones(5)
t = torch.from_numpy(n)

# Changes in the NumPy array reflects in the tensor.
np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")