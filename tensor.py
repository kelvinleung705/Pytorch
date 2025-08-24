import torch
import numpy as np
data = [[1, 2, 3],[4, 5, 6],[7, 8, 9]]
x_data = torch.tensor(data)
print(x_data)
print(f"First row: {x_data[0]}")
print(f"First column: {x_data[0:2, 0]}")
print(f"First column: {x_data[..., 0]}")
print(f"Last column: {x_data[:, -1]}")

tensor = torch.ones(4, 4)
tensor[:,1] = 0
print(tensor)

t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

# This computes the matrix multiplication between two tensors. y1, y2, y3 will have the same value
# ``tensor.T`` returns the transpose of a tensor
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)


# This computes the element-wise product. z1, z2, z3 will have the same value
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

print(y1)
print(z1)

agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))
