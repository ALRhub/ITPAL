from util.central_differences import central_differences, directed_central_differences

import numpy as np

np.random.seed(0)

# simple linear function
W = np.random.normal(size=[2, 3])
lf = lambda _x: (W @ _x)

# sample input and forward pass
x = np.random.normal(size=3)
y = lf(x)

delta = 1e-6
dx = np.random.normal(size=3)
dx = dx / np.linalg.norm(dx)
# sample loss gradient and backward pass
dy = (lf(x + delta * dx) - lf(x - delta * dx)) / (2 * delta)
#dy = np.random.normal(size=2)
sec_dx = dy @ W

# numerical approximation of the gradient
#num_dy = (lf(x + delta * dx) - lf(x - delta * dx)) / (2 * delta)

# should be the same
print("dx", dx)
print("dx", sec_dx)
print( dx / sec_dx)
