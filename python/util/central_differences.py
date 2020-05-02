import numpy as np

def central_differences(fn, x, dim=1,delta=1e-8):
    x_size = 1 if np.isscalar(x) else x.shape[0]
    grad = np.zeros((dim if isinstance(dim, list) else [dim]) + [x_size])
    for i in range(x_size):
        offset = np.zeros(x_size)
        offset[i] = delta
        x_u = fn(x + offset)
        x_l = fn(x - offset)
        grad[..., i] = (x_u - x_l) / (2 * delta)
    return grad


if __name__ == "__main__":

    def f(x):
        return 3 * x**3 + 4 * x**2 + 2 * x + 3

    def df(x):
        return 9 * x ** 2 + 8 * x + 2


    for i in range(100):
        x = np.random.uniform(low=-10, high=10)
        print(df(x), central_differences(f, x, delta=1e-8))

