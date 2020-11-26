import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sp_stat

def draw_gauss(mean, var, low=-3, high=4, *args, **kwargs):
    x = np.linspace(low, high, (high - low) * 100)
    y = sp_stat.norm.pdf(x, loc=mean, scale=np.sqrt(var))
    return plt.plot(x, y, *args, **kwargs)


def interpolate_frob(target_mean, target_var, old_mean, old_var, eta):
    mean = (target_mean + eta * old_mean) / (eta + 1)
    cov = (target_var + eta * old_var) / (eta + 1)
    return mean, cov


def interpolate_wasserstein(target_mean, target_var, old_mean, old_var, eta):
    mean = (target_mean + eta * old_mean) / (eta + 1)
    sqrt_var = (np.sqrt(target_var) + eta * np.sqrt(old_var)) / (eta + 1)
    return mean, sqrt_var ** 2


def interpolate_kl(target_mean, target_cov, old_mean, old_cov, eta):
    mean = (target_mean + eta * old_mean) / (eta + 1)
    inv_cov = ((1.0 / target_cov) + eta * (1.0 / old_cov)) / (eta + 1)
    return mean,  1.0 /inv_cov


def entropy(var):
    return 0.5 * np.log(2 * np.pi * np.e * var)


mean_old = 0.0
cov_old = 1.0
mean_target = 1.0
cov_target = 0.5

etas = [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100]

cmap = matplotlib.cm.get_cmap('winter')
plt.figure()
plt.subplot(1, 3, 1)
plt.title("Frobenius")

draw_gauss(mean_old, cov_old, c="black")
draw_gauss(mean_target, cov_target, c="black", linestyle="dashed")
for i, eta in enumerate(etas):
    mean, cov = interpolate_frob(mean_target, cov_target, mean_old, cov_old, eta=eta)
    draw_gauss(mean, cov, c=cmap(i / len(etas)))
plt.legend(["old", "target"] + ["eta=" + str(x) for x in etas])
plt.grid("on")


plt.subplot(1, 3, 2)
plt.title("Wasserstein")
draw_gauss(mean_old, cov_old, c="black")
draw_gauss(mean_target, cov_target, c="black", linestyle="dashed")
for i, eta in enumerate(etas):
    mean, cov = interpolate_wasserstein(mean_target, cov_target, mean_old, cov_old, eta=eta)
    draw_gauss(mean, cov, c=cmap(i / len(etas)))
plt.grid("on")

plt.subplot(1, 3, 3)
plt.title("KL")
draw_gauss(mean_old, cov_old, c="black")
draw_gauss(mean_target, cov_target, c="black", linestyle="dashed")
for i, eta in enumerate(etas):
    mean, cov = interpolate_kl(mean_target, cov_target, mean_old, cov_old, eta=eta)
    draw_gauss(mean, cov, c=cmap(i / len(etas)))
plt.grid("on")

plt.show()