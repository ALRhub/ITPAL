import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as sp_lin
import tikzplotlib
from plots.plot_util import build_to_cov, draw_2d_covariance

def interpolate_frob(target_mean, target_cov, old_mean, old_cov, eta):
    mean = (target_mean + eta * old_mean) / (eta + 1)
    cov = (target_cov + eta * old_cov) / (eta + 1)
    return mean, cov


def interpolate_wasserstein(target_mean, target_cov, old_mean, old_cov, eta):
    mean = (target_mean + eta * old_mean) / (eta + 1)
    sqrt_cov = (sp_lin.sqrtm(target_cov) + eta * sp_lin.sqrtm(old_cov)) / (eta + 1)
    return mean, sqrt_cov @ sqrt_cov


def interpolate_reverse_kl(target_mean, target_cov, old_mean, old_cov, eta):
    inv_cov = (np.linalg.inv(target_cov) + eta * np.linalg.inv(old_cov)) / (eta + 1)
    old_lin = np.linalg.solve(old_cov, old_mean)
    target_lin = np.linalg.solve(target_cov, target_mean)
    lin = (target_lin + eta * old_lin) / (eta + 1)
    mean = np.linalg.solve(inv_cov, lin)
    return mean, np.linalg.inv(inv_cov)


def interpolate_forward_kl(target_mean, target_cov, old_mean, old_cov, eta):
    mean = (target_mean + eta * old_mean) / (eta + 1)
    target_outer = np.outer(mean - target_mean, mean - target_mean)
    old_outer = np.outer(mean - old_mean, mean - old_mean)
    cov = (target_cov + target_outer + eta * (old_outer + old_cov)) / (eta + 1)
    return mean, cov


def entropy(cov):
    return 0.5 * np.log(2 * np.pi * np.e * sp_lin.det(cov))

mean_old = np.zeros(2)
cov_old = build_to_cov(np.array([5, 0.5]), -np.pi/3)
mean_target = 10 * np.ones(2)
cov_target = build_to_cov(np.array([5, 0.5]), np.pi / 3)

etas = [0.1, 0.33, 0.66, 1, 3.33, 6.66, 10]

cmap = matplotlib.cm.get_cmap('winter')

plt.figure()
plt.title("Frobenious")
draw_2d_covariance(mean_old, cov_old, chisquare_val=2, c="black")
draw_2d_covariance(mean_target, cov_target, chisquare_val=2, c="black", linestyle="dashed")
for i, eta in enumerate(etas):
    mean, cov = interpolate_frob(mean_target, cov_target, mean_old, cov_old, eta=eta)
    draw_2d_covariance(mean, cov, chisquare_val=2, c=cmap(i / len(etas)))

plt.legend(["old", "target"] + ["eta=" + str(x) for x in etas])
plt.grid("on")
plt.gca().set_aspect("equal")
#tikzplotlib.save("inter_frob.tex")

plt.figure()
plt.title("Wasserstein")
draw_2d_covariance(mean_old, cov_old, chisquare_val=2, c="black")
draw_2d_covariance(mean_target, cov_target, chisquare_val=2, c="black", linestyle="dashed")
for i, eta in enumerate(etas):
    mean, cov = interpolate_wasserstein(mean_target, cov_target, mean_old, cov_old, eta=eta)
    draw_2d_covariance(mean, cov, chisquare_val=2, c=cmap(i / len(etas)))

plt.grid("on")
plt.gca().set_aspect("equal")
#tikzplotlib.save("inter_wd.tex")

plt.figure()
plt.title("Reverse KL")
draw_2d_covariance(mean_old, cov_old, chisquare_val=2, c="black")
draw_2d_covariance(mean_target, cov_target, chisquare_val=2, c="black", linestyle="dashed")
for i, eta in enumerate(etas):
    mean, cov = interpolate_reverse_kl(mean_target, cov_target, mean_old, cov_old, eta=eta)
    draw_2d_covariance(mean, cov, chisquare_val=2, c=cmap(i / len(etas)))

plt.grid("on")
plt.gca().set_aspect("equal")
#tikzplotlib.save("inter_kl.tex")


etas = [0.01, 0.1,  1, 10, 100]
plt.figure()
plt.title("Forward KL")
draw_2d_covariance(mean_old, cov_old, chisquare_val=2, c="black")
draw_2d_covariance(mean_target, cov_target, chisquare_val=2, c="black", linestyle="dashed")
for i, eta in enumerate(etas):
    mean, cov = interpolate_forward_kl(mean_target, cov_target, mean_old, cov_old, eta=eta)
    draw_2d_covariance(mean, cov, chisquare_val=2, c=cmap(i / len(etas)))
plt.legend(["old", "target"] + ["eta=" + str(x) for x in etas])

plt.grid("on")
plt.gca().set_aspect("equal")

plt.figure()
ent_etas = 10 ** np.linspace(-4, 4, 1000)
plt.title("Entropies")
ent_frob = []
ent_wd = []
ent_rev_kl = []
ent_fwd_kl = []

for i, eta in enumerate(ent_etas):
    _, cov_fr = interpolate_frob(mean_target, cov_target, mean_old, cov_old, eta=eta)
    _, cov_wd = interpolate_wasserstein(mean_target, cov_target, mean_old, cov_old, eta=eta)
    _, cov_rev_kl = interpolate_reverse_kl(mean_target, cov_target, mean_old, cov_old, eta=eta)
    _, cov_fwd_kl = interpolate_forward_kl(mean_target, cov_target, mean_old, cov_old, eta=eta)
    ent_frob.append(entropy(cov_fr))
    ent_wd.append(entropy(cov_wd))
    ent_rev_kl.append(entropy(cov_rev_kl))
    ent_fwd_kl.append(entropy(cov_fwd_kl))


plt.plot(ent_etas, ent_frob)
plt.plot(ent_etas, ent_wd)
plt.plot(ent_etas, ent_rev_kl)
plt.plot(ent_etas, ent_fwd_kl)
plt.grid("on")
plt.semilogx()
plt.xlabel("eta")
plt.ylabel("entropy")
plt.legend(["frobenoius", "wasserstein", "revers kl", "foward kl"])
#tikzplotlib.save("inter_entropies.tex")

plt.show()

