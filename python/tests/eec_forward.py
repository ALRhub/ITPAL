from util.Gaussian import Gaussian
import numpy as np
from util.sample import sample_sympd
import cpp_projection as projection
import time as t

"""benchmark and regression test for forward pass"""

np.random.seed(0)

num_gaussians = 32
dim = 32

old_dists = []
target_dists = []
for i in range(num_gaussians):
    old_dists.append(Gaussian(np.random.uniform(low=-0.1, high=0.1, size=dim), sample_sympd(dim)))
    #target_dists.append(old_dists[-1])
    target_dists.append(Gaussian(np.random.uniform(low=-0.1, high=0.1, size=dim), sample_sympd(dim)))


eps = 0.5

target_entropies = np.zeros(num_gaussians)
om = np.zeros([num_gaussians, dim])
oc = np.zeros([num_gaussians, dim, dim])

for i, od in enumerate(old_dists):
    target_entropies[i] = od.entropy() + np.random.uniform(-0.5, 0.5)
    om[i] = od.mean
    oc[i] = od.covar

tm = np.zeros([num_gaussians, dim])
tc = np.zeros([num_gaussians, dim, dim])

for i, td in enumerate(target_dists):
    tm[i] = td.mean
    tc[i] = td.covar

bp = projection.BatchedProjection(num_gaussians, dim, eec=True, constrain_entropy=True)

try:
    t0 = t.time()
    pm, pc = bp.forward(eps * np.ones(num_gaussians), target_entropies, om, oc, tm, tc)
    print("Projection took", t.time() - t0)
except Exception as e:
    print(e)

projected_dists = []
for i in range(num_gaussians):
    try:
        projected_dists.append(Gaussian(pm[i], pc[i]))
        print(target_entropies[i], projected_dists[i].entropy(), old_dists[i].entropy())
        print(projected_dists[i].kl(old_dists[i]))
    except:
        print("bla")



