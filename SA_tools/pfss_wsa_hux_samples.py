import numpy as np
from sobol import generate_tensor_c, generate_tensor_d
import matplotlib.pyplot as plt
from scipy.stats import qmc
import matplotlib

font = {'family': 'serif',
        'size': 13}

matplotlib.rc('font', **font)
matplotlib.rc('xtick', labelsize=15)
matplotlib.rc('ytick', labelsize=15)


# dimensionality setting: number of samples and number of parameters
N = int(1e4)
d = 11
folder = "LHS"

# sample 2N of the input parameter.
sampler = qmc.LatinHypercube(d=d)
sample = sampler.random(n=2*N)

A = sample[:N, :]
B = sample[N:, :]

# lower and upper bounds
#          r_ss  v0.   v1.   alpha   beta   w  gamma delta  psi    acc     rh
l_bounds = [1.5, 200,   550,  0.05,   1,   0.01, 0.06,    1,  3,     0,    30]
u_bounds = [4,   400,   950,   0.5, 1.75,  0.4,  0.9,     5,  4,    0.5,  60]

# scale and sample
A_scaled = qmc.scale(sample=A,
                     l_bounds=l_bounds,
                     u_bounds=u_bounds)

B_scaled = qmc.scale(sample=B,
                     l_bounds=l_bounds,
                     u_bounds=u_bounds)


# save samples [A, B]
np.save(file="../SA_results/CR2058/" + str(folder) + "/samples/A_sample_scaled_" + str(N), arr=A_scaled)
np.save(file="../SA_results/CR2058/" + str(folder) + "/samples/B_sample_scaled_" + str(N), arr=B_scaled)

# generate matrix C=[C1, C2, C3, ..., C11].
C = generate_tensor_c(A=A_scaled, B=B_scaled, d=d)
np.save(file="../SA_results/CR2058/" + str(folder) + "/samples/C_sample_scaled_" + str(N), arr=C)

# generate matrix D=[D1, D2, D3, ..., D11].
D = generate_tensor_d(A=A_scaled, B=B_scaled, d=d)
np.save(file="../SA_results/CR2058/" + str(folder) + "/samples/D_sample_scaled_" + str(N), arr=D)

fig, ax = plt.subplots(ncols=d-1, nrows=d-1, figsize=(30, 30))

for jj in range(d-1):
    for ii in range(d-1):
        if jj <= ii:
            ax[ii, jj].scatter(A_scaled[:, jj], A_scaled[:, ii+1], s=2, c="r", alpha=0.2)
            ax[ii, jj].scatter(B_scaled[:, jj], B_scaled[:, ii + 1], s=2, c="b", alpha=0.2)
        else:
            ax[ii, jj].set_xticks([])
            ax[ii, jj].set_yticks([])
            ax[ii, jj].spines['top'].set_visible(False)
            ax[ii, jj].spines['right'].set_visible(False)
            ax[ii, jj].spines['bottom'].set_visible(False)
            ax[ii, jj].spines['left'].set_visible(False)

ax[-1, 0].set_xlabel(r"$r_{ss}$ [$R_{S}$]")
ax[-1, 1].set_xlabel(r"$v_{0}$ [km/s]")
ax[-1, 2].set_xlabel(r"$v_{1}$ [km/s]")
ax[-1, 3].set_xlabel(r"$\alpha$")
ax[-1, 4].set_xlabel(r"$\beta$")
ax[-1, 5].set_xlabel(r"$w$ [radians]")
ax[-1, 6].set_xlabel(r"$\gamma$")
ax[-1, 7].set_xlabel(r"$\delta$")
ax[-1, 8].set_xlabel(r"$\psi$")
ax[-1, 9].set_xlabel(r"$\alpha_{acc}$")

ax[0, 0].set_ylabel(r"$v_{0}$ [km/s]")
ax[1, 0].set_ylabel(r"$v_{1}$ [km/s]")
ax[2, 0].set_ylabel(r"$\alpha$")
ax[3, 0].set_ylabel(r"$\beta$")
ax[4, 0].set_ylabel(r"$w$ [radians]")
ax[5, 0].set_ylabel(r"$\gamma$")
ax[6, 0].set_ylabel(r"$\delta$")
ax[7, 0].set_ylabel(r"$\psi$")
ax[8, 0].set_ylabel(r"$\alpha_{acc}$")
ax[9, 0].set_ylabel(r"$r_{h}$ [$R_{S}$]")

fig.suptitle("Random Samples N = " + str(N))
plt.tight_layout()
#plt.savefig("figs/CR2408_random_samples_" + str(N) + ".png", dpi=500)
plt.show()
