import numpy as np
from sobol import estimate_sobol, sobol_mc, generate_tensor_c, generate_tensor_d, estimator_second_effect

import matplotlib.pyplot as plt
import matplotlib

font = {'family': 'serif',
        'size': 13}

matplotlib.rc('font', **font)
matplotlib.rc('xtick', labelsize=15)
matplotlib.rc('ytick', labelsize=15)


# Define the model inputs
def f_fun(z, a=5, b=0.1):
    z1 = z[0]
    z2 = z[1]
    z3 = z[2]
    return np.sin(z1) + a * (np.sin(z2) ** 2) + b * np.sin(z1) * (z3 ** 4)


# dimensionality setting: number of samples and number of parameters
N = 10 ** 7
d = 3

# sample 2N of the input parameter.
A = np.random.uniform(low=-np.pi, high=np.pi, size=(N, d))
B = np.random.uniform(low=-np.pi, high=np.pi, size=(N, d))

# generate matrix C=[C1, C2, C3].
C = generate_tensor_c(A=A, B=B, d=d)
# generate matrix D=[D1, D2, D3].
D = generate_tensor_d(A=A, B=B, d=d)
print(np.shape(C))

# evaluate the model for given samples.
YA, YB, YC, YD = sobol_mc(A=A, B=B, C=C, D=D, f=f_fun)

# estimate sobol indices.
main_effect, total_effect = estimate_sobol(YA=YA, YB=YB, YC=YC, type="sobol")

print("S1 = ", main_effect)
print("T1 = ", total_effect)

# [0, 1, 2]
for ii in range(d):
    # [0, 1, 2]
    for jj in range(d):
        if ii < jj:
            print("S" + str(ii+1) + str(jj+1) + " = " + str(estimator_second_effect(ii=ii,
                                                                                    jj=jj,
                                                                                    YA=YA,
                                                                                    YB=YB,
                                                                                    YC=YC,
                                                                                    YD=YD,
                                                                                    N=N,
                                                                                    type_estimator="sobol")))
