import numpy as np
import copy


def get_permutation_matrix(sample):
    """ Get the permutation matrix from:

    F. Gamboa et al., “Global sensitivity analysis:
    a new generation of mighty estimators based on rank statistics,” Bernoulli hal-02474902v4 (2021).

    :param sample: (ndarray) get sample matrix size [N x d]
    :return: permutation matrix (ndarray) size [N x d]
    """
    N, d = np.shape(sample)
    permutation_matrix = np.zeros((N, d), dtype=int)
    for jj in range(d):
        # Get X ordering
        px = np.argsort(sample[:, jj])
        # Get phi
        phi_j = np.argsort(px)
        # get N_j with shift
        argpiinv = np.mod(phi_j+1, N)
        # get permutation matrix for this sample.
        permutation_matrix[:, jj] = px[argpiinv]
    return permutation_matrix


def generate_tensor_c(A, B, d):
    """Generate Random variables in a matrix A, B, C.
    :param d: (int) number of random variables.
    :param A: (ndarray) matrix of size (N x d).
    :param B: (ndarray) matrix of size (N x d).
    :return: (ndarray) C tensor, size (d x N x d).
    """
    # copy matrix B and only modify one column.
    C = np.zeros((d, np.shape(B)[0], np.shape(B)[1]))
    for ii in range(d):
        C[ii, :, :] = copy.deepcopy(B)
        C[ii, :, ii] = A[:, ii]
    return C


def generate_tensor_d(A, B, d):
    """Generate random variables in a matrix A, B, D.
    :param d: (int) number of random variables.
    :param A: (ndarray) matrix of size (N x d).
    :param B: (ndarray) matrix of size (N x d).
    :return: (ndarray) C tensor, size (d x N x d).
    """
    # copy matrix A and only modify one column.
    D = np.zeros((d, np.shape(A)[0], np.shape(A)[1]))
    for ii in range(d):
        D[ii, :, :] = copy.deepcopy(A)
        D[ii, :, ii] = B[:, ii]
    return D


def sobol_mc(A, B, C, D, f):
    """Evaluate the QoI given input parameters.
    :param A: (ndarray) sampled matrix A.
    :param B: (ndarray) sampled matrix B.
    :param C: (ndarray) sampled matrix C.
    :param D: (ndarray) sampled matrix D.
    :param f: model function (mapping from input to output)-- output is scalar.
    :return: (ndarray) f(A) size (N x 1), (ndarray) f(B) size (N x 1), (ndarray) f(C) size (N x d),
     (ndarray) f(D) size (N x d)
    """
    # initialize matrices.
    N, d = np.shape(A)
    YA = np.zeros(N)
    YB = np.zeros(N)
    YC = np.zeros((N, d))
    YD = np.zeros((N, d))

    # evaluate function for each sample.
    for ii in range(N):
        YA[ii] = f(z=A[ii, :])
        YB[ii] = f(z=B[ii, :])
        for jj in range(d):
            YC[ii, jj] = f(z=C[jj, ii, :])
            YD[ii, jj] = f(z=D[jj, ii, :])
    return YA, YB, YC, YD


def estimate_sobol(YA, YB, YC, A, type_estimator="janon"):
    """Estimate Sobol' indices (main effect) Si and (total effect) Ti.
    :param YA: (ndarray) output of input sampled matrix A.
    :param YB: (ndarray) output of input sampled matrix B.
    :param YC: (ndarray) output of input sampled matrix C.
    :param A: (ndarray) samples of matrix A.
    :param type_estimator: (str) type of estimator ("sobol", "jansen", "janon", etc...)
    :return: (ndarray) S (main effect) size (d x 1), (ndarray) T (total effect) size (d x 1)
    """
    main_effect = estimator_main_effect(YA=YA, YB=YB, YC=YC, N=len(YA), A=A, type_estimator=type_estimator)
    total_effect = estimator_total_effect(YA=YA, YB=YB, YC=YC, N=len(YA), type_estimator=type_estimator)
    return main_effect, total_effect


def estimator_main_effect(YA, YB, YC, N, A,  type_estimator):
    """Computes the main effect sobol indices.
    :param YA: (ndarray) output of sampled matrix A size (N x 1).
    :param YB: (ndarray) output of sampled matrix B size (N x 1).
    :param YC: (ndarray) output of sampled matrix C size (N x d).
    :param N: (int) number of samples.
    :param A: (ndarray) sample points for Gambao et al. sampler size (N x d)
    :param type_estimator: (str) type of estimator ("sobol", "owen", "saltelli", "janon", etc...)
    :return: (ndarray) sobol indices main effect, size (d x 1).
    """
    if type_estimator == "jansen":
        """
        Michiel J.W. Jansen. Analysis of variance designs for model output. Computer Physics Communications, 117(1):35–43, 1999. ISSN 0010-4655.
        """
        #
        # ((2 * N) / (2 * N - 1) * (1 / N * YA.T @ YC -
        #                              ((np.mean(YA) + np.mean(YB)) / 2) ** 2 +
        #                              (np.var(YA) + np.var(YB)) / (4 * N))) / np.var(YA, ddof=1)
        S = np.zeros(np.shape(YC)[1])
        for ii in range(np.shape(YC)[1]):
            S[ii] = 1 - (1 / (2 * N) * np.sum((YA - YC[:, ii]) ** 2)) / np.var(np.r_[YA, YB], ddof=1)
        return S


    elif type_estimator == "sobol":
        """
        A. Saltelli, Making best use of model evaluations to compute SA_tools indices, 
        Computer Physics Communications, 145 (2002), pp. 280 – 297.
        """
        return (1 / N * YA.T @ YC - np.mean(YA)**2) / np.var(YA, ddof=1)

    elif type_estimator == "saltelli":
        """
        Andrea Saltelli. Making best use of model evaluations to compute sensitivity indices. Computer
        Physics Communications, 145(2):280–297, May 2002.
        """
        V = 1/(N-1) * np.sum((YA - np.sqrt(np.mean(YA) * np.mean(YB)))**2)
        return ((1 / (N - 1)) * YA.T @ YC - np.mean(YA) * np.mean(YB)) / V

    elif type_estimator == "janon":
        """ 
        A. Janon, T. Klein, A. Lagnoux, M. Nodet, and C. Prieur, Asymptotic normality and efficiency
        of two Sobol index estimators, ESAIM: Probability and Statistics, 18 (2014), pp. 342–364.
        """
        N, d = np.shape(YC)
        numerator = np.zeros(d)
        for ii in range(d):
            f02 = np.mean(YA) * np.mean(YC[:, ii])
            numerator[ii] = (1 / N) * YA.T @ YC[:, ii] - f02
        return numerator / np.var(YA, ddof=1)

    elif type_estimator == "gamboa":
        """
        F. Gamboa et al., “Global sensitivity analysis: a new generation of mighty estimators based on rank 
        statistics,” Bernoulli hal-02474902v4 (2021).
        """
        permutation_matrix = get_permutation_matrix(sample=A)
        return ((1 / N) * YA.T @ YA[permutation_matrix] - np.mean(YA) ** 2) / np.var(YA, ddof=1)


def estimator_second_effect(ii, jj, YA, YB, YC, YD, N, type_estimator):
    """ Computes the second "iteration" Sobol' indices.
    :param YA: (ndarray) output of sampled matrix A size (N x 1).
    :param YB: (ndarray) output of sampled matrix B size (N x 1).
    :param YC: (ndarray) output of sampled matrix C size (N x d).
    :param YD: (ndarray) output of sampled matrix D size (N x d).
    :param type_estimator: (str) type of estimator ("sobol", "owen", "saltelli", "janon", etc...)
    :param N: (int) number of samples.
    :return: (ndarray) Sobol' indices main effect, size (d x 1).

    formula from :

    A. Saltelli, Making best use of model evaluations to compute SA_tools indices,
    Computer Physics Communications, 145 (2002), pp. 280 – 297.
    """
    y = np.r_[YA, YB]
    Vij = np.mean(YC[:, jj] * YD[:, ii] - YA * YB) / np.var(y)
    Si = estimator_main_effect(YA=YA, YB=YB, YC=YC, N=N, A=np.zeros(np.shape(YC)), type_estimator=type_estimator)[ii]
    Sj = estimator_main_effect(YA=YA, YB=YB, YC=YC, N=N, A=np.zeros(np.shape(YC)), type_estimator=type_estimator)[jj]
    return Vij - Si - Sj


def estimator_total_effect(YA, YB, YC, N, type_estimator="sobol"):
    """Computes the total effect sobol indices.
    :param YA: (ndarray) output of sampled matrix A size (N x 1).
    :param YB: (ndarray) output of sampled matrix B size (N x 1).
    :param YC: (ndarray) output of sampled matrix C size (N x d).
    :param N: number of samples.
    :param type_estimator: "sobol", "owen", "saltelli", "janon", etc...
    :return: (ndarray) Sobol' total effect indices, size (d times 1).
    """
    if type_estimator == "sobol":
        """unbiased
        Ilya M. Sobol. Sensitivity estimates for nonlinear mathematical models. Mathematical Modelling and
        Computational Experiments, 1(4):407–414, 1993
        """
        return 1 - (1 / N * YB.T @ YC - np.mean(YA) ** 2) / np.var(YA, ddof=1)

    elif type_estimator == "saltelli":
        """bias O(1/n)
        Andrea Saltelli. Making best use of model evaluations to compute sensitivity indices. Computer
        Physics Communications, 145(2):280–297, May 2002.
        """
        V = 1 / (N-1) * np.sum((YA - np.sqrt(np.mean(YA) * np.mean(YB)))**2)
        return 1 - (1 / (N-1) * YB.T @ YC - np.mean(YA) * np.mean(YB)) / V

    elif type_estimator == "janon":
        """ bias O(1/n)
        A. Janon, T. Klein, A. Lagnoux, M. Nodet, and C. Prieur, Asymptotic normality and efficiency
        of two Sobol index estimators, ESAIM: Probability and Statistics, 18 (2014), pp. 342–364.
        """
        N, d = np.shape(YC)
        denominator = np.zeros(d)
        numerator = np.zeros(d)
        for ii in range(d):
            f02 = np.mean(np.r_[YB, YC[:, ii]]) ** 2
            denominator[ii] = np.mean((YB**2 + YC[:, ii]**2)/2) - f02
            numerator[ii] = (1 / N) * YB.T @ YC[:, ii] - f02
        return 1 - numerator / denominator

    elif type_estimator == "owen":
        """unbaised
        A. B. Owen, Variance components and generalized Sobol’ indices, SIAM/ASA Journal on Uncertainty
        Quantification, 1 (2013), pp. 19–41, https://doi.org/10.1137/120876782, http://dx.doi.org/10.1137/
        120876782, https://arxiv.org/abs/http://dx.doi.org/10.1137/120876782.
        """
        T = np.zeros(np.shape(YC)[1])
        for ii in range(np.shape(YC)[1]):
            T[ii] = (1 / (2 * N) * np.sum((YB - YC[:, ii]) ** 2)) / np.var(YA, ddof=1)
        return T

    elif type_estimator == "gamboa":
        """does not exist for this estimator --> returning all zeros. 
         F. Gamboa et al., “Global sensitivity analysis: a new generation of mighty estimators
          based on rank statistics,” Bernoulli hal-02474902v4 (2021).
        """
        N, d = np.shape(YC)
        return np.zeros(d)