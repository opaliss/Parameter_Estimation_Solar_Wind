import numpy as np
import emcee.backends.backend
from multiprocessing import Pool

from model_chain import run_chain_of_models_mcmc, get_ace_date
from sunpy.coordinates.sun import carrington_rotation_time
import sunpy.map
import astropy.units as u


# get data
# exclude 2051 since there were 3 CMEs during this time period.
list_of_carrington_rotations = [2048]
num_cr = len(list_of_carrington_rotations)
ACE_longitude = []
ACE_latitude = []
ACE_r = []
ACE_vr = []
ACE_obstime = []
gong_map = []

for cr in list_of_carrington_rotations:
    # get ace data
    start_time = carrington_rotation_time(int(cr)).to_datetime()
    end_time = carrington_rotation_time(int(cr) + 1).to_datetime()
    result = get_ace_date(start_time=start_time, end_time=end_time)
    ACE_longitude.append(result[0])
    ACE_latitude.append(result[1])
    ACE_r.append(result[2])
    ACE_vr.append(result[3])
    ACE_obstime.append(result[4])

    # get gong synoptic maps
    gong = sunpy.map.Map('GONG/CR' + str(cr) + '/cr' + str(cr) + '.fits.gz')
    gong.meta["bunit"] = u.gauss
    gong.meta["DATE"] = str(result[4][-1])
    gong.meta["DATE_OBS"] = str(result[4][-1])
    gong_map.append(gong)


def model(theta):
    """The model evaluation used in MCMC for obtaining the radial velocity of 10 CRs.

    :param theta: list of model parameters.
    :return: list of radial velocity results.
    """
    # the parameters are stored as a vector of values, so unpack them
    r_ss, v0, v1, alpha, beta, w, gamma = theta
    # full list of parameters used in the chain of models, the last four are non-influential.
    coefficients_vec = [r_ss, v0, v1, alpha, beta, w, gamma, 3, 3.5, 0.15, 50]
    # vr initialization
    vr = []

    for jj in range(num_cr):
        vr.append(run_chain_of_models_mcmc(ACE_longitude=ACE_longitude[jj],
                                           ACE_latitude=ACE_latitude[jj],
                                           ACE_r=ACE_r[jj],
                                           gong_map=gong_map[jj],
                                           coefficients_vec=coefficients_vec))
    return vr


def log_prior(theta):
    """check if current parameters are within in the volume.

    :param theta: list of uncertain parameters.
    :return: 0 or -np.inf
    """
    # the parameters are stored as a vector of values, so unpack them
    r_ss, v0, v1, alpha, beta, w, gamma = theta
    # we are using only uniform priors
    if 1.5 <= r_ss <= 4. and 200. <= v0 <= 400. and 550. <= v1 <= 950. and \
            0.05 <= alpha <= 0.5 and 1. <= beta <= 1.75 and 0.01 <= w <= 0.4 and 0.06 <= gamma <= 0.9:
        return 0.
    else:
        return -np.inf


def log_likelihood(theta, sigma_scale=1):
    """returns the log likelihood for the specific set of parameters theta.

    :param theta: list of model parameters.
    :param sigma_scale: the covariance noise matrix is a diagonal matrix with these values on its diagonal.
    :return: float, for likelihood probability.
    """
    # model evaluation for specific parameter sample 'theta'
    model_eval = model(theta=theta)

    # initialize log-likelihood
    ll = 0
    for jj in range(num_cr):
        # find indexes where the measurements are nan.
        ACE_vr_is_nan = np.isnan(ACE_vr[jj])
        data_model_diff = (ACE_vr[jj][~ACE_vr_is_nan]).to(u.km/u.s).value - model_eval[jj][~ACE_vr_is_nan]
        if sigma_scale == 1:
            # note for scaling issues I am not squaring the error.
            ll += - 0.5 * np.linalg.norm(data_model_diff, ord=2) ** 2
        else:
            sigma_inv = np.diag(np.ones(len(data_model_diff))) * sigma_scale
            ll += - 0.5 * data_model_diff.T @ sigma_inv @ data_model_diff
    return ll


def log_posterior(theta):
    """returns the log posterior for the specific set of parameters theta.

    :param theta: list of model parameters.
    :return float, for posterior probability.
    """
    # evaluate prior
    lp = log_prior(theta=theta)
    if not np.isfinite(lp):
        return -np.inf
    else:
        return log_likelihood(theta=theta)


if __name__ == "__main__":
    # initialization (taken from the notebook)
    initial = np.array([3.16435228e+00, 3.49519082e+02,
                        7.23709870e+02, 1.77866287e-01,
                        1.14697740e+00, 2.46571473e-02, 6.01763647e-01])

    l_bounds = np.array([1.5, 200, 550, 0.05, 1, 0.01, 0.06])
    u_bounds = np.array([4, 400, 950, 0.5, 1.75, 0.4, 0.9])

    n_walkers = 15
    n_samples = 200
    n_dim = len(initial)
    n_crs = len(list_of_carrington_rotations)
    p0 = [np.array(initial) + 1e-5 * (u_bounds - l_bounds) * np.random.randn(n_dim) for i in range(n_walkers)]

    filename = "MCMC_results/CR2048.h5"
    backend = emcee.backends.HDFBackend(filename)
    # If you want to restart from the last sample,
    # you can just leave out the call to backends.HDFBackend.reset():
    backend.reset(n_walkers, n_dim)

    # cpu_count = n_walkers
    # with Pool(cpu_count) as pool:
    sampler = emcee.EnsembleSampler(nwalkers=n_walkers, ndim=n_dim, log_prob_fn=log_posterior, backend=backend)
    print("Running MCMC...")
    # pos, prob, state = sampler.run_mcmc(initial_state=p0, nsteps=n_samples,
    #                                     progress=True, store=True)
    # maximum number of samples
    max_n = int(1e5)

    # we will track how the average autocorrelation time estimate changes
    index = 0
    autocorr = np.empty(max_n)

    # This will be useful to testing convergence
    old_tau = np.inf

    # now we will sample for up to max_n steps
    for sample in sampler.sample(initial_state=p0, iterations=max_n, progress=True, store=True):
        # only check convergence every 100 steps
        if sampler.iteration % 100:
            continue

        # compute the autocorrelation time so far using tol=0 means that we'll
        # always get an estimate even if it isn't trustworthy
        tau = sampler.get_autocorr_time(tol=0)
        autocorr[index] = np.mean(tau)
        index += 1

        # check convergence
        converged = np.all(tau * 100 < sampler.iteration)
        converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
        if converged:
            break
        old_tau = tau

