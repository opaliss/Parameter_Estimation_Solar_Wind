import numpy as np
import emcee.backends.backend
from multiprocessing import Pool

from model_chain import run_chain_of_models_mcmc, get_ace_date
from sunpy.coordinates.sun import carrington_rotation_time
import sunpy.map
import astropy.units as u

# global variable for multiprocessing purposes!
# get data for CR2058.
cr = int(2058)

# get ace data
# get time period
start_time = carrington_rotation_time(int(cr)).to_datetime()
end_time = carrington_rotation_time(int(cr) + 1).to_datetime()
result = get_ace_date(start_time=start_time, end_time=end_time)
# get trajectory location
ACE_longitude = result[0]
ACE_latitude = result[1]
ACE_r = result[2]
ACE_vr = result[3]
ACE_obstime = result[4]


# get gong synoptic maps from GONG folder (saved as fits)
gong = sunpy.map.Map('GONG/CR' + str(cr) + '/cr' + str(cr) + '.fits.gz')
gong.meta["bunit"] = u.gauss
gong.meta["DATE"] = str(result[4][-1])
gong.meta["DATE_OBS"] = str(result[4][-1])
gong_map = gong


def model(theta):
    """The model evaluation used in MCMC for obtaining the radial velocity at L1.

    :param theta: list of model parameters.
    :return: list of radial velocity results.
    """
    # the parameters are stored as a vector of values, so unpack them
    r_ss, v0, v1, alpha, beta, w, gamma = theta
    # full list of parameters used in the chain of models, the last four are non-influential
    # so we fix them to their nominal values.
    coefficients_vec = [r_ss, v0, v1, alpha, beta, w, gamma, 1.75, 3, 0.15, 50]
    return run_chain_of_models_mcmc(ACE_longitude=ACE_longitude,
                                    ACE_latitude=ACE_latitude,
                                    ACE_r=ACE_r,
                                    gong_map=gong_map,
                                    coefficients_vec=coefficients_vec)


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
        # return a finite value (does not matter what it is)
        return 0.
    else:
        # outside of the box (volume).
        return -np.inf


def log_likelihood(theta, sigma_scale=10):
    """returns the log likelihood for the specific set of parameters theta.

    :param theta: list of model parameters.
    :param sigma_scale: the covariance noise matrix is a diagonal matrix with these values on its diagonal.
    :return: float, for likelihood probability.
    """
    # model evaluation for specific parameter sample 'theta'
    model_eval = model(theta=theta)

    # find indexes where the measurements are nan.
    ACE_vr_is_nan = np.isnan(ACE_vr)
    data_model_diff = (ACE_vr[~ACE_vr_is_nan]).to(u.km / u.s).value - model_eval[~ACE_vr_is_nan]
    if sigma_scale == 1:
        # if Sigma = I then the log-likelihood simplifies.
        ll = - 0.5 * np.linalg.norm(data_model_diff, ord=2) ** 2
    else:
        sigma_inv = np.diag(np.ones(len(data_model_diff))) * (1/sigma_scale)
        ll = - 0.5 * data_model_diff.T @ sigma_inv @ data_model_diff
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
    # initial = np.array([3.16435228e+00, 3.49519082e+02,
    #                     7.23709870e+02, 1.77866287e-01,
    #                     1.14697740e+00, 2.46571473e-02, 6.01763647e-01])
    # number of walkers
    n_walkers = 15
    # number of uncertain parameters
    n_dim = 7

    filename = "MCMC_results/CR" + str(cr) + ".h5"
    backend = emcee.backends.HDFBackend(filename)
    # get the previous run last sample.
    initial = backend.get_chain(flat=False)[-1, :, :]

    # If you want to restart from the last sample,
    # you can just leave out the call to backends.HDFBackend.reset():
    # backend.reset(n_walkers, n_dim)

    # cpu_count = n_walkers
    # with Pool(cpu_count) as pool:
    sampler = emcee.EnsembleSampler(nwalkers=n_walkers,
                                    ndim=n_dim,
                                    log_prob_fn=log_posterior,
                                    backend=backend)
    print("Running MCMC...")
    # pos, prob, state = sampler.run_mcmc(initial_state=p0, nsteps=n_samples,
    #                                     progress=True, store=True)
    # maximum number of samples
    max_n = int(400)

    # we will track how the average autocorrelation time estimate changes
    index = 0
    autocorr = np.empty(max_n)

    # This will be useful to testing convergence
    old_tau = np.inf

    # now we will sample for up to max_n steps
    for sample in sampler.sample(initial_state=initial, iterations=max_n, progress=True, store=True):
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
