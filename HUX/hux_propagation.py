"""HUX-f and HUX-b propagation implemented. """
import numpy as np
import astropy.units as u


def apply_hux_f_model(initial_condition,
                      dr_vec,
                      dp_vec,
                      r0=(30 * u.solRad).to(u.km).value,
                      alpha=0.15,
                      rh=(50 * u.solRad).to(u.km).value,
                      add_v_acc=True,
                      theta=0):
    """Apply 1d upwind model to the inviscid burgers equation.
    r/phi grid. return and save all radial velocity slices.

    :param theta: constant theta slice in radians [-pi/2, pi/2]
    :param initial_condition: 1d array, initial condition (vr0). units = (km/sec).
    :param dr_vec: 1d array, mesh spacing in r. units = (km)
    :param dp_vec: 1d array, mesh spacing in p. units = (radians)
    :param alpha: float, hyper parameter for acceleration (default = 0.15).
    :param rh: float, hyper parameter for acceleration (default r=50*695700). units: (km)
    :param r0: float, initial radial location. units = (km).
    :param add_v_acc: bool, True will add acceleration boost.
    :return: velocity matrix dimensions (nr x np)
    """
    v = np.zeros((len(dr_vec) + 1, len(dp_vec) + 1))  # initialize array vr.
    v[0, :] = initial_condition

    # define omega rotation
    omega_rot = get_differential_rotation(theta=theta)

    if add_v_acc:
        v_acc = alpha * (v[0, :] * (1 - np.exp(-r0 / rh)))
        v[0, :] = v_acc + v[0, :]

    for i in range(len(dr_vec)):
        for j in range(len(dp_vec) + 1):

            if j == len(dp_vec):  # force periodicity
                v[i + 1, j] = v[i + 1, 0]

            else:
                if (omega_rot * dr_vec[i]) / (dp_vec[j] * v[i, j]) > 1:
                    print(dr_vec[i] - dp_vec[j] * v[i, j] / omega_rot)
                    print(i, j)  # courant condition

                frac1 = (v[i, j + 1] - v[i, j]) / v[i, j]
                frac2 = (omega_rot * dr_vec[i]) / dp_vec[j]
                v[i + 1, j] = v[i, j] + frac1 * frac2
    return v


def apply_hux_b_model(r_final,
                      dr_vec,
                      dp_vec,
                      alpha=0.15,
                      rh=(50 * u.solRad).to(u.km).value,
                      add_v_acc=True,
                      r0=(30 * u.solRad).to(u.km).value,
                      theta=0):
    """ Apply 1d backwards propagation.

    :param theta: constant theta slice in radians [-pi/2, pi/2]
    :param r_final: 1d array, initial velocity for backward propagation. units = (km/sec).
    :param dr_vec: 1d array, mesh spacing in r.
    :param dp_vec: 1d array, mesh spacing in p.
    :param alpha: float, hyper parameter for acceleration (default = 0.15).
    :param rh:  float, hyper parameter for acceleration (default r=50 rs). units: (km)
    :param add_v_acc: bool, True will add acceleration boost.
    :param r0: float, initial radial location. units = (km).
    :return: velocity matrix dimensions (nr x np) """

    v = np.zeros((len(dr_vec) + 1, len(dp_vec) + 1))  # initialize array vr.
    v[-1, :] = r_final

    # define omega rotation
    omega_rot = get_differential_rotation(theta=theta)

    for i in range(len(dr_vec)):
        for j in range(len(dp_vec) + 1):

            if j != len(dp_vec):
                # courant condition
                if (omega_rot * dr_vec[i]) / (dp_vec[j] * v[-(i + 1), j]) > 1:
                    print("CFL violated", dr_vec[i] - dp_vec[j] * v[-(i + 1), j] / omega_rot)
                    raise ValueError('CFL violated')

                frac2 = (omega_rot * dr_vec[i]) / dp_vec[j]
            else:
                frac2 = (omega_rot * dr_vec[i]) / dp_vec[0]

            frac1 = (v[-(i + 1), j - 1] - v[-(i + 1), j]) / v[-(i + 1), j]
            v[-(i + 2), j] = v[-(i + 1), j] + frac1 * frac2

    # add acceleration after upwind.
    if add_v_acc:
        v_acc = alpha * (v[0, :] * (1 - np.exp(-r0 / rh)))
        v[0, :] = -v_acc + v[0, :]

    return v


def get_differential_rotation(theta):
    """The angular frequency of the Sun’s rotation is evaluated at a constant Carrington latitude θ, estimated by
                                 Ωrot(θ) = 2π/25.38 − 2.77π/180 cos(θ)^2

    :param theta: constant theta slice [-pi/2, pi/2] in radians.
    :return: differential rotation rate in 1/seconds.
    """
    return ((np.pi * 2 / 25.38 - 2.77 * (np.pi / 180) * (np.cos(theta) ** 2))/u.day).to(1/u.s).value

