"""Module contains functions to interpolate 2d initial condition at 30Rs.

Authors: [Opal Issan] oissan@ucsd.edu
Version: August 21, 2022
"""
from scipy.interpolate import RegularGridInterpolator, interp2d
import numpy as np
import astropy.units as u


def interpolate_initial_condition(data, p_coord, t_coord, r_coord, p_interp, t_interp, r_interp):
    # coordinate grid
    pp, tt, rr = np.meshgrid(p_interp, t_interp, r_interp, indexing='ij')
    coordinate_grid = np.array([pp.T, tt.T, rr.T]).T
    # interpolate
    interp_function = RegularGridInterpolator(
        points=(p_coord, t_coord, r_coord),
        values=np.array(data),
        method="linear",
        bounds_error=False,
        fill_value=None)

    return interp_function(coordinate_grid)


def interpolate_ace_data(x, xp, fp, period):
    """return the interpolate values on new grid x of data (xp, fp)

    Parameters
    ----------
    x: array
        1d array of new grid coordinate
    xp: float
        1d array of old grid coordinate
    fp: bool
        1d array data on xp coordinate
    period: float
        periodicity

    Returns
    -------
    array
        1d array interpolated on xp
    """
    return np.interp(x=x, xp=xp[~np.isnan(fp)], fp=fp[~np.isnan(fp)], period=period)


def interp_2d_ace_hux(p_hux, r_hux, vr_hux, ACE_r, ACE_longitude):
    """ interpolate HUX results at ACE trajectory.

    :param p_hux: uniform grid in longitude. [0 -> 2pi]
    :param r_hux: uniform grid in the radial direction. [r_ss -> max(ACE_r)]
    :param vr_hux: radial velocity results. [km/s] on [r_hux, p_hux] grid.
    :param ACE_r: ACE radial trajectory in km.
    :param ACE_longitude: ACE longitude trajectory in degrees.
    :return: vr at ACE trajectory
    """
    f = interp2d(x=p_hux, y=r_hux, z=vr_hux, kind='linear')
    # initialize
    vr_ace_interp = np.zeros(len(ACE_r))
    # loop over each point
    for ii in range(len(ACE_r)):
        vr_ace_interp[ii] = f(x=ACE_longitude.to(u.rad).value[ii], y=ACE_r.to(u.km).value[ii])
    return vr_ace_interp


