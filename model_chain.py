import numpy as np
import astropy.units as u
from tools.interpolate import interpolate_ace_data, interp_2d_ace_hux
from sunpy.coordinates import frames
from astropy.coordinates import SkyCoord
from heliopy.data import ace
from HUX.hux_propagation import apply_hux_f_model
import astropy.constants as const
import sunpy.map.mapbase
import sunpy.map
import pfsspy
from pfsspy import tracing
from scipy.stats import pearsonr
import astropy
import os
import matplotlib.pyplot as plt
from sunpy.sun import constants


def convert_vector_to_dict(samples):
    """convert an array/list of coefficients to dictionary

    :param samples: 1d array or list of size 11 (uncertain input parameters). Note: order matters...
    The order: {r_ss, v0, v1, alpha, beta, w, gamma, delta, psi, alpha_acc, rh}
    :return: dictionary of size 11 (uncertain input parameters).
    """
    return {"r_ss": samples[0],
            "v0": samples[1],
            "v1": samples[2],
            "alpha": samples[3],
            "beta": samples[4],
            "w": samples[5],
            "gamma": samples[6],
            "delta": samples[7],
            "psi": samples[8],
            "alpha_acc": samples[9],
            "rh": samples[10]}


def get_ace_date(start_time, end_time):
    """ A function to get ACE position and measurements in HG coordinates for a single Carrington Rotation.

    :param start_time: datetime object
    :param end_time: datetime object
    :return: ACE longitude [deg.] (HG),
             ACE latitude [deg.] (HG),
             ACE radial distance from the Sun [km] (HG),
             Solar Wind Vr [km/s] (HG),
             ACE observation time [datetime] (HG)
    """
    ACE = ace.swe_h2(start_time, end_time)
    # get position of ACE position and measured velocity in Geocentric Solar Ecliptic (GSE) coordinates.
    # this system has its X axis towards the Sun and its Z axis perpendicular to the plane of the Earth's orbit
    # around the Sun (positive North).
    # define as a astropy skycoord object: https://docs.astropy.org/en/stable/coordinates/velocities.html
    GSE_COORDS = SkyCoord(x=ACE.quantity('SC_pos_GSE_0'),
                          y=ACE.quantity('SC_pos_GSE_1'),
                          z=ACE.quantity('SC_pos_GSE_2'),
                          v_x=ACE.quantity('V_GSE_0'),
                          v_y=ACE.quantity('V_GSE_1'),
                          v_z=ACE.quantity('V_GSE_2'),
                          representation_type='cartesian',
                          obstime=ACE.time,
                          frame=frames.GeocentricSolarEcliptic)
    # convert to Heliographic Carrington Coordinate system.
    # the origin is the center of the Sun. The Z-axis (+90 degrees latitude) is aligned with the Sun’s north pole.
    # the X-axis and Y-axis rotate with a period of 25.38 days.
    HG_COORDS = GSE_COORDS.transform_to(frames.HeliographicCarrington(observer='sun'))
    # location in spherical coordinates (longitude, latitude, radial distance).
    ACE_longitude = HG_COORDS.lon.to(u.deg)
    ACE_latitude = HG_COORDS.lat.to(u.deg)
    ACE_r = HG_COORDS.radius.to(u.km)
    return ACE_longitude, ACE_latitude, ACE_r, np.abs(ACE.quantity('V_GSE_0')), ACE.time


def pfss2flines(pfsspy_out, nth=180, nph=360, trace_from_SS=False, max_steps=1000):
    """Field line tracing from photosphere or source surface.

    :param pfsspy_out: pfsspy output object
    :param nth: number of tracing grid points in latitude
    :param nph: number of tracing grid points in longitude
    :param trace_from_SS: if False : start trace from photosphere, if True, start tracing from source surface
    :param max_steps: max steps tracer should take before giving up
    :return: trace of each field line back to photosphere or out to solar surface
    """
    lons, lats = np.meshgrid(np.linspace(0, 360, nph), np.linspace(-90, 90, nth))
    if not trace_from_SS:
        # trace up from photosphere
        alt = 1 * u.R_sun
    else:
        # trace down from solar surface
        alt = 2.5 * u.R_sun
    # get all tuples of the coordinates.
    alt = [alt] * len(lons.ravel())
    # create an astropy seed in HG coordinates.
    seeds = astropy.coordinates.SkyCoord(lons.ravel() * u.deg,
                                         lats.ravel() * u.deg,
                                         alt,
                                         frame=pfsspy_out.coordinate_frame)
    # trace all field lines.
    field_lines = pfsspy_out.trace(pfsspy.tracing.FortranTracer(max_steps=max_steps), seeds)
    if not trace_from_SS:
        return field_lines.polarities.reshape([nth, nph])
    else:
        return field_lines.expansion_factors.reshape([nth, nph])


def distance_to_coronal_hole_boundary(topologies, field_lines_fp):
    """compute the distance to the nearest coronal hole boundary.
    **in practice: compute the distance from the footprint to the nearest closed footprint.

    :param topologies: coronal hole map. array of size [n_theta~180, n_phi~360]
    :param field_lines_fp: field line tracing for ace simulation_output.
    :return: distance to coronal hole for each solar source field line footpoint on the photosphere.
    """
    # initialize the distance to coronal hole vector.
    d = np.zeros(len(field_lines_fp))
    # longitude and latitude uniform grids in radians.
    latitude = np.linspace(-np.pi / 2, np.pi / 2, np.shape(topologies)[0])
    longitude = np.linspace(0, 2 * np.pi, np.shape(topologies)[1])
    # location of closed magnetic field lines (footprint).
    latitude = latitude[np.where(topologies == 0)[0]]
    longitude = longitude[np.where(topologies == 0)[1]]

    for ii in range(len(d)):
        try:
            phi2 = field_lines_fp[ii].solar_footpoint.lon.to(u.rad).value
            theta2 = field_lines_fp[ii].solar_footpoint.lat.to(u.rad).value
            d_full_sun = np.arccos(np.cos(latitude) * np.cos(theta2) * (np.sin(phi2) * np.sin(longitude) +
                                                                        np.cos(longitude) * np.cos(phi2)) + np.sin(
                latitude) * np.sin(theta2))
            d[ii] = np.min(d_full_sun)
        except:
            d[ii] = 0
    return d


def wsa(fp, d, coeff):
    """ WSA is an empirical model of the ambient solar wind.
    The WSA model is based on the inverse relationship between the solar wind speed
    and the magnetic field expansion factor 𝑓𝑝 and the minimum angular distance that
    an open field footpoint lies from the nearest coronal hole boundary 𝑑.

    :param fp: magnetic expansion factor on computational grid.
    :param d: distance to coronal hole boundary computational grid.
    :param coeff: model parameters (8 parameters)
    :return: velocity profile at the source surface.
    """
    a1 = (coeff["v1"] - coeff["v0"]) / ((1. + fp) ** coeff["alpha"])
    a2 = (coeff["beta"] - coeff["gamma"] * np.exp(-(d / coeff["w"]) ** coeff["delta"]))
    a3 = a2 ** coeff["psi"]
    return coeff["v0"] + a1 * a3


def run_chain_of_models(ACE_longitude,
                        ACE_latitude,
                        ACE_r,
                        ACE_vr,
                        gong_map,
                        coefficients_vec,
                        n_r_pfss=100,
                        n_r_hux=300,
                        n_theta_ch=180,
                        n_phi_ch=360,
                        QoI="ALL",
                        sample_id=None,
                        id=0,
                        CR="2053",
                        folder="B"):
    """functionality to run a chain of empirical and reduced-physics models:

                                [PFSS] -----> [WSA] -----> [HUX]

    :param CR: current carrington rotation EX: "2053"
    :param sample_id: save results in file with this ID.
    :param ACE_vr: ACE radial velocity in units of [km/s].
    :param ACE_r: ACE simulation_output distance from the Sun.
    :param ACE_latitude: ACE simulation_output latitude [-90, 90] in degrees.
    :param ACE_longitude: ACE simulation_output longitude [0, 360] in degrees.
    :param n_phi_ch: number of phi mesh grid points in tracing coronal hole maps.
    :param n_theta_ch: number of theta mesh grid points in tracing coronal hole maps.
    :param n_r_hux: number of radial mesh grid points in the HUX finite difference uniform mesh.
    :param QoI: Add quantity of interest. current options:{"RMSE", "MSE", "PCC", "ALL"}
    :param coefficients_vec: 11 parameters of PSS, WSA, and HUX.
    :param n_r_pfss: number of radial cells in finite differencing in PFSS.
    :param gong_map: SunPy Map object.
    :return: QoI evaluated for a specific CR. (float)
    """
    # convert coefficients to dictionary for readability.
    coeff = convert_vector_to_dict(samples=coefficients_vec)

    # PFSS parameters + simulate.
    pfss_in = pfsspy.Input(br=gong_map, nr=n_r_pfss, rss=coeff["r_ss"])
    pfss_out = pfsspy.pfss(input=pfss_in)

    # trace the magnetic field lines for the ACE projection to obtain the magnetic expansion factor.
    tracer = tracing.FortranTracer()
    seeds = SkyCoord(ACE_longitude.to(u.rad),
                     ACE_latitude.to(u.rad),
                     coeff["r_ss"] * const.R_sun,
                     frame=pfss_out.coordinate_frame)
    field_lines_fp = tracer.trace(seeds=seeds, output=pfss_out)
    fp_ace_traj = field_lines_fp.expansion_factors

    # coronal hole mapping
    topologies = pfss2flines(pfsspy_out=pfss_out, nth=n_theta_ch, nph=n_phi_ch)
    d_ace_traj = distance_to_coronal_hole_boundary(topologies=topologies, field_lines_fp=field_lines_fp)

    # WSA empirical model.
    v_wsa = wsa(fp=fp_ace_traj, d=d_ace_traj, coeff=coeff)

    # define HUX grid.
    r_hux = (np.linspace(coeff["r_ss"], np.max(ACE_r.to(u.solRad)).value, n_r_hux) * u.solRad).to(u.km).value
    p_hux = np.linspace(0, 2 * np.pi, len(ACE_longitude))

    # interpolate WSA velocity results on HUX grid.
    v_wsa_interp = interpolate_ace_data(x=p_hux, xp=ACE_longitude.to(u.rad).value, fp=v_wsa, period=2 * np.pi)

    # simulate HUX for the entire grid [phi, r].
    vr_hux_wsa = apply_hux_f_model(initial_condition=v_wsa_interp,
                                   dr_vec=r_hux[1:] - r_hux[:-1],
                                   dp_vec=p_hux[1:] - p_hux[:-1],
                                   alpha=coeff["alpha_acc"],
                                   rh=coeff["rh"],
                                   r0=coeff["r_ss"],
                                   theta=np.mean(np.pi / 2 - ACE_latitude.to(u.rad).value))

    # interpolate back to ACE longitude and radial trajectory..
    vr_hux_wsa_interp = interp_2d_ace_hux(p_hux=p_hux,
                                          r_hux=r_hux,
                                          vr_hux=vr_hux_wsa,
                                          ACE_r=ACE_r,
                                          ACE_longitude=ACE_longitude)
    # save results.
    if sample_id is not None:
        new_dir = os.getcwd() + "/SA_results/CR" + str(CR) + "/" + str(folder) + "/simulation_output/" + \
                  str(sample_id) + str(id)
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        np.save(file=new_dir + "/vr_hux_sample_" + str(sample_id) + str(id), arr=vr_hux_wsa_interp)
        np.save(file=new_dir + "/vr_wsa_sample_" + str(sample_id) + str(id), arr=v_wsa)
        np.save(file=new_dir + "/distance_to_coronal_hole_" + str(sample_id) + str(id), arr=d_ace_traj)
        np.save(file=new_dir + "/expansion_factor_" + str(sample_id) + str(id), arr=fp_ace_traj)

    # find where we have issues in the measurements.
    ACE_vr_is_nan = np.isnan(ACE_vr)
    if QoI == "RMSE":
        # root mean squared error
        return np.sqrt(np.mean((vr_hux_wsa_interp[~ACE_vr_is_nan] - ACE_vr.to(u.km / u.s).value[~ACE_vr_is_nan]) ** 2))
    elif QoI == "MAE":
        # mean absolute error
        return np.mean(np.abs(vr_hux_wsa_interp[~ACE_vr_is_nan] - ACE_vr.to(u.km / u.s).value)[~ACE_vr_is_nan])
    elif QoI == "PCC":
        # pearson correlation coefficient
        return pearsonr(vr_hux_wsa_interp[~ACE_vr_is_nan], ACE_vr.to(u.km / u.s).value[~ACE_vr_is_nan])[0]
    elif QoI == "ALL":
        # return RMSE, MAE, PCC (in order)
        return [
            np.sqrt(np.mean((vr_hux_wsa_interp[~ACE_vr_is_nan] - ACE_vr.to(u.km / u.s).value[~ACE_vr_is_nan]) ** 2)),
            np.mean(np.abs(vr_hux_wsa_interp[~ACE_vr_is_nan] - ACE_vr.to(u.km / u.s).value[~ACE_vr_is_nan])),
            pearsonr(vr_hux_wsa_interp[~ACE_vr_is_nan], ACE_vr.to(u.km / u.s).value[~ACE_vr_is_nan])[0]]
    else:
        return None


def run_chain_of_models_mcmc(ACE_longitude,
                             ACE_latitude,
                             ACE_r,
                             gong_map,
                             coefficients_vec,
                             n_r_pfss=100,
                             n_r_hux=300,
                             n_theta_ch=180,
                             n_phi_ch=360):
    """functionality to run a chain of empirical and reduced-physics models:

                                [PFSS] -----> [WSA] -----> [HUX]

    :param ACE_r: ACE simulation_output distance from the Sun.
    :param ACE_latitude: ACE simulation_output latitude [-90, 90] in degrees.
    :param ACE_longitude: ACE simulation_output longitude [0, 360] in degrees.
    :param n_phi_ch: number of phi mesh grid points in tracing coronal hole maps.
    :param n_theta_ch: number of theta mesh grid points in tracing coronal hole maps.
    :param n_r_hux: number of radial mesh grid points in the HUX finite difference uniform mesh.
    :param coefficients_vec: 11 parameters of PSS, WSA, and HUX.
    :param n_r_pfss: number of radial cells in finite differencing in PFSS.
    :param gong_map: SunPy Map object.
    :return: QoI evaluated for a specific CR. (float)
    """
    # convert coefficients to dictionary for readability.
    coeff = convert_vector_to_dict(samples=coefficients_vec)

    # PFSS parameters + simulate.
    pfss_in = pfsspy.Input(br=gong_map, nr=n_r_pfss, rss=coeff["r_ss"])
    pfss_out = pfsspy.pfss(input=pfss_in)

    # trace the magnetic field lines for the ACE projection to obtain the magnetic expansion factor.
    tracer = tracing.FortranTracer()
    seeds = SkyCoord(ACE_longitude.to(u.rad),
                     ACE_latitude.to(u.rad),
                     coeff["r_ss"] * const.R_sun,
                     frame=pfss_out.coordinate_frame)
    field_lines_fp = tracer.trace(seeds=seeds, output=pfss_out)
    fp_ace_traj = field_lines_fp.expansion_factors

    # coronal hole mapping
    topologies = pfss2flines(pfsspy_out=pfss_out, nth=n_theta_ch, nph=n_phi_ch)
    d_ace_traj = distance_to_coronal_hole_boundary(topologies=topologies, field_lines_fp=field_lines_fp)

    # WSA empirical model.
    v_wsa = wsa(fp=fp_ace_traj, d=d_ace_traj, coeff=coeff)

    # define HUX grid.
    r_hux = (np.linspace(coeff["r_ss"], np.max(ACE_r.to(u.solRad)).value, n_r_hux) * u.solRad).to(u.km).value
    p_hux = np.linspace(0, 2 * np.pi, len(ACE_longitude))

    # interpolate WSA velocity results on HUX grid.
    v_wsa_interp = interpolate_ace_data(x=p_hux, xp=ACE_longitude.to(u.rad).value, fp=v_wsa, period=2 * np.pi)

    # simulate HUX for the entire grid [phi, r].
    vr_hux_wsa = apply_hux_f_model(initial_condition=v_wsa_interp,
                                   dr_vec=r_hux[1:] - r_hux[:-1],
                                   dp_vec=p_hux[1:] - p_hux[:-1],
                                   alpha=coeff["alpha_acc"],
                                   rh=coeff["rh"],
                                   r0=coeff["r_ss"],
                                   theta=np.mean(np.pi / 2 - ACE_latitude.to(u.rad).value))

    # interpolate back to ACE longitude and radial trajectory..
    vr_hux_wsa_interp = interp_2d_ace_hux(p_hux=p_hux,
                                          r_hux=r_hux,
                                          vr_hux=vr_hux_wsa,
                                          ACE_r=ACE_r,
                                          ACE_longitude=ACE_longitude)
    return vr_hux_wsa_interp


if __name__ == "__main__":
    import datetime as dt
    from sunpy.coordinates.sun import carrington_rotation_time
    import time

    CR = "2048"
    folder = "LHS"
    start_time = carrington_rotation_time(int(CR)).to_datetime()
    end_time = carrington_rotation_time(int(CR) + 1).to_datetime()

    # get ace data
    ACE_longitude, ACE_latitude, ACE_r, ACE_vr, ACE_obstime = get_ace_date(start_time=start_time, end_time=end_time)

    A = np.load("SA_results/CR" + str(CR) + "/" + str(folder) + "/samples/A_sample_scaled_10000.npy")
    # size [N, d]
    B = np.load("SA_results/CR" + str(CR) + "/" + str(folder) + "/samples/B_sample_scaled_10000.npy")
    # size [d, N, d]
    C = np.load("SA_results/CR" + str(CR) + "/" + str(folder) + "/samples/C_sample_scaled_10000.npy")
    # get gong synoptic map
    gong_map = sunpy.map.Map('GONG/CR' + str(CR) + '/cr2048.fits.gz')
    gong_map.meta["bunit"] = u.gauss

    idx = 441
    idx_d = 0
    coefficients = C[idx_d, idx, :]

    stime = time.time()
    run_chain_of_models(ACE_longitude=ACE_longitude,
                        ACE_latitude=ACE_latitude,
                        ACE_r=ACE_r,
                        ACE_vr=ACE_vr,
                        gong_map=gong_map,
                        coefficients_vec=coefficients,
                        QoI="RMSE",
                        sample_id="C",
                        id=str(idx) + "_" + str(idx_d),
                        CR=CR,
                        folder=folder)

    print(time.time() - stime)
