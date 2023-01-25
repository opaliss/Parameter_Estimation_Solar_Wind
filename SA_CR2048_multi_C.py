from model_chain import run_chain_of_models, get_ace_date
from multiprocess import Pool, cpu_count, get_context, set_start_method
from sunpy.coordinates.sun import carrington_rotation_time
import datetime as dt
import numpy as np
import sunpy.map
from functools import partial
import matplotlib.pyplot as plt
import os


# run chain model for samples in "A" and "B"
def run_chain_models_AB_samples(idx, ACE_longitude_, ACE_latitude_, ACE_r_,
                                ACE_vr_, gong_map_, A_val, QoI_, sample_id_, CR_, folder_):
    return run_chain_of_models(ACE_longitude=ACE_longitude_,
                               ACE_latitude=ACE_latitude_,
                               ACE_r=ACE_r_,
                               ACE_vr=ACE_vr_,
                               gong_map=gong_map_,
                               coefficients_vec=A_val[idx, :],
                               QoI=QoI_,
                               sample_id=sample_id_,
                               id=str(idx),
                               CR=CR_,
                               folder=folder_)


# run chain model for samples in "C" and "D"
def run_chain_models_CD_samples(idx, d_val, ACE_longitude_, ACE_latitude_, ACE_r_,
                                ACE_vr_, gong_map_, C_val, QoI_, sample_id_, CR_, folder_):
    YC_results = np.zeros((d_val, 3))
    for jj in range(d_val):
        dir_location = os.getcwd() + "/SA_results/CR" + \
                       str(CR) + "/" + str(folder) + "/simulation_output/" + \
                       str(sample_id_) + str(idx) + "_" + str(jj)
        if not os.path.exists(dir_location):
            YC_results[jj, :] = run_chain_of_models(ACE_longitude=ACE_longitude_,
                                                    ACE_latitude=ACE_latitude_,
                                                    ACE_r=ACE_r_,
                                                    ACE_vr=ACE_vr_,
                                                    gong_map=gong_map_,
                                                    coefficients_vec=C_val[jj, idx, :],
                                                    QoI=QoI_,
                                                    sample_id=sample_id_,
                                                    id=str(idx) + "_" + str(jj),
                                                    CR=CR_,
                                                    folder=folder_)
    return YC_results


if __name__ == "__main__":
    # set up carrington rotation.
    CR = "2048"
    folder = "LHS"
    start_time = carrington_rotation_time(int(CR)).to_datetime()
    end_time = carrington_rotation_time(int(CR) + 1).to_datetime()

    # get ace data.
    ACE_longitude, ACE_latitude, ACE_r, ACE_vr, ACE_obstime = get_ace_date(start_time=start_time, end_time=end_time)

    # get gong synoptic map.
    gong_map = sunpy.map.Map('GONG/CR' + str(CR) + '/mrmqs061004t2131c2048_000.fits.gz')
    gong_map.meta["bunit"] = "gauss"
    gong_map.meta["DATE"] = str(ACE_obstime[-1])
    gong_map.meta["DATE_OBS"] = str(ACE_obstime[-1])

    # get samples
    # size [N, d]
    A = np.load("SA_results/CR" + str(CR) + "/" + str(folder) + "/samples/A_sample_scaled_10000.npy")
    # size [N, d]
    B = np.load("SA_results/CR" + str(CR) + "/" + str(folder) + "/samples/B_sample_scaled_10000.npy")
    # size [d, N, d]
    C = np.load("SA_results/CR" + str(CR) + "/" + str(folder) + "/samples/C_sample_scaled_10000.npy")
    # size [d, N, d]
    D = np.load("SA_results/CR" + str(CR) + "/" + str(folder) + "/samples/D_sample_scaled_10000.npy")
    # get number of samples (N=1000), and number of uncertain parameters (11)
    # N, d = np.shape(A)
    N = 6000
    d = 11

    # number of iterations (i.e. samples)
    N_iterate = np.arange(N)

    # list of size [N, 3]
    print("cpu count = ", cpu_count())

    # # list of size [N, 3]
    # YA = pool.map(partial(run_chain_models_AB_samples, ACE_longitude_=ACE_longitude,
    #                       ACE_latitude_=ACE_latitude, ACE_r_=ACE_r, ACE_vr_=ACE_vr,
    #                       gong_map_=gong_map, A_val=A,
    #                       QoI_="ALL", sample_id_="A",
    #                       CR_=CR, folder_=folder), N_iterate, chuncksize=int(N/(cpu_count()-10)))
    #
    # # list of size [N, 3]
    # YB = pool.map(partial(run_chain_models_AB_samples, ACE_longitude_=ACE_longitude, ACE_latitude_=ACE_latitude,
    #                       ACE_r_=ACE_r, ACE_vr_=ACE_vr, gong_map_=gong_map, A_val=B, QoI_="ALL", sample_id_="B",
    #                       CR_=CR, folder_=folder), N_iterate, chuncksize=int(N/(cpu_count()-10)))
    cpu_count = 40
    with Pool(cpu_count) as pool:
        # list of size [N, d, 3]
        YC = pool.map(partial(run_chain_models_CD_samples, d_val=d, ACE_longitude_=ACE_longitude,
                              ACE_latitude_=ACE_latitude, ACE_r_=ACE_r, ACE_vr_=ACE_vr, gong_map_=gong_map,
                              C_val=C, QoI_="ALL", sample_id_="C", CR_=CR,
                              folder_=folder), N_iterate, chunksize=int(N / cpu_count))
    # close process when done.
    pool.close()
    # # list of size [N, d, 3]
    # used to compute second order indicies.
    # YD = pool.map(partial(run_chain_models_CD_samples, d_val=d, ACE_longitude_=ACE_longitude,
    #                       ACE_latitude_=ACE_latitude, ACE_r_=ACE_r, ACE_vr_=ACE_vr, gong_map_=gong_map,
    #                       C_val=D, QoI_="ALL"), N_iterate)

    # # convert from list to array
    # YA = np.array(YA)
    # YB = np.array(YB)
    YC = np.array(YC)
    # YD = np.array(YD)

    # # save simulation_output results after full simulation_output.
    # np.save(file="SA_results/CR" + str(CR) + "/" + str(folder) + "/simulation_RMSE/YA", arr=YA[:, 0])
    # np.save(file="SA_results/CR" + str(CR) + "/" + str(folder) + "/simulation_MAE/YA", arr=YA[:, 1])
    # np.save(file="SA_results/CR" + str(CR) + "/" + str(folder) + "/simulation_PCC/YA", arr=YA[:, 2])
    #
    # np.save(file="SA_results/CR" + str(CR) + "/" + str(folder) + "/simulation_RMSE/YB", arr=YB[:, 0])
    # np.save(file="SA_results/CR" + str(CR) + "/" + str(folder) + "/simulation_MAE/YB", arr=YB[:, 1])
    # np.save(file="SA_results/CR" + str(CR) + "/" + str(folder) + "/simulation_PCC/YB", arr=YB[:, 2])

    # np.save(file="SA_results/CR" + str(CR) + "/" + str(folder) + "/simulation_RMSE/YC_multi", arr=YC[:, :, 0])
    # np.save(file="SA_results/CR" + str(CR) + "/" + str(folder) + "/simulation_MAE/YC_multi", arr=YC[:, :, 1])
    # np.save(file="SA_results/CR" + str(CR) + "/" + str(folder) + "/simulation_PCC/YC_multi", arr=YC[:, :, 2])

    # np.save(file="SA_results/CR" + str(CR) + "/B/simulation_RMSE/YD", arr=YD[:, :, 0])
    # np.save(file="SA_results/CR" + str(CR) + "/B/simulation_MAE/YD", arr=YD[:, :, 1])
    # np.save(file="SA_results/CR" + str(CR) + "/B/simulation_PCC/YD", arr=YD[:, :, 2])
