# Import useful functions
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, interpolate
from scipy.integrate import trapezoid, cumulative_trapezoid
from scipy.optimize import newton
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from pathlib import Path
import subprocess
import json


def Fused_Hugoniot_us_up_0(up):
    """
    Gives the relation between Us and Up for the Fused Silica
    """
    num = len(up)
    A = np.random.normal(4.925, 0.094, num)
    B = np.random.normal(1.234, 0.005, num)
    C = np.random.normal(2.074, 0.084, num)
    D = np.random.normal(0.352, 0.011, num)
    us = A + B * up - C * up * np.exp(-D * up)
    return us


def init_Hugoniot_functions(function_name, Us_limits, force_regenerate=False):
    """
    Create the Hugoniot function for the fused silica based on existing data
    """
    if not function_name in globals():
        raise NameError("function " + function_name + " not found")
    function = globals()[function_name]
    my_file = Path(function_name + ".txt")
    if force_regenerate or not my_file.is_file():
        Up = np.random.uniform(0, 50, 1000000)
        x = function(Up)
        y = Up
        nbins = 100
        n, _ = np.histogram(x, bins=nbins, range=Us_limits)
        sy, _ = np.histogram(x, bins=nbins, weights=y, range=Us_limits)
        sy2, center = np.histogram(x, bins=nbins, weights=y * y, range=Us_limits)
        Hugoniot_Us = sy / n
        Hugoniot_Us_filt = savgol_filter(Hugoniot_Us, 51, 2)
        Hugoniot_Us_std = np.sqrt(sy2 / n - Hugoniot_Us * Hugoniot_Us)
        Hugoniot_Us_std_filt = savgol_filter(Hugoniot_Us_std, 51, 2)
        Hugoniot_Up = (center[1:] + center[:-1]) / 2
        allvec = np.vstack((Hugoniot_Up, Hugoniot_Us_filt, Hugoniot_Us_std_filt)).T
        np.savetxt(my_file, allvec, header="Hugoniot_Up,Hugoniot_Us_filt,Hugoniot_Us_std_filt")
    else:
        (Hugoniot_Up, Hugoniot_Us_filt, Hugoniot_Us_std_filt) = np.loadtxt(my_file).T
    f1 = interpolate.interp1d(Hugoniot_Up, Hugoniot_Us_filt, fill_value="extrapolate")
    f2 = interpolate.interp1d(Hugoniot_Up, Hugoniot_Us_std_filt, fill_value="extrapolate")
    return f1, f2


def sesame_hugoniot(material, initCompression=1.0, initTemperature=300, initP=1e-3, endP=10, points=50, fileout=None):
    """
    Open SESAME Hugoniot curve
    """

    mydir = f"sesame-{material:04d}"
    if not fileout:
        fileout = f"hug_{initCompression}_{initTemperature}_{initP}_{endP}_{points}.txt"
    arr = np.loadtxt(mydir + "/" + fileout).T
    r, P, E, T, Us, Up = arr
    return r, P * 100, T * 11604.5, Us, Up


def hug_plus_release_reshock(material):
    """
    Extract release/reshock curves from json files
    """
    json_filename = f"release_reshock{material}.json"
    with open(json_filename, "r") as file:
        alldata = json.load(file)
    release = {float(k): alldata["release"][k] for k in alldata["release"]}
    reshock = {float(k): alldata["reshock"][k] for k in alldata["reshock"]}
    return release, reshock


def not_outlier(res, thresh=3.5):
    """
    Select points in res with a Z-score below thresh (threshold, default 3.5)
    """
    p0 = np.array(res["rho2"])
    median = np.median(p0, axis=0)
    diff = (p0 - median) ** 2
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)
    modified_z_score = 0.6745 * diff / med_abs_deviation
    filtered = modified_z_score < thresh
    print(f"keeping {100*sum(filtered)/len(p0):.1f}% of data")
    for i in res:
        res[i] = np.array(res[i])[filtered]
    return res


def desadapt_fused(second_hug, Up_from_Us, Up_from_Us_err, us0, us1, rho0, rho1, doPlot=False):
    """
    Compute the average reshock or release curve from abaque curves from Up0 point (computed from us0)
    Return the intersection (Up_cross, P_cross, Rho_cross) with the Rayleigh curve with slope rho1*us1

    second_hug : Array containing release/reshock curves
    us0 : Wave speed in material 0 (The know material, ex : fused silica) in km/s
    us1 : Wave speed in material 1 (The sought material, ex : a liquid) in km/s
    rho0 : Material 0 initial density in gm/cm³
    rho1 : Material 1 initial density in gm/cm³
    """

    up0 = Up_from_Us(us0)
    up0_err = Up_from_Us_err(us0)
    if doPlot:
        up_random = up0
    else:
        up_random = np.random.normal(up0, up0_err, 1)[0]
    p0 = rho0 * us0 * up_random

    if rho0 * us0 > rho1 * us1:
        secondary_hugoniot = second_hug[0]  # releases
    else:
        secondary_hugoniot = second_hug[1]  # reshock

    release_starting_p = sorted(secondary_hugoniot.keys())

    # print(release_starting_p)
    index = np.flatnonzero(release_starting_p < p0)[-1]
    alpha = (release_starting_p[index + 1] - p0) / (release_starting_p[index + 1] - release_starting_p[index])
    state1 = secondary_hugoniot[release_starting_p[index]]
    state2 = secondary_hugoniot[release_starting_p[index + 1]]

    interp1 = interpolate.interp1d(state1[4], state1[1], fill_value="extrapolate")
    interp2 = interpolate.interp1d(state2[4], state2[1], fill_value="extrapolate")
    release_avg = lambda x: alpha * interp1(x) + (1 - alpha) * interp2(x)
    rayleigh = lambda x: rho1 * us1 * x

    # print(interp1, interp2)

    Up_at_crossing = newton(lambda x: release_avg(x) - rayleigh(x), x0=0)
    P_at_crossing = rayleigh(Up_at_crossing)
    Rho_at_crossing = rho1 * us1 / (us1 - Up_at_crossing)

    if doPlot:
        print(
            f"us0={us0:7.3f} up0={up0:7.3f} p0={p0:8.3f} i{index:4d} [{release_starting_p[index]:8.3f}  {release_starting_p[index+1]:8.3f}] {alpha:7.3f}"
        )
        plt.errorbar([up_random], [p0], xerr=[up0_err], marker="o", color="#00a933", label="$z_{A1}$")
        # plt.plot(up_random+state1[4],state1[1], ".")
        # plt.plot(up_random+state2[4],state2[1], ".")
        # plt.plot(up_grid, interp1(up_grid))
        # plt.plot(up_grid, interp2(up_grid))
        if rho0 * us0 > rho1 * us1:
            up_grid = np.linspace(0, 2 * up0, 1000, endpoint=True)
        else:
            up_grid = np.linspace(0, up_random, endpoint=True)
        plt.plot(up_grid, release_avg(up_grid), color="deepskyblue", linestyle="--", label="A Release curve")
        plt.plot(
            np.linspace(0, 2 * up0, 1000, endpoint=True),
            rayleigh(np.linspace(0, 2 * up0, 1000, endpoint=True)),
            label="B Rayleigh curve",
            color="y",
        )
        # plt.plot([Up_at_crossing], [P_at_crossing], "o", color="purple", label="$z_{A2} ; z_{B1}$")
        # plt.legend(loc="center left", frameon=False)
    return Up_at_crossing, P_at_crossing, Rho_at_crossing


def montecarlo(
    secondary_hugoniot,
    Up_from_Us,
    Up_from_Us_err,
    us0,
    us1,
    rho0,
    rho1,
    us0_err,
    us1_err,
    rho0_err,
    rho1_err,
    NM,
    doPlot=False,
):
    """
    Create a list of points plausible in the error bar of rho0, us0, rho1, us1 and compute the Up_cross, P_cross, Rho_cross
    Return only the point compatible with the z-score made with not_outlier(res, thresh=3.5)

    secondary_hugoniot : Array containing release/reshock curves
    us0 : Wave speed in material 0 (The know material, ex : fused silica) in km/s
    us1 : Wave speed in material 1 (The sought material, ex : a liquid) in km/s
    rho0 : Material 0 initial density in gm/cm³
    rho1 : Material 1 initial density in gm/cm³
    us0_err : Error on us0 measured velocity in km/s
    us1_err : Error on us1 measured velocity in km/s
    rho0_err : Error on material 0 initial density in gm/cm³
    rho1_err : Error on material 1 initial density in gm/cm³
    NM : Number of points considered in the Monte-Carlo
    """
    desadapt_fused(secondary_hugoniot, Up_from_Us, Up_from_Us_err, us0, us1, rho0, rho1, doPlot)

    # Create list of densities and speed velocities in accordance with the error bar
    rho0_vec = np.random.normal(rho0, rho0_err, NM)
    us0_vec = np.random.normal(us0, us0_err, NM)
    rho1_vec = np.random.normal(rho1, rho1_err, NM)
    us1_vec = np.random.normal(us1, us1_err, NM)

    up2_vec = []
    p2_vec = []
    rho2_vec = []

    for i in range(NM):
        up, p, rho = desadapt_fused(
            second_hug=secondary_hugoniot,
            Up_from_Us=Up_from_Us,
            Up_from_Us_err=Up_from_Us_err,
            us0=us0_vec[i],
            us1=us1_vec[i],
            rho0=rho0_vec[i],
            rho1=rho1_vec[i],
        )
        # if rho>0 and rho<max_rho:
        up2_vec.append(up)
        p2_vec.append(p)
        rho2_vec.append(rho)

    if doPlot:
        plt.plot(up2_vec, p2_vec, ".", alpha=1, markersize=1, color="cyan", label="Monte-Carlo")
        plt.errorbar(
            np.mean(up2_vec),
            np.mean(p2_vec),
            xerr=np.std(up2_vec),
            yerr=np.std(p2_vec),
            color="#bf0041",
            label="$z_{A2} ; z_{B1}$",
        )
        plt.legend(loc="center left", frameon=False)
    res = {}
    res["up2"] = up2_vec
    res["p2"] = p2_vec
    res["rho2"] = rho2_vec

    return not_outlier(res)


def resume(
    secondary_hugoniot,
    Up_from_Us,
    Up_from_Us_err,
    us0,
    us1,
    rho0,
    rho1,
    us0_err=0.1,
    us1_err=0.1,
    rho0_err=0.01,
    rho1_err=0.01,
    NM=1000,
    doPlot=True,
):
    """
    Make a Monte-carlo estimation of the values possible at the crossing between the Rayleigh curve at the material 1 and the reshock/release from material 0
    Print values and errors
    Return values and errors

    secondary_hugoniot : Array containing release/reshock curves
    us0 : Wave speed in material 0 (The know material, ex : fused silica) in km/s
    us1 : Wave speed in material 1 (The seeked material, ex : a liquid) in km/s
    rho0 : Material 0 initial density in gm/cm³
    rho1 : Material 1 initial density in gm/cm³
    us0_err : Error on us0 measured velocity in km/s
    us1_err : Error on us1 measured velocity in km/s
    rho0_err : Error on material 0 initial density in gm/cm³
    rho1_err : Error on material 1 initial density in gm/cm³
    NM : Number of points considered in the Monte-Carlo
    """
    res = montecarlo(
        secondary_hugoniot=secondary_hugoniot,
        Up_from_Us=Up_from_Us,
        Up_from_Us_err=Up_from_Us_err,
        us0=us0,
        us1=us1,
        rho0=rho0,
        rho1=rho1,
        us0_err=us0_err,
        us1_err=us1_err,
        rho0_err=rho0_err,
        rho1_err=rho1_err,
        NM=NM,
        doPlot=doPlot,
    )
    fmt = "7.3f"
    print(f'{"us0":4s} = {us0:{fmt}} +- {us0_err:<{fmt}}')
    print(f'{"us1":4s} = {us1:{fmt}} +- {us1_err:<{fmt}}')
    res2 = {}
    for x in res:
        x_mean, x_err = np.mean(res[x]), np.std(res[x])
        res2[x] = (x_mean, x_err)
        print(f"{x:4s} = {x_mean:{fmt}} +- {x_err:<{fmt}}")
    return res, res2


def plot_velocity(file_neutrino, block=None, list_times=None):

    if block is None:
        with open(file_neutrino, "r") as f:
            rawvisars = f.read().strip().split("\n\n")
            data_VISAR_1 = []
            data_VISAR_2 = []
            for i in range(2):
                raw_visar = rawvisars[i]
                lines = raw_visar.splitlines()
                for line in lines:
                    if not line.startswith("#"):
                        data_values = list(map(float, line.strip().split()))
                        if len(data_values):
                            if i == 0:
                                data_VISAR_1.append(data_values)
                            else:
                                data_VISAR_2.append(data_values)

        data_VISAR_1 = np.array(data_VISAR_1).T
        data_VISAR_2 = np.array(data_VISAR_2).T
        prefered_visar = 2  # 1 : VISAR 1, 2 : VISAR 2 - Which VISAR to take as a time reference

        # Interpolate velocities from one visar to the other to be able to make the mean
        if prefered_visar == 1:
            velocity_int = np.interp(data_VISAR_1[0], data_VISAR_2[0], data_VISAR_2[1])
            vel_error_int = np.interp(data_VISAR_1[0], data_VISAR_2[0], data_VISAR_2[2])
            time_vel, velocity, vel_error = (
                data_VISAR_1[0],
                np.mean(data_VISAR_1[1] + velocity_int),
                1 / 2 * np.sqrt(np.square(data_VISAR_1[2]) + np.square(vel_error_int)),
            )

        else:
            velocity_int = np.interp(data_VISAR_2[0], data_VISAR_1[0], data_VISAR_1[1])
            vel_error_int = np.interp(data_VISAR_2[0], data_VISAR_1[0], data_VISAR_1[2])
            time_vel, velocity, vel_error = (
                data_VISAR_2[0],
                1 / 2 * (data_VISAR_2[1] + velocity_int),
                1 / 2 * np.sqrt(np.square(data_VISAR_2[2]) + np.square(vel_error_int)),
            )

    else:
        with open(file_neutrino, "r") as f:
            rawvisars = f.read().strip().split("\n\n")
            raw_visar = rawvisars[block - 1]
            data = []
            lines = raw_visar.splitlines()
            for line in lines:
                if not line.startswith("#"):
                    data_values = list(map(float, line.strip().split()))
                    if len(data_values):
                        data.append(data_values)

        data = np.array(data).T
        time_vel, velocity, vel_error = data[0], data[1], data[2]

    plt.ion()
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    if block is None:
        ax.errorbar(data_VISAR_1[0], data_VISAR_1[1], data_VISAR_1[2], color="darkred", label="VISAR 1")
        ax.errorbar(data_VISAR_2[0], data_VISAR_2[1], data_VISAR_2[2], color="darkgreen", label="VISAR 2")
        ax.errorbar(time_vel, velocity, vel_error, color="red", label="Mean velocity")
    else:
        ax.errorbar(time_vel, velocity, vel_error, color="red")

    fig.tight_layout(h_pad=6)
    # ax.grid()
    ax.set_xlabel(f"Time [s]")
    ax.set_ylabel(f"Velocity $U_S$ [$km.s^{-1}$]")
    ax.errorbar(time_vel, velocity, vel_error, color="red")
    # plt.legend(frameon=False)
    if list_times is not None:
        # Plotting the selected zones for saving
        ax.axvline(x=list_times[0], linestyle="--", color="blue")
        ax.axvline(x=list_times[1], linestyle="--", color="blue")

        # Finding indices associated with these time limit
        idx_min = np.searchsorted(time_vel, list_times[0])
        idx_max = np.searchsorted(time_vel, list_times[1])

        Us = velocity[idx_min:idx_max]
        Us_err = vel_error[idx_min:idx_max]
        return Us, Us_err


def update_file_hugoniot(
    file_path,
    tir_number,
    Us_standard,
    Us_studied,
    Up,
    P,
    rho,
    us_standard_err=0.5,
    us_studied_err=0.5,
    up_err=0.5,
    p_err=10,
    rho_err=0.1,
):
    # Lire le contenu du fichier
    try:
        with open(file_path, "r") as file:
            lines = file.readlines()
    except FileNotFoundError:
        lines = ["Shot\tUs_Standard\tUs_studied\tUp\tP\trho\tUs_err\tUp_err\tP_err\trho_err\n"]

    # Vérifier si le numéro de tir existe déjà
    found = False
    for i, line in enumerate(lines):
        if line.split("\t")[0] == str(tir_number):
            # Mettre à jour les valeurs existantes
            lines[i] = (
                f"{tir_number}\t{Us_standard}\t{Us_studied}\t{Up}\t{P}\t{rho}\t{us_standard_err}\t{us_studied_err}\t{up_err}\t{p_err}\t{rho_err}\n"
            )
            found = True
            break

    # Si le numéro de tir n'existe pas, ajouter une nouvelle ligne
    if not found:
        lines.append(
            f"{tir_number}\t{Us_standard}\t{Us_studied}\t{Up}\t{P}\t{rho}\t{us_standard_err}\t{us_studied_err}\t{up_err}\t{p_err}\t{rho_err}\n"
        )

    # Enregistrer les modifications dans le fichier
    with open(file_path, "w") as file:
        file.writelines(lines)


def save_full_release(
    file_path, tir_number, Us_standard, Us_studied, Up, P, rho, us_err, up_err, p_err, rho_err, save=False
):
    if save:
        with open(file_path, "w") as file:
            data = np.column_stack(
                (
                    tir_number,
                    Us_standard,
                    Us_studied,
                    Up,
                    P,
                    rho,
                    us_err,
                    up_err,
                    p_err,
                    rho_err,
                )
            )
            np.savetxt(
                file,
                data,
                fmt="%e",
                delimiter=" ",
                header=f"Shot\tUs_Standard\tUs_studied\tUp\tP\trho\tUs_err\tUp_err\tP_err\trho_err\n",
            )
