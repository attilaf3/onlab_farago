import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from optimize_two_users import optimize_two_users

# Bemeneti adatok beolvasása
df = pd.read_csv('input_tobb_haztartas.csv', sep=';', index_col=0, parse_dates=True)
df_filtered = df.resample("1h").sum()
na_p_consumed = df_filtered[["consumer1", "consumer2"]].to_numpy()
na_p_pv = df_filtered[["pv1", "pv2"]].to_numpy()
na_p_ut = df_filtered[["thermal_user1", "thermal_user2"]].to_numpy()

# Optimalizálás futtatása
results, status, objective, user_ids, num_vars, num_constraints = optimize_two_users(
    p_pv=na_p_pv,
    p_ut=na_p_ut,
    p_consumed=na_p_consumed,
    size_elh=np.array([2, 2.5]),
    size_bess=20,
    size_hss=np.array([130, 150]),
    run_lp=False,
    objective="environmental"
)

# Eredmények kirajzolása
def plot_user_results(results, user_id):
    time = np.arange(results["p_cl_with"].shape[1])
    fig, axs = plt.subplots(3, 1, figsize=(20, 12), sharex=True)

    axs[0].plot(time, results["p_inj_user"][user_id], label="P_inj_user")
    axs[0].plot(time, results["p_with_user"][user_id], label="P_with_user")
    axs[0].set_title(f"User {user_id+1} - Grid")
    axs[0].legend()
    axs[0].grid()

    axs[1].plot(time, results["p_cl_with"][user_id], label="P_cl_with")
    axs[1].plot(time, results["p_cl_grid"][user_id], label="P_cl_grid")
    axs[1].plot(time, results["p_cl_rec"][user_id], label="P_cl_rec")
    axs[1].plot(time, results["p_elh_in"][user_id], label="P_elh_in")
    axs[1].plot(time, results["p_elh_out"][user_id], label="P_elh_out")
    axs[1].set_title(f"User {user_id+1} - Thermal Loads")
    axs[1].legend()
    axs[1].grid()

    axs[2].plot(time, results["e_hss_stor"][user_id], label="E_hss_stor")
    axs[2].plot(time, results["t_hss"][user_id], label="T_hss (°C)")
    axs[2].set_title(f"User {user_id+1} - Heat Storage")
    axs[2].legend()
    axs[2].grid()

    plt.xlabel("Time (h)")
    plt.tight_layout()
    plt.show()

# Összesített ábra: CSC és BESS
def plot_community(results):
    time = np.arange(results["p_bess_in"].shape[0])
    plt.figure(figsize=(18, 6))

    plt.plot(time, results["p_bess_in"], label="P_bess_in")
    plt.plot(time, results["p_bess_out"], label="P_bess_out")
    plt.plot(time, results["e_bess_stor"], label="E_bess_stor")
    plt.plot(time, results["p_shared"], label="P_shared")
    plt.plot(time, results["p_grid_in"], label="P_grid_in")
    plt.plot(time, results["p_grid_out"], label="P_grid_out")

    plt.title("Community Battery and Shared Power")
    plt.xlabel("Time (h)")
    plt.ylabel("Power / Energy (kW / kWh)")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

# Kirajzolás
plot_user_results(results, user_id=0)
plot_user_results(results, user_id=1)
plot_community(results)

def display_figures_two_users(results, p_pv, p_ut):
    t0s = [0, 2184, 4368, 6552]  # szezonkezdetek (óra)
    dt = 168  # egy hét
    titles = ['Winter', 'Spring', 'Summer', 'Autumn']
    path_effect = lambda lw: [pe.Stroke(linewidth=1.5 * lw, foreground='w'), pe.Normal()]
    bar_kw = dict(width=0.8)
    plot_kw = dict(lw=3, path_effects=path_effect(3))
    fontsize = 13

    for i, t0 in enumerate(t0s):
        tf = t0 + dt
        time = np.arange(t0, tf)

        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(20, 12), sharex=True)

        # --- User 1: elektromos + hő
        ax = axes[0]
        bottom = np.zeros_like(time, dtype=float)
        ax.bar(time, p_pv[t0:tf, 0], bottom=bottom, label='PV1', **bar_kw)
        bottom += p_pv[t0:tf, 0]
        ax.bar(time, results['p_cl_rec'][0][t0:tf], bottom=bottom, label='CL_rec1', **bar_kw)
        bottom += results['p_cl_rec'][0][t0:tf]
        ax.bar(time, results['p_cl_grid'][0][t0:tf], bottom=bottom, label='CL_grid1', **bar_kw)

        bottom = np.zeros_like(time)
        ax.bar(time, -results['p_with_user'][0][t0:tf], bottom=bottom, label='With1', **bar_kw)
        bottom -= results['p_with_user'][0][t0:tf]
        ax.bar(time, -results['p_elh_out'][0][t0:tf], bottom=bottom, label='ELH_out1', **bar_kw)
        ax.bar(time, -p_ut[t0:tf, 0], bottom=bottom, label='ThermalNeed1', **bar_kw)

        ax.set_title("User 1", fontsize=fontsize)
        ax.grid()
        ax.legend(fontsize=fontsize)
        ax.set_ylabel("Power (kW)", fontsize=fontsize)

        # --- User 2: elektromos + hő
        ax = axes[1]
        bottom = np.zeros_like(time)
        ax.bar(time, p_pv[t0:tf, 1], bottom=bottom, label='PV2', **bar_kw)
        bottom += p_pv[t0:tf, 1]
        ax.bar(time, results['p_cl_rec'][1][t0:tf], bottom=bottom, label='CL_rec2', **bar_kw)
        bottom += results['p_cl_rec'][1][t0:tf]
        ax.bar(time, results['p_cl_grid'][1][t0:tf], bottom=bottom, label='CL_grid2', **bar_kw)

        bottom = np.zeros_like(time)
        ax.bar(time, -results['p_with_user'][1][t0:tf], bottom=bottom, label='With2', **bar_kw)
        bottom -= results['p_with_user'][1][t0:tf]
        ax.bar(time, -results['p_elh_out'][1][t0:tf], bottom=bottom, label='ELH_out2', **bar_kw)
        ax.bar(time, -p_ut[t0:tf, 1], bottom=bottom, label='ThermalNeed2', **bar_kw)

        ax.set_title("User 2", fontsize=fontsize)
        ax.grid()
        ax.legend(fontsize=fontsize)
        ax.set_ylabel("Power (kW)", fontsize=fontsize)

        # --- Community CSC rész: grid, BESS, shared
        ax = axes[2]
        p_inj = results["p_inj"][t0:tf]
        p_with = results["p_with"][t0:tf]
        p_shared = results["p_shared"][t0:tf]
        t_plot = np.linspace(t0, tf, 1000)
        f_plot = lambda x: np.interp(t_plot, time, x)
        inj_plot = f_plot(p_inj)
        with_plot = f_plot(p_with)
        shared_plot = np.minimum(inj_plot, with_plot)

        ax.plot(time, p_inj, label='Injection', color='tab:red', **plot_kw)
        ax.plot(time, p_with, label='Withdrawal', color='tab:blue', **plot_kw)
        ax.plot(time, p_shared, label='Shared', color='tab:green', marker='s', linestyle='', **plot_kw)

        ax.fill_between(t_plot, shared_plot, with_plot, where=with_plot > shared_plot,
                        label='From Grid', color='tab:blue', alpha=0.5)
        ax.fill_between(t_plot, shared_plot, inj_plot, where=inj_plot > shared_plot,
                        label='To Grid', color='tab:red', alpha=0.5)
        ax.fill_between(t_plot, 0, shared_plot, where=shared_plot > 0,
                        label='Shared Energy', color='tab:green', alpha=0.4)

        ax.set_title("Community (CSC + BESS + Grid)", fontsize=fontsize)
        ax.grid()
        ax.legend(fontsize=fontsize)
        ax.set_xlabel("Time (h)", fontsize=fontsize)
        ax.set_ylabel("Power (kW)", fontsize=fontsize)

        plt.suptitle(f"{titles[i]} Season Overview", fontsize=fontsize + 3)
        plt.tight_layout()
        plt.show()