import pandas as pd
from matplotlib import pyplot as plt, patheffects as pe
import numpy as np
from test2 import optimize
import seaborn as sns

# Ctrl+Shift+Alt+L: code cleanup: hosszú sorok tördelése+importok optimalizálása


df = pd.read_csv('input.csv', sep=';', index_col=0, parse_dates=True)
# df_filtered = df[~df.index.astype(str).str.contains(":15|:30|:45", regex=True)]  # df.resample("1h").first()
df_filtered = df.resample("1h").sum()
na_values = df_filtered.values  # numpy array


def display_figures(p_pv, p_bess_out, p_with, p_ue, p_bess_in, p_inj, e_bess_stor, p_elh_out,
                    p_ut, p_shared, p_cl_rec, p_cl_grid, p_cl_with, e_hss_stor, p_hss_out, p_hss_in, t_hss, d_cl,
                    p_grid_in, p_grid_out, diff_hss_out=None, diff_hss_in=None, params=None):
    global time, t_plot
    # One week in each season
    figsize = (20, 15)
    fontsize = 15
    t0s = [0, 2184, 4368, 6552]
    dt = 72
    titles = ['Winter', 'Spring', 'Summer', 'Autumn']
    path_effect = lambda lw: [pe.Stroke(linewidth=1.5 * lw, foreground='w'), pe.Normal()]
    bar_kw = dict(width=0.8, )
    plot_kw = dict(lw=3, path_effects=path_effect(3))
    area_kw = dict(alpha=0.6)
    for i, t0 in enumerate(t0s):
        # Useful time variables
        tf = t0 + dt
        time = np.arange(t0, tf)

        # Electric node of the condominium
        fig, axes = plt.subplots(ncols=2, figsize=(16, 6), gridspec_kw=dict(width_ratios=[0.8, 0.2]))
        ax = axes[0]
        legend_ax = axes[1]
        # ax.set_xlim(t0, tf - 1)
        # Plot "positive" half
        bottom = np.zeros_like(time, dtype=float)
        ax.bar(time, p_pv[t0:tf], bottom=bottom, label=r'$P_\mathrm{pv}$', **bar_kw)
        bottom += p_pv[t0:tf]
        ax.bar(time, p_bess_out[t0:tf], bottom=bottom, label=r'$P_\mathrm{bess,out}$', **bar_kw)
        bottom += p_bess_out[t0:tf]
        ax.bar(time, p_grid_out[t0:tf], bottom=bottom, label=r'$P_\mathrm{grid,out}$', **bar_kw)

        # Plot "negative" half
        bottom = np.zeros_like(time, dtype=float)
        ax.bar(time, -p_ue[t0:tf], bottom=bottom, label=r'$P_\mathrm{ue}$', **bar_kw)
        bottom -= p_ue[t0:tf]
        ax.bar(time, -p_bess_in[t0:tf], bottom=bottom, label=r'$P_\mathrm{bess,in}$', **bar_kw)
        bottom -= p_bess_in[t0:tf]
        ax.bar(time, -p_cl_grid[t0:tf], bottom=bottom, label=r'$P_\mathrm{cl,grid}$', **bar_kw)
        bottom -= p_cl_grid[t0:tf]
        ax.bar(time, -p_cl_rec[t0:tf], bottom=bottom, label=r'$P_\mathrm{cl,rec}$', **bar_kw)
        bottom -= p_cl_rec[t0:tf]
        ax.bar(time, -p_grid_in[t0:tf], bottom=bottom, label=r'$P_\mathrm{grid,in}$', **bar_kw)

        # Plot storage SOC
        axtw = ax.twinx()
        axtw.plot(time, e_bess_stor[t0:tf], color='black', ls='--', label=r"$\mathrm{E_{stor,bess}}$")

        # Adjust and show
        ax.set_xlabel("Time (h)", fontsize=fontsize)
        ax.set_ylabel("Power (kW)", fontsize=fontsize)
        ax.set_title(f"{titles[i]} – Electric hub", fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)
        ax.grid()

        axtw.set_ylabel("Stored energy (kWh)")
        axtw.spines['right'].set_color('black')
        axtw.tick_params(axis='y', colors='black')
        axtw.yaxis.label.set_color('black')

        # Legend
        handles1, labels1 = ax.get_legend_handles_labels()
        handles2, labels2 = axtw.get_legend_handles_labels()
        handles3, labels3 = axtd.get_legend_handles_labels() if 'axtd' in locals() else ([], [])  # ha van vezérlés

        legend_ax.legend(handles1 + handles2 + handles3, labels1 + labels2 + labels3, loc='center', fontsize=fontsize,
                         ncol=2)
        legend_ax.axis('off')
        plt.tight_layout()
        plt.show()

        # Thermal node
        fig, axes = plt.subplots(ncols=2, figsize=(16, 6), gridspec_kw=dict(width_ratios=[0.8, 0.2]))
        ax = axes[0]
        legend_ax = axes[1]
        # ax.set_xlim(t0, tf - 1)
        # # Plot "positive" half
        bottom = np.zeros_like(time, dtype=float)

        ax.bar(time, p_cl_grid[t0:tf], bottom=bottom, label=r'$P_\mathrm{cl,grid}$', **bar_kw)
        bottom += p_cl_grid[t0:tf]
        ax.bar(time, p_cl_rec[t0:tf], bottom=bottom, label=r'$P_\mathrm{cl,rec}$', **bar_kw)
        bottom += p_cl_rec[t0:tf]
        ax.bar(time, p_hss_out[t0:tf], bottom=bottom, label=r'$P_\mathrm{hss,out}$', **bar_kw)
        bottom += p_hss_out[t0:tf]
        # if diff_hss_out is not None:
        #     ax.bar(time, diff_hss_out[t0:tf], bottom=bottom, label=r'$\Delta P_\mathrm{hss,out}$', color="orangered",
        #            **bar_kw)
        #     bottom += diff_hss_out[t0:tf]

        # # Plot "negative" half
        bottom = np.zeros_like(time, dtype=float)
        # if diff_hss_in is not None:
        #     ax.bar(time, diff_hss_in[t0:tf], bottom=bottom, label=r'$\Delta P_\mathrm{hss,in}$', color="cyan",
        #            **bar_kw)
        #     bottom += diff_hss_in[t0:tf]
        if p_elh_out is not None:
            ax.bar(time, -p_elh_out[t0:tf], bottom=bottom, label=r'$P_\mathrm{elh,out}$', **bar_kw)
            bottom -= p_elh_out[t0:tf]
        if p_ut is not None:
            ax.bar(time, -p_ut[t0:tf], bottom=bottom, label=r'$P_\mathrm{ut}$', **bar_kw)
            bottom -= p_ut[t0:tf]

        # # Plot storage SOC
        axtw = ax.twinx()
        axtw.plot(time, e_hss_stor[t0:tf], color='black', ls='--', label=r"$\mathrm{E_{stor,hss}}$")

        axtw.set_ylabel("Stored energy (kWh)")
        axtw.spines['right'].set_color('black')
        axtw.tick_params(axis='y', colors='black')
        axtw.yaxis.label.set_color('black')
        axtw.legend().set_visible(False)

        # # Plot control signal
        axtd = ax.twinx()
        axtd.plot(time, d_cl[t0:tf], '.', ls='-', color='lightgrey', label=r"$\mathrm{control\ signal\ (on/off)}$")

        axtd.spines["top"].set_visible(False)
        axtd.spines["right"].set_visible(False)
        axtd.spines["left"].set_visible(False)
        axtd.spines["bottom"].set_visible(False)
        axtd.tick_params(axis="both", which='both', length=0, labelcolor="none")
        axtd.legend().set_visible(False)

        # # Adjust and show
        ax.set_xlabel("Time (h)", fontsize=fontsize)
        ax.set_ylabel("Power (kW)", fontsize=fontsize)
        ax.set_title(f"{titles[i]} – Thermal node", fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)
        ax.grid()

        # # Legend
        handles1, labels1 = ax.get_legend_handles_labels()
        handles2, labels2 = axtw.get_legend_handles_labels() if 'axtw' in locals() else ([], [])
        handles3, labels3 = axtd.get_legend_handles_labels() if 'axtd' in locals() else ([], [])

        legend_ax.legend(handles1 + handles2 + handles3, labels1 + labels2 + labels3, loc='center', fontsize=fontsize,
                         ncol=2)
        legend_ax.axis('off')
        plt.tight_layout()
        plt.show()

        # CSC
        fig, axes = plt.subplots(ncols=2, figsize=(16, 6), gridspec_kw=dict(width_ratios=[0.8, 0.2]))
        ax = axes[0]
        legend_ax = axes[1]
        # ax.set_xlim(t0, tf - 1)
        # Interpolate for graphical purposes
        t_plot = np.linspace(t0, tf, 1000)
        f_plot = lambda x: np.interp(t_plot, time, x)
        p_inj_plot = f_plot(p_inj[t0:tf])
        p_with_plot = f_plot(p_with[t0:tf])
        p_shared_plot = np.minimum(p_inj_plot, p_with_plot)

        # Plot
        # ax.plot(time, p_inj[t0:tf], label=r'$P_\mathrm{inj}$', color='tab:red', **plot_kw)
        # ax.plot(time, p_with[t0:tf], label=r'$P_\mathrm{with}$', color='tab:blue', **plot_kw)
        # ax.plot(time, p_shared[t0:tf], label=r'$P_\mathrm{shared}$', color='tab:green', ls='', marker='s', **plot_kw)

        # Sima görbék vonalként, nem marker
        ax.plot(t_plot, p_inj_plot, label=r'$P_\mathrm{inj}$', color='tab:red', **plot_kw)
        ax.plot(t_plot, p_with_plot, label=r'$P_\mathrm{with}$', color='tab:blue', **plot_kw)
        ax.plot(t_plot, p_shared_plot, label=r'$P_\mathrm{shared}$', color='tab:green', **plot_kw)

        # ax.fill_between(t_plot, p_shared_plot, p_with_plot, where=p_with_plot > p_shared_plot,
        #                 label=r'E$_\mathrm{\leftarrow grid}$', color='tab:blue', **area_kw)
        # ax.fill_between(t_plot, p_shared_plot, p_inj_plot, where=p_inj_plot > p_shared_plot,
        #                 label=r'E$_\mathrm{\rightarrow grid}$', color='tab:red', **area_kw)
        # ax.fill_between(t_plot, 0, p_shared_plot, where=p_shared_plot > 0, label=r'E$_\mathrm{shared}$',
        #                 color='tab:green')
        ax.fill_between(t_plot, p_shared_plot, p_with_plot, where=p_with_plot > p_shared_plot,
                        label=r'E$_\mathrm{\leftarrow grid}$', color='tab:blue', alpha=0.3)
        ax.fill_between(t_plot, p_shared_plot, p_inj_plot, where=p_inj_plot > p_shared_plot,
                        label=r'E$_\mathrm{\rightarrow grid}$', color='tab:red', alpha=0.3)
        ax.fill_between(t_plot, 0, p_shared_plot, where=p_shared_plot > 0,
                        label=r'E$_\mathrm{shared}$', color='tab:green', alpha=0.3)

        # Adjust and show
        ax.set_xlabel("Time (h)", fontsize=fontsize)
        ax.set_ylabel("Power (kW)", fontsize=fontsize)
        ax.set_title(f"{titles[i]} – CSC", fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)
        ax.grid()

        # Legend
        handles1, labels1 = ax.get_legend_handles_labels()
        handles2, labels2 = axtw.get_legend_handles_labels() if 'axtw' in locals() else ([], [])
        handles3, labels3 = axtd.get_legend_handles_labels() if 'axtd' in locals() else ([], [])

        legend_ax.legend(handles1 + handles2 + handles3, labels1 + labels2 + labels3, loc='center', fontsize=fontsize,
                         ncol=2)
        legend_ax.axis('off')
        plt.tight_layout()
        plt.show()

def extract_results_and_show(results):
    # Get results
    p_inj = results.get('p_inj')
    p_with = results.get('p_with')
    p_bess_in = results.get('p_bess_in')
    p_bess_out = results.get('p_bess_out')
    e_bess_stor = results.get('e_bess_stor')
    p_elh_in = results.get('p_elh_in')
    # p_elh_out = results.get('p_elh_out')  # ezt kikommentezheted
    p_elh_out = results.get('p_cl_grid') + results.get('p_cl_rec')  # új kiszámítás
    p_hss_in = results.get('p_hss_in')
    p_hss_out = results.get('p_hss_out')
    e_hss_stor = results.get('e_hss_stor')
    t_hss = results.get('t_hss')
    p_shared = results.get('p_shared')
    p_cl_grid = results.get('p_cl_grid')
    p_cl_rec = results.get('p_cl_rec')
    p_cl_with = results.get('p_cl_with')
    d_cl = results.get('d_cl')
    p_grid_in = results.get('p_grid_in')
    p_grid_out = results.get('p_grid_out')
    # diff_hss_out = results.get('diff_hss_out').sum(axis=0)
    # diff_hss_in = results.get('diff_hss_in').sum(axis=0)
    p_pv = na_values[:, 1]
    p_ue = na_values[:, 0]
    p_ut = na_values[:, 2]
    display_figures(p_pv, p_bess_out, p_with, p_ue, p_bess_in, p_inj, e_bess_stor, p_elh_out, p_ut, p_shared, p_cl_rec,
                    p_cl_grid, p_cl_with, e_hss_stor, p_hss_out, p_hss_in, t_hss, d_cl, p_grid_in, p_grid_out, p_elh_in)


results, status, objective, num_vars, num_constraints = optimize(p_pv=na_values[:, 1], p_consumed=na_values[:, 0], p_ut=na_values[:, 2],
                                                                 size_elh=2, size_bess=8, size_hss=4, run_lp=False, gapRel=0.01,
                                                                 objective="environmental")
extract_results_and_show(results)

pv_ratios = [0.25, 0.5, 1.0, 1.5, 2.0]
bess_sizes = [0, 2, 4, 6, 8, 10]

shape = (len(pv_ratios), len(bess_sizes))
sc_profile = np.zeros(shape)
ss_profile = np.zeros(shape)
sc_optimal = np.zeros(shape)
ss_optimal = np.zeros(shape)

base_pv = df_filtered['pv1'].values
p_consumed = df_filtered['consumer1'].values
p_ut_profile = df_filtered['thermal_user1'].values
p_dhw = df_filtered['dhw'].values

for i, pv_ratio in enumerate(pv_ratios):
    for j, bess_size in enumerate(bess_sizes):
        p_pv = base_pv * pv_ratio

        # Profile szcenárió (kézi logika)
        bess_soc = np.zeros(len(p_pv))
        grid_in = np.zeros(len(p_pv))
        bess_out = np.zeros(len(p_pv))
        shared_profile = np.zeros(len(p_pv))
        bess_capacity = bess_size

        for t in range(len(p_pv)):
            demand = p_consumed[t] + p_ut_profile[t]
            net = p_pv[t] - demand
            if net >= 0:
                charge = min(net, bess_capacity - bess_soc[t-1] if t > 0 else bess_capacity)
                bess_soc[t] = (bess_soc[t-1] if t > 0 else 0) + charge
                grid_in[t] = net - charge
            else:
                discharge = min(-net, bess_soc[t-1] if t > 0 else 0)
                bess_out[t] = discharge
                bess_soc[t] = (bess_soc[t-1] if t > 0 else 0) - discharge

        p_inj_profile = p_pv + bess_out
        p_with_profile = p_consumed + p_ut_profile
        shared_profile = np.minimum(p_inj_profile, p_with_profile)

        sc_profile[i, j] = np.sum(shared_profile) / np.sum(p_inj_profile)
        ss_profile[i, j] = np.sum(shared_profile) / np.sum(p_with_profile)
        # Optimalizált szcenárió
        results, *_ = optimize(p_pv, p_consumed, p_dhw,
                               size_elh=2, size_bess=bess_size, size_hss=4, vol_hss_water=120,
                               dt=1, msg=False, objective="environmental",gapRel=0.01)

        sc_optimal[i, j] = np.sum(results['p_shared']) / np.sum(results['p_inj'])
        # Hálózatból vett energia = grid_in
        # p_grid_in = results['p_grid_in']
        # ss_optimal[i, j] = 1 - np.sum(p_grid_in) / (np.sum(p_consumed) + np.sum(cl_with));
        ss_optimal[i, j] = np.sum(results['p_shared']) / np.sum(results['p_with'])

# 1. Heatmapek
def plot_heatmap(data, title):
    plt.figure(figsize=(7, 5))
    ax = sns.heatmap(data, annot=True, fmt=".2f", cmap="YlGnBu",
                     xticklabels=bess_sizes, yticklabels=pv_ratios)
    ax.set_xlabel("BESS size [kWh]")
    ax.set_ylabel("PV ratio")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

plot_heatmap(sc_profile, "SC - Profile")
plot_heatmap(ss_profile, "SS - Profile")
plot_heatmap(sc_optimal, "SC - Optimal Control")
plot_heatmap(ss_optimal, "SS - Optimal Control")

import matplotlib.patches as mpatches

# Élénkebb színpaletta
colors = sns.color_palette("tab10", len(pv_ratios))

# SCI vs. SSI diagram: PV ratio → szín, BESS size → méret, origó [0.2, 0.2]
plt.figure(figsize=(10, 6))

for i, pv in enumerate(pv_ratios):
    for j, bess in enumerate(bess_sizes):
        x = ss_profile[i, j]
        y = sc_profile[i, j]
        size = 40 + bess * 10
        plt.scatter(x, y, s=size, c=[colors[i]], marker='o', alpha=0.9,
                    edgecolors='black', linewidths=0.6, label=f"PV={pv}" if j == 0 else "")

plt.xlabel("SSI (Self-Sufficiency Index)")
plt.ylabel("SCI (Self-Consumption Index)")
plt.title("SCI vs. SSI (Profile Scenario)\nPV arány színekkel, BESS méret pontmérettel")
plt.xlim(0.2, 1.0)
plt.ylim(0.2, 1.0)

# PV színekhez legenda
patches_pv = [mpatches.Patch(color=colors[i], label=f"PV={pv_ratios[i]}") for i in range(len(pv_ratios))]
legend_pv = plt.legend(handles=patches_pv, title="PV ratio", loc="lower right")

# BESS mérethez (üres kör) legenda
scatter_handles = [
    plt.scatter([], [], s=40 + b * 10, edgecolors='black', facecolors='none', marker='o', label=f'{b} kWh')
    for b in bess_sizes
]
legend_bess = plt.legend(handles=scatter_handles, title="BESS size", loc="lower center",
                         bbox_to_anchor=(0.5, -0.25), ncol=len(bess_sizes))

plt.gca().add_artist(legend_pv)
plt.grid(True)
plt.tight_layout()
plt.show()

# Optimalizált SCI és SSI alapján: PV arány színnel, BESS méret pontmérettel
plt.figure(figsize=(10, 6))
colors = sns.color_palette("tab10", len(pv_ratios))

for i, pv in enumerate(pv_ratios):
    for j, bess in enumerate(bess_sizes):
        x = ss_optimal[i, j]
        y = sc_optimal[i, j]
        size = 40 + bess * 10
        plt.scatter(x, y, s=size, c=[colors[i]], marker='o', alpha=0.9,
                    edgecolors='black', linewidths=0.6, label=f"PV={pv}" if j == 0 else "")

plt.xlabel("SSI (Self-Sufficiency Index)")
plt.ylabel("SCI (Self-Consumption Index)")
plt.title("SCI vs. SSI (Optimal Control Scenario)\nPV arány színekkel, BESS méret pontmérettel")

# PV színekhez legenda
patches_pv = [mpatches.Patch(color=colors[i], label=f"PV={pv_ratios[i]}") for i in range(len(pv_ratios))]
legend_pv = plt.legend(handles=patches_pv, title="PV ratio", loc="lower right")

# BESS mérethez legenda
scatter_handles = [
    plt.scatter([], [], s=40 + b * 10, edgecolors='black', facecolors='none', marker='o', label=f'{b} kWh')
    for b in bess_sizes
]
legend_bess = plt.legend(handles=scatter_handles, title="BESS size", loc="lower center",
                         bbox_to_anchor=(0.5, -0.25), ncol=len(bess_sizes))

plt.gca().add_artist(legend_pv)
plt.grid(True)
plt.tight_layout()
plt.show()