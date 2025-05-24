from matplotlib import pyplot as plt, patheffects as pe
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.dates as mdates
from optimize_with_hp import optimize_with_hp

df = pd.read_csv('input.csv', sep=';', index_col=0, parse_dates=True)
# negyedórás → órás felbontás
df_filtered = df.resample("1h").agg({
    'consumer1': 'sum',  # energiamérleg-igények összeadása
    'pv1': 'sum',  # PV-termelés összeadva (kWh)
    'thermal_user1': 'sum',  # fűtési igény összeadva (kWh)
    'dhw': 'sum',  # DHW-igény összeadva (kWh)
    'temperature': 'mean',  # átlagos külső hőmérséklet (°C)
    'solar_radiation_direct': 'mean',  # átlagos direkt irradiancia (W/m²)
    'solar_radiation_diffuse': 'mean'  # átlagos szórt irradiancia (W/m²)
})

# Hiányzó hőmérséklet és napenergia adatok pótlása
for col in ['temperature', 'solar_radiation_direct', 'solar_radiation_diffuse']:
    df_filtered[col] = (
        df_filtered[col]
        .interpolate(method='linear')
        .bfill()
        .ffill()
    )

na_values = df_filtered.values
p_consumed = na_values[:, 0]
p_pv = na_values[:, 1]
p_ut = na_values[:, 2]
T_env = na_values[:, 4]
solar_dir = na_values[:, 5]
solar_dif = na_values[:, 6]
COP = 2.0 + 1.5 / (1 + np.exp(-0.2 * (T_env - 5)))

results, status, objective, num_vars, num_constraints = optimize_with_hp(p_pv=p_pv, p_consumed=p_consumed, p_ut=p_ut,
                                                                         size_elh=2, size_bess=8, size_hss=4,
                                                                         run_lp=False,
                                                                         objective="environmental", T_env_vector=T_env,
                                                                         cop_hp_vector=COP,
                                                                         solar_radiation_direct=solar_dir,
                                                                         solar_radiation_diffuse=solar_dif,
                                                                         gapRel=0.01)

# Index generálása
n_hours = len(results["T_zone"])
time_index = pd.date_range(start="2023-01-01", periods=n_hours, freq="h")

# Kombinált HP működés (ha bármelyik irány aktív)
d_hp_heat = np.array(results["d_hp_heat"])
d_hp_cool = np.array(results["d_hp_cool"])
d_hp_total = np.clip(d_hp_heat + d_hp_cool, 0, 1)


# Időszakok (évszakok)
def get_weekly_slice(start_str):
    start_time = pd.Timestamp(start_str)
    start_idx = time_index.get_indexer([start_time])[0]
    end_idx = start_idx + 24 * 7
    return slice(start_idx, end_idx)


seasons = {
    "Tél": get_weekly_slice("2023-01-01"),
    "Tavasz": get_weekly_slice("2023-04-01"),
    "Nyár": get_weekly_slice("2023-07-01"),
    "Ősz": get_weekly_slice("2023-10-01"),
}

# Ábra
fig, axs = plt.subplots(4, 2, figsize=(18, 10), sharex=False)

for i, (season, sl) in enumerate(seasons.items()):
    # Hőmérsékletek
    ax_temp = axs[i, 0]
    ax_temp.plot(time_index[sl], results["T_zone"][sl], label="T_zone")
    ax_temp.plot(time_index[sl], results["T_mass"][sl], label="T_mass")
    ax_temp.plot(time_index[sl], T_env[sl], label="T_env", linestyle="dashed", color="green")
    ax_temp.set_ylabel("Hőmérséklet [°C]")
    ax_temp.set_title(f"{season} – Hőmérsékletek")
    ax_temp.legend(loc="lower left")
    ax_temp.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

    # HP működés
    ax_hp = axs[i, 1]
    ax_hp.plot(time_index[sl], d_hp_heat[sl], label="HP fűtés", color="red")
    ax_hp.plot(time_index[sl], d_hp_cool[sl], label="HP hűtés", color="blue")
    ax_hp.set_ylabel("HP státusz")
    ax_hp.set_ylim(-0.05, 1.05)
    ax_hp.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    if i == 0:
        ax_hp.legend(loc="upper right")

fig.tight_layout()
plt.show()


def display_figures(p_pv, p_bess_out, p_with, p_ue, p_bess_in, p_inj, e_bess_stor, p_elh_out,
                    p_ut, p_shared, p_cl_rec, p_cl_grid, p_cl_with, e_hss_stor, p_hss_out, p_hss_in, t_hss, d_cl,
                    p_grid_in, p_grid_out, diff_hss_out=None, diff_hss_in=None, params=None, p_hp_th=None, p_hp_el=None,
                    d_hp_heat=None,
                    d_hp_cool=None,
                    T_zone=None,
                    T_mass=None,
                    T_env=None):
    global time, t_plot
    # One week in each season
    figsize = (20, 15)
    fontsize = 15
    t0s = [0+168+48, 2184, 4368 + 168*2, 6552]
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

        # Make figure
        fig, axes = plt.subplots(nrows=4, ncols=2, figsize=figsize, sharex=True,
                                 gridspec_kw=dict(width_ratios=[0.8, 0.2]))

        # Electric node of the condominium
        ax = axes[0, 0]
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
        bottom -= p_grid_in[t0:tf]
        ax.bar(time, -p_hp_el[t0:tf], bottom=bottom, label=r'$P_\mathrm{hp,el}$', **bar_kw)

        # Plot storage SOC
        axtw = ax.twinx()
        axtw.plot(time, e_bess_stor[t0:tf], color='lightgrey', ls='--')

        # Adjust and show
        ax.set_xlabel("Time (h)", fontsize=fontsize)
        ax.set_ylabel("Power (kW)", fontsize=fontsize)
        ax.set_title("Electric hub", fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)
        ax.grid()

        axtw.set_ylabel("Stored energy (kWh)")
        axtw.spines['right'].set_color('lightgrey')
        axtw.tick_params(axis='y', colors='lightgrey')
        axtw.yaxis.label.set_color('lightgrey')

        # Legend
        handles, labels = ax.get_legend_handles_labels()
        axes[0, 1].legend(handles, labels, fontsize=fontsize, loc='center')
        axes[0, 1].axis('off')

        # Thermal node
        ax = axes[1, 0]
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
        ax.bar(time, -p_elh_out[t0:tf], bottom=bottom, label=r'$P_\mathrm{elh,out}$', **bar_kw)
        bottom -= p_elh_out[t0:tf]
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
        ax.set_title("Thermal node", fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)
        ax.grid()

        # # Legend
        handles, labels = ax.get_legend_handles_labels()
        axtd_handles, axtd_labels = axtd.get_legend_handles_labels()
        axtw_handles, axtw_labels = axtw.get_legend_handles_labels()
        axes[1, 1].legend(handles + axtd_handles + axtw_handles, labels + axtd_labels + axtw_labels, fontsize=fontsize,
                          loc='center', ncol=2)
        axes[1, 1].axis('off')

        # CSC
        ax = axes[2, 0]
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
        ax.set_title("CSC", fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)
        ax.grid()

        # Legend
        handles, labels = ax.get_legend_handles_labels()
        axes[2, 1].legend(handles, labels, fontsize=fontsize, loc='center')
        axes[2, 1].axis('off')

        # 4. subplot – hőszivattyú működés + hőmérsékletek
        ax = axes[3, 0]
        tf = t0 + dt
        time = np.arange(t0, tf)

        # HP működés (összevonva)
        d_hp_total = np.clip(np.array(d_hp_heat[t0:tf]) + np.array(d_hp_cool[t0:tf]), 0, 1)

        # Hőmérsékletek
        ax.plot(time, T_zone[t0:tf], label="T_zone", color="tab:blue")
        ax.plot(time, T_mass[t0:tf], label="T_mass", color="tab:orange")
        ax.plot(time, T_env[t0:tf], label="T_env", linestyle="dashed", color="tab:green")
        ax.set_ylabel("T [°C]")
        ax.set_title("HP és hőmérsékletek")
        ax.legend(loc="lower left")
        ax.grid()

        # HP működés overlay
        ax2 = ax.twinx()
        ax2.plot(time, d_hp_total, label="HP működés", color="red", linestyle="dotted")
        ax2.set_ylim(-0.05, 1.05)
        ax2.set_ylabel("HP on/off")
        ax2.legend(loc="upper right")

        # Üres jobb oldali subplot
        axes[3, 1].axis("off")

        # Adjust and show
        plt.subplots_adjust(top=0.55)  # Adjust the position of the overall title
        fig.suptitle(titles[i], fontsize=fontsize)
        fig.tight_layout()
        # fig.subplots_adjust(left=0.063, bottom=0.095, right=0.905,top=0.88, wspace=0.413, hspace=0.564)
        # plt.savefig('oneuser_'+titles[i].lower(), dpi=300, bbox_inches='tight')
        # output_dir = config.get("path", "figures_output")
        # makedirs(output_dir, exist_ok=True)
        # plt.savefig(
        #     join(output_dir, f"{titles[i]}_{config.get('simulation', 'experiment')}{filename_post(params)}.png"))
        plt.show()


def extract_results_and_show(results):
    # Get results
    p_inj = results.get('p_inj')
    p_with = results.get('p_with')
    p_bess_in = results.get('p_bess_in')
    p_bess_out = results.get('p_bess_out')
    e_bess_stor = results.get('e_bess_stor')
    p_elh_in = results.get('p_elh_in')
    # p_elh_out = results.get('p_elh_out')
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
    display_figures(p_pv, p_bess_out, p_with, p_ue, p_bess_in, p_inj, e_bess_stor,
                    p_elh_out, p_ut, p_shared, p_cl_rec, p_cl_grid, p_cl_with,
                    e_hss_stor, p_hss_out, p_hss_in, t_hss, d_cl,
                    p_grid_in, p_grid_out, p_elh_in,
                    p_hp_th=results.get("p_hp_th"),
                    p_hp_el=results.get("p_hp_el"),
                    d_hp_heat=results.get("d_hp_heat"),
                    d_hp_cool=results.get("d_hp_cool"),
                    T_zone=results.get("T_zone"),
                    T_mass=results.get("T_mass"),
                    T_env=T_env)

#extract_results_and_show(results)

    # # Napi működési idő külön fűtésre és hűtésre
    # s_hp_heat = pd.Series(d_hp_heat, index=time_index).resample("D").sum()
    # s_hp_cool = pd.Series(d_hp_cool, index=time_index).resample("D").sum()
    #
    # plt.figure(figsize=(14, 5))
    # plt.bar(s_hp_heat.index, s_hp_heat.values, label="Fűtés (óra)", color="red")
    # plt.bar(s_hp_cool.index, s_hp_cool.values, bottom=s_hp_heat.values, label="Hűtés (óra)", color="blue")
    # plt.axhline(20, color="gray", linestyle="--", label="Max. napi működés")
    # plt.ylabel("HP működés [óra/nap]")
    # plt.xlabel("Dátum")
    # plt.title("Hőszivattyú napi működési ideje bontva (fűtés-hűtés)")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()


# Get results
p_inj = results.get('p_inj')
p_with = results.get('p_with')
p_bess_in = results.get('p_bess_in')
p_bess_out = results.get('p_bess_out')
e_bess_stor = results.get('e_bess_stor')
p_elh_in = results.get('p_elh_in')
# p_elh_out = results.get('p_elh_out')
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
p_hp_el = results.get('p_hp_el')
p_hp_th = results.get('p_hp_th')
T_mass = results.get('T_mass')
T_zone = results.get('T_zone')
p_pv = na_values[:, 1]
p_ue = na_values[:, 0]
p_ut = na_values[:, 2]
# diff_hss_out = results.get('diff_hss_out').sum(axis=0)
# diff_hss_in = results.get('diff_hss_in').sum(axis=0)

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

# 4 szezon
t0s = [168 * 4, 2184, 4368 + 168 * 3, 6552]
titles = ['Winter', 'Spring', 'Summer', 'Autumn']
dt = 72
fontsize = 14
bar_kw = dict(width=0.8)

for i, t0 in enumerate(t0s):
    tf = t0 + dt
    time = np.arange(t0, tf)
    title = titles[i]

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(f"{title} – Electric hub & HP", fontsize=18)

    gs = GridSpec(nrows=2, ncols=2, width_ratios=[0.8, 0.2], height_ratios=[1, 1], hspace=0.3, wspace=0.25)

    # --- Electric hub subplot (bal felső) ---
    ax = fig.add_subplot(gs[0, 0])
    bottom = np.zeros_like(time, dtype=float)
    h1 = ax.bar(time, p_pv[t0:tf], bottom=bottom, label=r'$P_\mathrm{pv}$', **bar_kw)
    bottom += p_pv[t0:tf]
    h2 = ax.bar(time, p_bess_out[t0:tf], bottom=bottom, label=r'$P_\mathrm{bess,out}$', **bar_kw)
    bottom += p_bess_out[t0:tf]
    h3 = ax.bar(time, p_grid_out[t0:tf], bottom=bottom, label=r'$P_\mathrm{grid,out}$', **bar_kw)

    bottom = np.zeros_like(time, dtype=float)
    h4 = ax.bar(time, -p_ue[t0:tf], bottom=bottom, label=r'$P_\mathrm{ue}$', **bar_kw)
    bottom -= p_ue[t0:tf]
    h5 = ax.bar(time, -p_bess_in[t0:tf], bottom=bottom, label=r'$P_\mathrm{bess,in}$', **bar_kw)
    bottom -= p_bess_in[t0:tf]
    h6 = ax.bar(time, -p_cl_grid[t0:tf], bottom=bottom, label=r'$P_\mathrm{cl,grid}$', **bar_kw)
    bottom -= p_cl_grid[t0:tf]
    h7 = ax.bar(time, -p_cl_rec[t0:tf], bottom=bottom, label=r'$P_\mathrm{cl,rec}$', **bar_kw)
    bottom -= p_cl_rec[t0:tf]
    h8 = ax.bar(time, -p_grid_in[t0:tf], bottom=bottom, label=r'$P_\mathrm{grid,in}$', **bar_kw)
    bottom -= p_grid_in[t0:tf]
    h9 = ax.bar(time, -p_hp_el[t0:tf], bottom=bottom, label=r'$P_\mathrm{hp,el}$', color="orange", **bar_kw)

    axtw = ax.twinx()
    h10, = axtw.plot(time, e_bess_stor[t0:tf], color='black', ls='--', label=r'$E_\mathrm{stor,bess}$')
    axtw.set_ylabel("Stored energy (kWh)")
    axtw.spines["right"].set_position(("axes", 1.0))
    axtw.tick_params(labelsize=fontsize)

    ax.set_title("Electric hub", fontsize=fontsize)
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Power (kW)")
    ax.grid()
    ax.tick_params(labelsize=fontsize)

    # --- Legend (jobb felső) ---
    legend_ax = fig.add_subplot(gs[0, 1])
    legend_ax.axis('off')
    legend_ax.legend([h1, h2, h3, h4, h5, h6, h7, h8, h9, h10],
                     [r'$P_\mathrm{pv}$', r'$P_\mathrm{bess,out}$', r'$P_\mathrm{grid,out}$',
                      r'$P_\mathrm{ue}$', r'$P_\mathrm{bess,in}$', r'$P_\mathrm{cl,grid}$',
                      r'$P_\mathrm{cl,rec}$', r'$P_\mathrm{grid,in}$', r'$P_\mathrm{hp,el}$',
                      r'$E_\mathrm{stor,bess}$'],
                     fontsize=fontsize, loc='center')

    # --- Temp subplot (bal alsó) ---
    ax = fig.add_subplot(gs[1, 0])
    d_hp_total = np.clip(np.array(d_hp_heat[t0:tf]) + np.array(d_hp_cool[t0:tf]), 0, 1)

    line1, = ax.plot(time, T_zone[t0:tf], label="T_zone", color="tab:blue")
    line2, = ax.plot(time, T_mass[t0:tf], label="T_mass", color="tab:orange")
    line3, = ax.plot(time, T_env[t0:tf], label="T_env", linestyle="dashed", color="tab:green")
    ax.set_ylabel("T [°C]")
    ax.set_title("Temperatures & HP on/off ")
    ax.grid()

    ax2 = ax.twinx()
    line4, = ax2.plot(time, d_hp_total, label="HP on/off", color="red", linestyle="dotted")
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_ylabel("HP on/off", )

    # --- Legend (jobb alsó) ---
    legend_ax2 = fig.add_subplot(gs[1, 1])
    legend_ax2.axis('off')
    legend_ax2.legend([line1, line2, line3, line4],
                      ["T_zone", "T_mass", "T_env", "HP működés"],
                      fontsize=fontsize, loc="center")

    plt.tight_layout()
    plt.show()
