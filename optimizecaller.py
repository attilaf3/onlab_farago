import pandas as pd
from matplotlib import pyplot as plt, patheffects as pe
import numpy as np
from optimize import optimize

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
    dt = 168
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
        fig, axes = plt.subplots(nrows=3, ncols=2, figsize=figsize, sharex=True,
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
        ax.plot(time, p_inj[t0:tf], label=r'$P_\mathrm{inj}$', color='tab:red', **plot_kw)
        ax.plot(time, p_with[t0:tf], label=r'$P_\mathrm{with}$', color='tab:blue', **plot_kw)
        ax.plot(time, p_shared[t0:tf], label=r'$P_\mathrm{shared}$', color='tab:green', ls='', marker='s', **plot_kw)
        ax.fill_between(t_plot, p_shared_plot, p_with_plot, where=p_with_plot > p_shared_plot,
                        label=r'E$_\mathrm{\leftarrow grid}$', color='tab:blue', **area_kw)
        ax.fill_between(t_plot, p_shared_plot, p_inj_plot, where=p_inj_plot > p_shared_plot,
                        label=r'E$_\mathrm{\rightarrow grid}$', color='tab:red', **area_kw)
        ax.fill_between(t_plot, 0, p_shared_plot, where=p_shared_plot > 0, label=r'E$_\mathrm{shared}$',
                        color='tab:green')

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
    p_elh_out = results.get('p_elh_out')
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


results, status, objective, num_vars, num_constraints = optimize(na_values[:, 1], na_values[:, 0], na_values[:, 2],
                                                                 size_elh=4, size_bess=20, size_hss=130, run_lp=False,
                                                                 objective="environmental")
# results, status, objective, num_vars, num_constraints = optimize(na_values[:, 0], na_values[:, 1], na_values[:, 2], size_elh=4, size_bess=20)
extract_results_and_show(results)
