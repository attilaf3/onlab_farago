import pandas as pd
import numpy as np
from matplotlib import pyplot as plt, patheffects as pe
from optimize_two_users import optimize_two_users, p_shared

# Adatok betöltése
df = pd.read_csv('input_tobb_haztartas.csv', sep=';', index_col=0, parse_dates=True)
df_filtered = df.resample("1h").sum()
na_p_consumed = df_filtered[["consumer1", "consumer2"]].to_numpy()  # P_ue
na_p_pv = df_filtered[["pv1", "pv2"]].to_numpy()  # P_pv
na_p_ut = df_filtered[["thermal_user1", "thermal_user2"]].to_numpy()  # P_ut


def ensure_numeric(arr, default_value=0.0, var_name="variable"):
    """Biztosítja, hogy a tömb numerikus legyen, None vagy nem numerikus értékeket default_value-ra cserélve."""
    if arr is None:
        print(f"Warning: {var_name} is None, replacing with zeros")
        return np.zeros(
            (8760, 2) if 'user' in var_name or 'cl_' in var_name or 'elh_' in var_name or 'hss_' in var_name else 8760)
    arr = np.array(arr, dtype=object)  # Objektum típusként kezeljük először
    # Cseréljük None vagy nem numerikus értékeket default_value-ra
    arr = np.where(arr == None, default_value, arr)  # None értékek cseréje
    try:
        arr = arr.astype(np.float64)  # Konvertálás float64-re
    except ValueError as e:
        print(f"Warning: {var_name} contains non-numeric values, replacing with {default_value}. Error: {e}")
        arr = np.full_like(arr, default_value, dtype=np.float64)
    return arr


def display_figures_two_users(results, p_pv, p_consumed, p_ut, hss_flag=True):
    global time, t_plot
    # Egy hét minden évszakban
    figsize = (24, 15)  # Nagyobb méret a 3x3-as elrendezéshez
    fontsize = 15
    t0s = [0, 2184, 4368, 6552]  # Kezdő időt lépések (tél, tavasz, nyár, ősz)
    dt = 168  # Egy hét (168 óra)
    titles = ['Winter', 'Spring', 'Summer', 'Autumn']
    path_effect = lambda lw: [pe.Stroke(linewidth=1.5 * lw, foreground='w'), pe.Normal()]
    bar_kw = dict(width=0.4)  # Keskenyebb sávok
    plot_kw = dict(lw=3, path_effects=path_effect(3))
    area_kw = dict(alpha=0.6)

    for i, t0 in enumerate(t0s):
        tf = t0 + dt
        time = np.arange(t0, tf)

        # 3x3-as subplotok
        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=figsize, sharex=True,
                                 gridspec_kw=dict(width_ratios=[0.4, 0.4, 0.2]))

        # 1. felhasználó (0. oszlop)
        # Elektromos csomópont
        ax = axes[0, 0]
        bottom = np.zeros_like(time, dtype=float)
        ax.bar(time, p_pv[t0:tf, 0], bottom=bottom, label=r'$P_\mathrm{pv}$', **bar_kw)
        bottom += p_pv[t0:tf, 0]
        ax.bar(time, results['p_bess_out'][t0:tf], bottom=bottom, label=r'$P_\mathrm{bess,out}$', **bar_kw)
        bottom += results['p_bess_out'][t0:tf]
        ax.bar(time, results['p_grid_out'][t0:tf], bottom=bottom, label=r'$P_\mathrm{grid,out}$', **bar_kw)

        bottom = np.zeros_like(time, dtype=float)
        ax.bar(time, -p_consumed[t0:tf, 0], bottom=bottom, label=r'$P_\mathrm{ue}$', **bar_kw)
        bottom -= p_consumed[t0:tf, 0]
        ax.bar(time, -results['p_bess_in'][t0:tf], bottom=bottom, label=r'$P_\mathrm{bess,in}$', **bar_kw)
        bottom -= results['p_bess_in'][t0:tf]
        ax.bar(time, -results['p_cl_grid'][t0:tf, 0], bottom=bottom, label=r'$P_\mathrm{cl,grid}$', **bar_kw)
        bottom -= results['p_cl_grid'][t0:tf, 0]
        ax.bar(time, -results['p_cl_rec'][t0:tf, 0], bottom=bottom, label=r'$P_\mathrm{cl,rec}$', **bar_kw)
        bottom -= results['p_cl_rec'][t0:tf, 0]
        ax.bar(time, -results['p_grid_in'][t0:tf], bottom=bottom, label=r'$P_\mathrm{grid,in}$', **bar_kw)

        # Akkumulátor SOC
        axtw = ax.twinx()
        axtw.plot(time, results['e_bess_stor'][t0:tf], color='lightgrey', ls='--')
        axtw.set_ylabel("Stored energy (kWh)")
        axtw.spines['right'].set_color('lightgrey')
        axtw.tick_params(axis='y', colors='lightgrey')
        axtw.yaxis.label.set_color('lightgrey')

        ax.set_xlabel("Time (h)", fontsize=fontsize)
        ax.set_ylabel("Power (kW)", fontsize=fontsize)
        ax.set_title("Electric hub (User 1)", fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)
        ax.grid()

        # Termikus csomópont
        ax = axes[1, 0]
        bottom = np.zeros_like(time, dtype=float)
        ax.bar(time, results['p_cl_grid'][t0:tf, 0], bottom=bottom, label=r'$P_\mathrm{cl,grid}$', **bar_kw)
        bottom += results['p_cl_grid'][t0:tf, 0]
        ax.bar(time, results['p_cl_rec'][t0:tf, 0], bottom=bottom, label=r'$P_\mathrm{cl,rec}$', **bar_kw)
        bottom += results['p_cl_rec'][t0:tf, 0]
        if hss_flag:
            ax.bar(time, results['p_hss_out'][t0:tf, 0], bottom=bottom, label=r'$P_\mathrm{hss,out}$', **bar_kw)
            bottom += results['p_hss_out'][t0:tf, 0]

        bottom = np.zeros_like(time, dtype=float)
        ax.bar(time, -results['p_elh_out'][t0:tf, 0], bottom=bottom, label=r'$P_\mathrm{elh,out}$', **bar_kw)
        bottom -= results['p_elh_out'][t0:tf, 0]
        ax.bar(time, -p_ut[t0:tf, 0], bottom=bottom, label=r'$P_\mathrm{ut}$', **bar_kw)
        bottom -= p_ut[t0:tf, 0]
        if hss_flag:
            ax.bar(time, -results['p_hss_in'][t0:tf, 0], bottom=bottom, label=r'$P_\mathrm{hss,in}$', **bar_kw)
            bottom -= results['p_hss_in'][t0:tf, 0]

        # Termikus tároló SOC
        if hss_flag:
            axtw = ax.twinx()
            axtw.plot(time, results['e_hss_stor'][t0:tf, 0], color='black', ls='--', label=r"$\mathrm{E_{stor,hss}}$")
            axtw.set_ylabel("Stored energy (kWh)")
            axtw.spines['right'].set_color('black')
            axtw.tick_params(axis='y', colors='black')
            axtw.yaxis.label.set_color('black')
            axtw.legend().set_visible(False)

        # Vezérlőjel
        axtd = ax.twinx()
        axtd.plot(time, results['d_cl'][t0:tf], '.', ls='-', color='lightgrey',
                  label=r"$\mathrm{control\ signal\ (on/off)}$")
        axtd.spines["top"].set_visible(False)
        axtd.spines["right"].set_visible(False)
        axtd.spines["left"].set_visible(False)
        axtd.spines["bottom"].set_visible(False)
        axtd.tick_params(axis="both", which='both', length=0, labelcolor="none")
        axtd.legend().set_visible(False)

        ax.set_xlabel("Time (h)", fontsize=fontsize)
        ax.set_ylabel("Power (kW)", fontsize=fontsize)
        ax.set_title("Thermal node (User 1)", fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)
        ax.grid()

        # CSC (User 1)
        ax = axes[2, 0]
        time = np.arange(t0, tf)
        p_inj = results['p_inj'][t0:tf]
        p_with = results['p_with'][t0:tf]
        p_shared = results['p_shared'][t0:tf]

        ax.plot(time, p_inj, label=r'$P_\mathrm{inj}$', color='tab:red')
        ax.plot(time, p_with, label=r'$P_\mathrm{with}$', color='tab:blue')
        ax.plot(time, p_shared, label=r'$P_\mathrm{shared}$', color='tab:green')

        ax.fill_between(time, p_shared, p_with, where=p_with > p_shared, label='E_from_grid', color='tab:blue',
                        alpha=0.3)
        ax.fill_between(time, p_shared, p_inj, where=p_inj > p_shared, label='E_to_grid', color='tab:red', alpha=0.3)
        ax.fill_between(time, 0, p_shared, where=p_shared > 0, label='E_shared', color='tab:green', alpha=0.6)

        ax.set_title("Community CSC – teljes rendszer energiacseréi")
        ax.set_xlabel("Időlépés")
        ax.set_ylabel("Teljesítmény [kW]")
        ax.legend()
        ax.grid(True)

        # 2. felhasználó (1. oszlop)
        # Elektromos csomópont
        ax = axes[0, 1]
        bottom = np.zeros_like(time, dtype=float)
        ax.bar(time, p_pv[t0:tf, 1], bottom=bottom, label=r'$P_\mathrm{pv}$', **bar_kw)
        bottom += p_pv[t0:tf, 1]
        ax.bar(time, results['p_bess_out'][t0:tf], bottom=bottom, label=r'$P_\mathrm{bess,out}$', **bar_kw)
        bottom += results['p_bess_out'][t0:tf]
        ax.bar(time, results['p_grid_out'][t0:tf], bottom=bottom, label=r'$P_\mathrm{grid,out}$', **bar_kw)

        bottom = np.zeros_like(time, dtype=float)
        ax.bar(time, -p_consumed[t0:tf, 1], bottom=bottom, label=r'$P_\mathrm{ue}$', **bar_kw)
        bottom -= p_consumed[t0:tf, 1]
        ax.bar(time, -results['p_bess_in'][t0:tf], bottom=bottom, label=r'$P_\mathrm{bess,in}$', **bar_kw)
        bottom -= results['p_bess_in'][t0:tf]
        ax.bar(time, -results['p_cl_grid'][t0:tf, 1], bottom=bottom, label=r'$P_\mathrm{cl,grid}$', **bar_kw)
        bottom -= results['p_cl_grid'][t0:tf, 1]
        ax.bar(time, -results['p_cl_rec'][t0:tf, 1], bottom=bottom, label=r'$P_\mathrm{cl,rec}$', **bar_kw)
        bottom -= results['p_cl_rec'][t0:tf, 1]
        ax.bar(time, -results['p_grid_in'][t0:tf], bottom=bottom, label=r'$P_\mathrm{grid,in}$', **bar_kw)

        # Akkumulátor SOC
        axtw = ax.twinx()
        axtw.plot(time, results['e_bess_stor'][t0:tf], color='lightgrey', ls='--')
        axtw.set_ylabel("Stored energy (kWh)")
        axtw.spines['right'].set_color('lightgrey')
        axtw.tick_params(axis='y', colors='lightgrey')
        axtw.yaxis.label.set_color('lightgrey')

        ax.set_xlabel("Time (h)", fontsize=fontsize)
        ax.set_ylabel("Power (kW)", fontsize=fontsize)
        ax.set_title("Electric hub (User 2)", fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)
        ax.grid()

        # Termikus csomópont
        ax = axes[1, 1]
        bottom = np.zeros_like(time, dtype=float)
        ax.bar(time, results['p_cl_grid'][t0:tf, 1], bottom=bottom, label=r'$P_\mathrm{cl,grid}$', **bar_kw)
        bottom += results['p_cl_grid'][t0:tf, 1]
        ax.bar(time, results['p_cl_rec'][t0:tf, 1], bottom=bottom, label=r'$P_\mathrm{cl,rec}$', **bar_kw)
        bottom += results['p_cl_rec'][t0:tf, 1]
        if hss_flag:
            ax.bar(time, results['p_hss_out'][t0:tf, 1], bottom=bottom, label=r'$P_\mathrm{hss,out}$', **bar_kw)
            bottom += results['p_hss_out'][t0:tf, 1]

        bottom = np.zeros_like(time, dtype=float)
        ax.bar(time, -results['p_elh_out'][t0:tf, 1], bottom=bottom, label=r'$P_\mathrm{elh,out}$', **bar_kw)
        bottom -= results['p_elh_out'][t0:tf, 1]
        ax.bar(time, -p_ut[t0:tf, 1], bottom=bottom, label=r'$P_\mathrm{ut}$', **bar_kw)
        bottom -= p_ut[t0:tf, 1]
        if hss_flag:
            ax.bar(time, -results['p_hss_in'][t0:tf, 1], bottom=bottom, label=r'$P_\mathrm{hss,in}$', **bar_kw)
            bottom -= results['p_hss_in'][t0:tf, 1]

        # Termikus tároló SOC
        if hss_flag:
            axtw = ax.twinx()
            axtw.plot(time, results['e_hss_stor'][t0:tf, 1], color='black', ls='--', label=r"$\mathrm{E_{stor,hss}}$")
            axtw.set_ylabel("Stored energy (kWh)")
            axtw.spines['right'].set_color('black')
            axtw.tick_params(axis='y', colors='black')
            axtw.yaxis.label.set_color('black')
            axtw.legend().set_visible(False)

        # Vezérlőjel
        axtd = ax.twinx()
        axtd.plot(time, results['d_cl'][t0:tf], '.', ls='-', color='lightgrey',
                  label=r"$\mathrm{control\ signal\ (on/off)}$")
        axtd.spines["top"].set_visible(False)
        axtd.spines["right"].set_visible(False)
        axtd.spines["left"].set_visible(False)
        axtd.spines["bottom"].set_visible(False)
        axtd.tick_params(axis="both", which='both', length=0, labelcolor="none")
        axtd.legend().set_visible(False)

        ax.set_xlabel("Time (h)", fontsize=fontsize)
        ax.set_ylabel("Power (kW)", fontsize=fontsize)
        ax.set_title("Thermal node (User 2)", fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)
        ax.grid()

        ax = axes[2, 1]
        eff = np.divide(results['p_shared'][t0:tf], results['p_inj'][t0:tf],
                        out=np.zeros_like(results['p_shared'][t0:tf]), where=results['p_inj'][t0:tf] > 0)
        ax.plot(time, eff, color='purple', lw=2)
        ax.set_title("Megosztás hatékonysága ($P_{shared}/P_{inj}$)")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Arány")
        ax.set_xlabel("Időlépés")
        ax.grid(True)

        # Jelmagyarázatok (3. oszlop)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        axes[0, 2].legend(handles, labels, fontsize=fontsize, loc='center')
        axes[0, 2].axis('off')

        handles, labels = axes[1, 0].get_legend_handles_labels()
        axtd_handles, axtd_labels = axtd.get_legend_handles_labels()
        if hss_flag:
            axtw_handles, axtw_labels = axtw.get_legend_handles_labels()
            axes[1, 2].legend(handles + axtd_handles + axtw_handles, labels + axtd_labels + axtw_labels,
                              fontsize=fontsize, loc='center', ncol=2)
        else:
            axes[1, 2].legend(handles + axtd_handles, labels + axtd_labels, fontsize=fontsize, loc='center', ncol=2)
        axes[1, 2].axis('off')

        handles, labels = axes[2, 0].get_legend_handles_labels()
        axes[2, 2].legend(handles, labels, fontsize=fontsize, loc='center')
        axes[2, 2].axis('off')

        # Ábrázolás beállítása
        plt.subplots_adjust(top=0.55)
        fig.suptitle(titles[i], fontsize=fontsize)
        fig.tight_layout()
        plt.show()


def extract_results_and_show_two_users(results, p_pv, p_consumed, p_ut):
    # Eredmények kinyerése és numerikus típus biztosítása
    p_inj_user = ensure_numeric(results.get('p_inj_user'), var_name="p_inj_user")
    p_with_user = ensure_numeric(results.get('p_with_user'), var_name="p_with_user")
    p_cl_with = ensure_numeric(results.get('p_cl_with'), var_name="p_cl_with")
    p_grid_in = ensure_numeric(results.get('p_grid_in'), var_name="p_grid_in")
    p_grid_out = ensure_numeric(results.get('p_grid_out'), var_name="p_grid_out")
    p_bess_in = ensure_numeric(results.get('p_bess_in'), var_name="p_bess_in")
    p_bess_out = ensure_numeric(results.get('p_bess_out'), var_name="p_bess_out")
    e_bess_stor = ensure_numeric(results.get('e_bess_stor'), var_name="e_bess_stor")
    p_elh_out = ensure_numeric(results.get('p_elh_out'), var_name="p_elh_out")
    p_cl_grid = ensure_numeric(results.get('p_cl_grid'), var_name="p_cl_grid")
    p_cl_rec = ensure_numeric(results.get('p_cl_rec'), var_name="p_cl_rec")
    d_cl = ensure_numeric(results.get('d_cl'), var_name="d_cl")
    p_inj = ensure_numeric(results.get('p_inj'), var_name="p_inj")
    p_with = ensure_numeric(results.get('p_with'), var_name="p_with")
    # Hőszivattyús változók (csak ha hss_flag=True)
    p_hss_in = results.get('p_hss_in')
    p_hss_out = results.get('p_hss_out')
    e_hss_stor = results.get('e_hss_stor')
    t_hss = results.get('t_hss')
    hss_flag = p_hss_in is not None and p_hss_out is not None and e_hss_stor is not None and t_hss is not None
    if hss_flag:
        p_hss_in = ensure_numeric(p_hss_in, var_name="p_hss_in")
        p_hss_out = ensure_numeric(p_hss_out, var_name="p_hss_out")
        e_hss_stor = ensure_numeric(e_hss_stor, var_name="e_hss_stor")
        t_hss = ensure_numeric(t_hss, var_name="t_hss")
    else:
        p_hss_in = p_hss_out = e_hss_stor = t_hss = None

    # Frissített results szótár
    results = {
        'p_inj_user': p_inj_user,
        'p_with_user': p_with_user,
        'p_cl_with': p_cl_with,
        'p_grid_in': p_grid_in,
        'p_grid_out': p_grid_out,
        'p_bess_in': p_bess_in,
        'p_bess_out': p_bess_out,
        'e_bess_stor': e_bess_stor,
        'p_elh_out': p_elh_out,
        'p_cl_grid': p_cl_grid,
        'p_cl_rec': p_cl_rec,
        'd_cl': d_cl,
        'p_hss_in': p_hss_in,
        'p_hss_out': p_hss_out,
        'e_hss_stor': e_hss_stor,
        't_hss': t_hss,
        'p_shared': p_shared,
        'p_inj': p_inj,
        'p_with': p_with
    }

    display_figures_two_users(results, p_pv, p_consumed, p_ut, hss_flag)


# Optimalizálás futtatása
results, status, objective, user_ids, num_vars, num_constraints = optimize_two_users(
    p_pv=na_p_pv, p_ut=na_p_ut, p_consumed=na_p_consumed,
    size_elh=np.array([2, 2.5]), size_bess=20, size_hss=np.array([100, 120]), run_lp=False, objective="environmental",
    gapRel=0.001
)

# Eredmények megjelenítése
extract_results_and_show_two_users(results, na_p_pv, na_p_consumed, na_p_ut)
