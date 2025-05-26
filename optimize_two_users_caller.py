import pandas as pd
import numpy as np
from matplotlib import pyplot as plt, patheffects as pe
from optimize_two_users import optimize_two_users


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
    dt = 72  # Egy hét (168 óra)
    titles = ['Winter', 'Spring', 'Summer', 'Autumn']
    path_effect = lambda lw: [pe.Stroke(linewidth=1.5 * lw, foreground='w'), pe.Normal()]
    bar_kw = dict(width=0.8)  # Keskenyebb sávok
    plot_kw = dict(lw=3, path_effects=path_effect(3))
    area_kw = dict(alpha=0.6)

    for i, t0 in enumerate(t0s):
        tf = t0 + dt
        time = np.arange(t0, tf)

        # Ábra
        fig, (ax, ax_legend) = plt.subplots(1, 2, figsize=(18, 8), gridspec_kw={'width_ratios': [0.85, 0.15]})
        bottom_pos = np.zeros_like(time, dtype=float)
        bottom_neg = np.zeros_like(time, dtype=float)

        # Felfelé: termelés és kisütés
        ax.bar(time, na_p_pv[t0:tf, 0], bottom=bottom_pos, label=r'$P_\mathrm{pv,u0}$', color='lightgreen', **bar_kw)
        bottom_pos += na_p_pv[t0:tf, 0]
        ax.bar(time, na_p_pv[t0:tf, 1], bottom=bottom_pos, label=r'$P_\mathrm{pv,u1}$', color='green', **bar_kw)
        bottom_pos += na_p_pv[t0:tf, 1]
        ax.bar(time, results['p_bess_out'][t0:tf], bottom=bottom_pos, label=r'$P_\mathrm{bess,out}$', color='orange', **bar_kw)
        bottom_pos += results['p_bess_out'][t0:tf]
        ax.bar(time, results['p_grid_out'][t0:tf], bottom=bottom_pos, label=r'$P_\mathrm{grid,out}$', color='blue', **bar_kw)

        # Lefelé: igény és vételezés
        ax.bar(time, -na_p_consumed[t0:tf, 0], bottom=bottom_neg, label=r'$P_\mathrm{ue,u0}$', color='lightcoral',
               **bar_kw)
        bottom_neg -= na_p_consumed[t0:tf, 0]
        ax.bar(time, -na_p_consumed[t0:tf, 1], bottom=bottom_neg, label=r'$P_\mathrm{ue,u1}$', color='red', **bar_kw)
        bottom_neg -= na_p_consumed[t0:tf, 1]
        ax.bar(time, -results['p_cl_rec'][t0:tf, 0], bottom=bottom_neg, label=r'$P_\mathrm{cl,rec,u0}$', color='lightgrey',
                **bar_kw)
        bottom_neg -= results['p_cl_rec'][t0:tf, 0].astype(float)
        ax.bar(time, -results['p_cl_rec'][t0:tf, 1], bottom=bottom_neg, label=r'$P_\mathrm{cl,rec,u1}$', color='grey',
               **bar_kw)
        bottom_neg -= results['p_cl_rec'][t0:tf, 1].astype(float)
        ax.bar(time, -results['p_cl_grid'][t0:tf, 0].astype(float), bottom=bottom_neg, label=r'$P_\mathrm{cl,grid,u0}$',
               color='mediumorchid', **bar_kw)
        bottom_neg -= results['p_cl_grid'][t0:tf, 0].astype(float)
        ax.bar(time, -results['p_cl_grid'][t0:tf, 1].astype(float), bottom=bottom_neg, label=r'$P_\mathrm{cl,grid,u1}$',
               color='indigo', **bar_kw)
        bottom_neg -= results['p_cl_grid'][t0:tf, 1].astype(float)
        ax.bar(time, -results['p_bess_in'][t0:tf], bottom=bottom_neg, label=r'$P_\mathrm{bess,in}$', color='purple', **bar_kw)
        bottom_neg -= results['p_bess_in'][t0:tf].astype(float)
        ax.bar(time, -results['p_grid_in'][t0:tf], bottom=bottom_neg, label=r'$P_\mathrm{grid,in}$', color='navy', **bar_kw)

        # BESS SOC
        ax2 = ax.twinx()
        ax2.plot(time, results['e_bess_stor'][t0:tf], color='black', ls='--', label=r'$E_\mathrm{stor, bess}$')
        ax2.set_ylabel("Stored energy (kWh)", color='black')
        ax2.tick_params(axis='y', colors='black')
        ax2.spines['right'].set_color('black')

        # Beállítások
        ax.set_xlabel("Time (h)")
        ax.set_ylabel("Power (kW)")
        ax.set_title(f"{titles[i]} – Electric Hub", fontsize=14)
        handles, labels = ax.get_legend_handles_labels()
        ax_legend.legend(handles, labels, loc='center', fontsize=12)
        ax_legend.axis('off')
        ax.grid(True)
        plt.tight_layout()
        plt.show()

        # # Új ábra: Thermal node külön figure-ben, 3 subplottal (User1, User2, Legend)
        # fig_th, axes_th = plt.subplots(1, 3, figsize=(18, 6), gridspec_kw={'width_ratios': [0.4, 0.4, 0.2]},
        #                                sharex=True)
        # fig_th.suptitle(f"{titles[i]} – Thermal Node", fontsize=16)
        #
        # handles_all = []
        #
        # for u in range(2):
        #     ax_th = axes_th[u]
        #     bottom = np.zeros_like(time, dtype=float)
        #
        #     # Felfelé: CL grid, CL rec, HSS out
        #     ax_th.bar(time, results['p_cl_grid'][t0:tf, u], bottom=bottom, label=r'$P_\mathrm{cl,grid}$', **bar_kw)
        #     bottom += results['p_cl_grid'][t0:tf, u]
        #     ax_th.bar(time, results['p_cl_rec'][t0:tf, u], bottom=bottom, label=r'$P_\mathrm{cl,rec}$', **bar_kw)
        #     bottom += results['p_cl_rec'][t0:tf, u]
        #     if hss_flag:
        #         ax_th.bar(time, results['p_hss_out'][t0:tf, u], bottom=bottom, label=r'$P_\mathrm{hss,out}$', **bar_kw)
        #         bottom += results['p_hss_out'][t0:tf, u]
        #
        #     # Lefelé: ELH out, P_ut, HSS in
        #     bottom = np.zeros_like(time, dtype=float)
        #     ax_th.bar(time, -results['p_elh_out'][t0:tf, u], bottom=bottom, label=r'$P_\mathrm{elh,out}$', **bar_kw)
        #     bottom -= results['p_elh_out'][t0:tf, u]
        #     ax_th.bar(time, -p_ut[t0:tf, u], bottom=bottom, label=r'$P_\mathrm{ut}$', **bar_kw)
        #     bottom -= p_ut[t0:tf, u]
        #     if hss_flag:
        #         ax_th.bar(time, -results['p_hss_in'][t0:tf, u], bottom=bottom, label=r'$P_\mathrm{hss,in}$', **bar_kw)
        #         bottom -= results['p_hss_in'][t0:tf, u]
        #
        #     # Tároló SOC
        #     if hss_flag:
        #         ax_tw = ax_th.twinx()
        #         ax_tw.plot(time, results['e_hss_stor'][t0:tf, u], color='black', ls='--', label=r'$E_\mathrm{stor, hss}$')
        #         ax_tw.set_ylabel("Stored energy (kWh)", color='black')
        #         ax_tw.tick_params(axis='y', colors='black')
        #         ax_tw.spines['right'].set_color('black')
        #
        #     # Vezérlőjel
        #     ax_td = ax_th.twinx()
        #     ax_td.plot(time, results['d_cl'][t0:tf], '.', ls='-', color='lightgrey')
        #     ax_td.axis('off')
        #
        #     ax_th.set_title(f"User {u + 1}")
        #     ax_th.set_xlabel("Time (h)")
        #     ax_th.set_ylabel("Power (kW)")
        #     ax_th.grid(True)
        #
        #     if u == 0:
        #         handles_all, labels_all = ax_th.get_legend_handles_labels()
        #
        # # Jelmagyarázat a jobb oldali subplotba
        # axes_th[2].legend(handles_all, labels_all, loc='center', fontsize=12)
        # axes_th[2].axis('off')
        #
        # plt.tight_layout()
        # plt.show()

        # Két külön thermal diagram userenként
        for u in range(2):
            fig_th, (ax_th, legend_th) = plt.subplots(1, 2, figsize=(18, 8), gridspec_kw={'width_ratios': [0.85, 0.15]})
            fig_th.suptitle(f"{titles[i]} – Thermal Node – User {u}", fontsize=16)

            bottom_pos = np.zeros_like(time, dtype=float)
            bottom_neg = np.zeros_like(time, dtype=float)

            # Felső oldal: termelés (cl_grid, cl_rec, hss_out)
            ax_th.bar(time, results['p_cl_grid'][t0:tf, u], bottom=bottom_pos,
                      label=r'$P_\mathrm{cl,grid}$', color='mediumorchid', **bar_kw)
            bottom_pos += results['p_cl_grid'][t0:tf, u]
            ax_th.bar(time, results['p_cl_rec'][t0:tf, u], bottom=bottom_pos,
                      label=r'$P_\mathrm{cl,rec}$', color='grey', **bar_kw)
            bottom_pos += results['p_cl_rec'][t0:tf, u]
            ax_th.bar(time, results['p_hss_out'][t0:tf, u], bottom=bottom_pos,
                      label=r'$P_\mathrm{hss,out}$', color='orange', **bar_kw)

            # Alsó oldal: igények (elh_out, ut)
            ax_th.bar(time, -results['p_elh_out'][t0:tf, u], bottom=bottom_neg,
                      label=r'$P_\mathrm{elh,out}$', color='teal', **bar_kw)
            bottom_neg -= results['p_elh_out'][t0:tf, u]
            ax_th.bar(time, -p_ut[t0:tf, u], bottom=bottom_neg,
                      label=r'$P_\mathrm{ut}$', color='firebrick', **bar_kw)

            # SOC overlay
            ax_soc = ax_th.twinx()
            ax_soc.plot(time, results['e_hss_stor'][t0:tf, u], color='black', ls='--', label=r'$E_\mathrm{stor,hss}$')
            ax_soc.set_ylabel("Stored energy (kWh)", color='black')
            ax_soc.tick_params(axis='y', colors='black')
            ax_soc.spines['right'].set_color('black')

            # Vezérlés overlay
            ax_ctrl = ax_th.twinx()
            ax_ctrl.plot(time, results['d_cl'][t0:tf], '.', ls='-', color='lightgrey', label=r"$\mathrm{control\ signal\ (on/off)}$")
            ax_ctrl.axis('off')

            # Tengely és feliratok
            ax_th.set_xlabel("Time (h)")
            ax_th.set_ylabel("Power (kW)")
            ax_th.set_title(f"Thermal balance – User {u}")
            ax_th.grid(True)

            # Jelmagyarázat
            handles, labels = ax_th.get_legend_handles_labels()
            legend_th.legend(handles, labels, loc='center', fontsize=12)
            legend_th.axis('off')

            plt.tight_layout()
            plt.show()

        # CSC
        fig, (ax, legend_ax) = plt.subplots(ncols=2, figsize=(18, 8), gridspec_kw=dict(width_ratios=[0.8, 0.2]))
        fig.suptitle(f"{titles[i]} – CSC", fontsize=fontsize)

        # Interpoláció a sima grafikonhoz
        t_plot = np.linspace(t0, tf, 1000)
        f_plot = lambda x: np.interp(t_plot, time, x)
        p_inj_plot = f_plot(results['p_inj'][t0:tf])
        p_with_plot = f_plot(results['p_with'][t0:tf])
        p_shared_plot = np.minimum(p_inj_plot, p_with_plot)

        # Görbék
        ax.plot(t_plot, p_inj_plot, label=r'$P_\mathrm{inj}$', color='tab:red', **plot_kw)
        ax.plot(t_plot, p_with_plot, label=r'$P_\mathrm{with}$', color='tab:blue', **plot_kw)
        ax.plot(t_plot, p_shared_plot, label=r'$P_\mathrm{shared}$', color='tab:green', **plot_kw)

        # Kitöltött területek
        ax.fill_between(t_plot, p_shared_plot, p_with_plot, where=p_with_plot > p_shared_plot,
                        label=r'E$_\mathrm{\leftarrow grid}$', color='tab:blue', **area_kw)
        ax.fill_between(t_plot, p_shared_plot, p_inj_plot, where=p_inj_plot > p_shared_plot,
                        label=r'E$_\mathrm{\rightarrow grid}$', color='tab:red', **area_kw)
        ax.fill_between(t_plot, 0, p_shared_plot, where=p_shared_plot > 0,
                        label=r'E$_\mathrm{shared}$', color='tab:green', alpha=0.3)

        # Beállítások
        ax.set_xlabel("Time (h)", fontsize=fontsize)
        ax.set_ylabel("Power (kW)", fontsize=fontsize)
        ax.set_title("Community exchange", fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)
        ax.grid()

        # Jelmagyarázat
        handles1, labels1 = ax.get_legend_handles_labels()
        legend_ax.legend(handles1, labels1, loc='center', fontsize=fontsize, ncol=2)
        legend_ax.axis('off')

        plt.tight_layout()
        plt.show()

def display_figures_hub_csc(results, p_pv, p_consumed, p_ut, hss_flag=True):
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    import numpy as np
    from matplotlib.gridspec import GridSpec

    figsize = (24, 16)
    fontsize = 15
    t0s = [0, 2184, 4368, 6552]
    dt = 72
    titles = ['Winter', 'Spring', 'Summer', 'Autumn']
    path_effect = lambda lw: [pe.Stroke(linewidth=1.5 * lw, foreground='w'), pe.Normal()]
    bar_kw = dict(width=0.8)
    plot_kw = dict(lw=3, path_effects=path_effect(3))
    area_kw = dict(alpha=0.6)

    for i, t0 in enumerate(t0s):
        tf = t0 + dt
        time = np.arange(t0, tf)

        # GridSpec használata az elrendezés pontos meghatározásához (etalon alapján)
        fig = plt.figure(figsize=(18, 10))

        gs = GridSpec(nrows=2, ncols=2, width_ratios=[0.8, 0.2], height_ratios=[1, 1], hspace=0.3, wspace=0.25)

        # --- ELECTRIC HUB (bal felső) ---
        ax = fig.add_subplot(gs[0, 0])
        bottom_pos = np.zeros_like(time, dtype=float)
        bottom_neg = np.zeros_like(time, dtype=float)

        # Felfelé: termelés és kisütés
        h1 = ax.bar(time, p_pv[t0:tf, 0], bottom=bottom_pos, label=r'$P_\mathrm{pv,u0}$', color='lightgreen', **bar_kw)
        bottom_pos += p_pv[t0:tf, 0]
        h2 = ax.bar(time, p_pv[t0:tf, 1], bottom=bottom_pos, label=r'$P_\mathrm{pv,u1}$', color='green', **bar_kw)
        bottom_pos += p_pv[t0:tf, 1]
        h3 = ax.bar(time, results['p_bess_out'][t0:tf], bottom=bottom_pos, label=r'$P_\mathrm{bess,out}$', color='orange', **bar_kw)
        bottom_pos += results['p_bess_out'][t0:tf]
        h4 = ax.bar(time, results['p_grid_out'][t0:tf], bottom=bottom_pos, label=r'$P_\mathrm{grid,out}$', color='blue', **bar_kw)

        # Lefelé: fogyasztás és töltés
        h5 = ax.bar(time, -p_consumed[t0:tf, 0], bottom=bottom_neg, label=r'$P_\mathrm{ue,u0}$', color='lightcoral', **bar_kw)
        bottom_neg -= p_consumed[t0:tf, 0]
        h6 = ax.bar(time, -p_consumed[t0:tf, 1], bottom=bottom_neg, label=r'$P_\mathrm{ue,u1}$', color='red', **bar_kw)
        bottom_neg -= p_consumed[t0:tf, 1]
        h7 = ax.bar(time, -results['p_cl_rec'][t0:tf, 0], bottom=bottom_neg, label=r'$P_\mathrm{cl,rec,u0}$', color='lightgrey', **bar_kw)
        bottom_neg -= results['p_cl_rec'][t0:tf, 0].astype(float)
        h8 = ax.bar(time, -results['p_cl_rec'][t0:tf, 1], bottom=bottom_neg, label=r'$P_\mathrm{cl,rec,u1}$', color='grey', **bar_kw)
        bottom_neg -= results['p_cl_rec'][t0:tf, 1].astype(float)
        h9 = ax.bar(time, -results['p_cl_grid'][t0:tf, 0], bottom=bottom_neg, label=r'$P_\mathrm{cl,grid,u0}$', color='mediumorchid', **bar_kw)
        bottom_neg -= results['p_cl_grid'][t0:tf, 0].astype(float)
        h10 = ax.bar(time, -results['p_cl_grid'][t0:tf, 1], bottom=bottom_neg, label=r'$P_\mathrm{cl,grid,u1}$', color='indigo', **bar_kw)
        bottom_neg -= results['p_cl_grid'][t0:tf, 1].astype(float)
        h11 = ax.bar(time, -results['p_bess_in'][t0:tf], bottom=bottom_neg, label=r'$P_\mathrm{bess,in}$', color='purple', **bar_kw)
        bottom_neg -= results['p_bess_in'][t0:tf]
        h12 = ax.bar(time, -results['p_grid_in'][t0:tf], bottom=bottom_neg, label=r'$P_\mathrm{grid,in}$', color='navy', **bar_kw)

        # BESS SOC
        axtw = ax.twinx()
        h13, = axtw.plot(time, results['e_bess_stor'][t0:tf], color='black', ls='--', label=r'$E_\mathrm{stor,bess}$')
        axtw.set_ylabel("Stored energy (kWh)", color='black')
        axtw.tick_params(axis='y', colors='black')
        axtw.spines['right'].set_color('black')

        # Beállítások
        ax.set_xlabel("Time (h)", fontsize=fontsize)
        ax.set_ylabel("Power (kW)", fontsize=fontsize)
        ax.set_title("Electric Hub", fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)
        ax.grid(True)

        # Jelmagyarázat (jobb felső)
        ax_legend = fig.add_subplot(gs[0, 1])
        ax_legend.axis('off')
        ax_legend.legend([h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13],
                         [r'$P_\mathrm{pv,u0}$', r'$P_\mathrm{pv,u1}$', r'$P_\mathrm{bess,out}$', r'$P_\mathrm{grid,out}$',
                          r'$P_\mathrm{ue,u0}$', r'$P_\mathrm{ue,u1}$', r'$P_\mathrm{cl,rec,u0}$', r'$P_\mathrm{cl,rec,u1}$',
                          r'$P_\mathrm{cl,grid,u0}$', r'$P_\mathrm{cl,grid,u1}$', r'$P_\mathrm{bess,in}$', r'$P_\mathrm{grid,in}$',
                          r'$E_\mathrm{stor,bess}$'],
                         loc='center', fontsize=16, ncol=2)

        # --- CSC (bal alsó) ---
        ax_csc = fig.add_subplot(gs[1, 0])
        t_plot = np.linspace(t0, tf, 1000)
        interp = lambda x: np.interp(t_plot, time, x)

        p_inj_plot = interp(results['p_inj'][t0:tf])
        p_with_plot = interp(results['p_with'][t0:tf])
        p_shared_plot = np.minimum(p_inj_plot, p_with_plot)

        line1, = ax_csc.plot(t_plot, p_inj_plot, label=r'$P_\mathrm{inj}$', color='tab:red', **plot_kw)
        line2, = ax_csc.plot(t_plot, p_with_plot, label=r'$P_\mathrm{with}$', color='tab:blue', **plot_kw)
        line3, = ax_csc.plot(t_plot, p_shared_plot, label=r'$P_\mathrm{shared}$', color='tab:green', **plot_kw)

        ax_csc.fill_between(t_plot, p_shared_plot, p_with_plot, where=p_with_plot > p_shared_plot,
                            label=r'E$_\mathrm{\leftarrow grid}$', color='tab:blue', **area_kw)
        ax_csc.fill_between(t_plot, p_shared_plot, p_inj_plot, where=p_inj_plot > p_shared_plot,
                            label=r'E$_\mathrm{\rightarrow grid}$', color='tab:red', **area_kw)
        ax_csc.fill_between(t_plot, 0, p_shared_plot, where=p_shared_plot > 0,
                            label=r'E$_\mathrm{shared}$', color='tab:green', alpha=0.3)

        # Beállítások
        ax_csc.set_xlabel("Time (h)", fontsize=fontsize)
        ax_csc.set_ylabel("Power (kW)", fontsize=fontsize)
        ax_csc.set_title("Community Exchange", fontsize=fontsize)
        ax_csc.tick_params(labelsize=fontsize)
        ax_csc.grid(True)

        # Jelmagyarázat (jobb alsó)
        ax_csc_legend = fig.add_subplot(gs[1, 1])
        ax_csc_legend.axis('off')
        ax_csc_legend.legend([line1, line2, line3] + ax_csc.get_legend_handles_labels()[0][3:],
                             [r'$P_\mathrm{inj}$', r'$P_\mathrm{with}$', r'$P_\mathrm{shared}$',
                              r'E$_\mathrm{\leftarrow grid}$', r'E$_\mathrm{\rightarrow grid}$', r'E$_\mathrm{shared}$'],
                             loc='center', fontsize=16, ncol=2)

        # Cím pozícionálása felül középen, az etalon alapján
        fig.suptitle(f"{titles[i]} – Community Energy System", fontsize=fontsize)

        # Tight_layout használata az etalonhoz hasonlóan
        plt.tight_layout()

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
    # p_elh_out = ensure_numeric(results.get('p_elh_out'), var_name="p_elh_out")
    p_elh_out = ensure_numeric(results.get('p_cl_grid') + results.get('p_cl_rec'), var_name="p_elh_out")
    p_cl_grid = ensure_numeric(results.get('p_cl_grid'), var_name="p_cl_grid")
    p_cl_rec = ensure_numeric(results.get('p_cl_rec'), var_name="p_cl_rec")
    d_cl = ensure_numeric(results.get('d_cl'), var_name="d_cl")
    p_inj = ensure_numeric(results.get('p_inj'), var_name="p_inj")
    p_with = ensure_numeric(results.get('p_with'), var_name="p_with")
    p_shared = ensure_numeric(results.get('p_shared'), var_name="p_shared")
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
        'p_inj': p_inj,
        'p_with': p_with,
        'p_shared': p_shared
    }

    # display_figures_two_users(results, p_pv, p_consumed, p_ut, hss_flag)
    display_figures_hub_csc(results, p_pv, p_consumed, p_ut, hss_flag)


# Optimalizálás futtatása
results, status, objective, user_ids, num_vars, num_constraints = optimize_two_users(
    p_pv=na_p_pv, p_ut=na_p_ut, p_consumed=na_p_consumed,
    size_elh=np.array([2, 2.5]), size_bess=12, size_hss=np.array([4, 5]), run_lp=False, objective="environmental",
    gapRel=0.005
)

# Eredmények megjelenítése
extract_results_and_show_two_users(results, na_p_pv, na_p_consumed, na_p_ut)


# # Paraméterek
# pv_ratios = [0.25, 0.5, 1.0, 1.5, 2.0]
# bess_sizes = [0, 4, 8, 12, 16]
# shape = (len(pv_ratios), len(bess_sizes))
#
# sc_profile = np.zeros(shape)
# ss_profile = np.zeros(shape)
# sc_optimal = np.zeros(shape)
# ss_optimal = np.zeros(shape)
#
# # Forrásadatok
# p_consumed_total = na_p_consumed.sum(axis=1)  # P_ue összesítve
# p_ut_total = na_p_ut.sum(axis=1)              # P_ut összesítve
# base_pv_total = na_p_pv.sum(axis=1)           # P_pv összesítve
#
# for i, pv_ratio in enumerate(pv_ratios):
#     for j, bess_size in enumerate(bess_sizes):
#         # --- PROFILE logika ---
#         p_pv = base_pv_total * pv_ratio
#         p_ue = p_consumed_total + p_ut_total
#         bess_soc = np.zeros(len(p_pv))
#         bess_out = np.zeros(len(p_pv))
#         bess_in = np.zeros(len(p_pv))
#
#         for t in range(len(p_pv)):
#             net = p_pv[t] - p_ue[t]
#             if net >= 0:
#                 charge = min(net, bess_size - (bess_soc[t-1] if t > 0 else 0))
#                 bess_soc[t] = (bess_soc[t-1] if t > 0 else 0) + charge
#                 bess_in[t] = charge
#             else:
#                 discharge = min(-net, bess_soc[t-1] if t > 0 else 0)
#                 bess_soc[t] = (bess_soc[t-1] if t > 0 else 0) - discharge
#                 bess_out[t] = discharge
#
#         p_inj_profile = p_pv + bess_out
#         p_with_profile = p_ue
#         shared_profile = np.minimum(p_inj_profile, p_with_profile)
#
#         sc_profile[i, j] = np.sum(shared_profile) / np.sum(p_inj_profile) if np.sum(p_inj_profile) > 0 else 0
#         ss_profile[i, j] = np.sum(shared_profile) / np.sum(p_with_profile) if np.sum(p_with_profile) > 0 else 0
#
#         # --- OPTIMAL logika ---
#         p_pv_2user = na_p_pv * pv_ratio
#         results, *_ = optimize_two_users(
#             p_pv=p_pv_2user,
#             p_consumed=na_p_consumed,
#             p_ut=na_p_ut,
#             size_elh=[2, 2.5],
#             size_bess=bess_size,
#             size_hss=[4, 4],
#             vol_hss_water=[120, 160],
#             dt=1,
#             msg=False,
#             objective="environmental",
#             gapRel=0.01
#         )
#
#         p_inj_opt = ensure_numeric(results['p_inj'], var_name='p_inj')
#         p_with_opt = ensure_numeric(results['p_with'], var_name='p_with')
#         p_shared_opt = ensure_numeric(results['p_shared'], var_name='p_shared')
#
#         sc_optimal[i, j] = np.sum(p_shared_opt) / np.sum(p_inj_opt) if np.sum(p_inj_opt) > 0 else 0
#         ss_optimal[i, j] = np.sum(p_shared_opt) / np.sum(p_with_opt) if np.sum(p_with_opt) > 0 else 0
#
# import seaborn as sns
# import matplotlib.pyplot as plt
#
# def plot_heatmap(data, title):
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(data, annot=True, fmt=".2f", cmap="YlGnBu",
#                 xticklabels=bess_sizes, yticklabels=pv_ratios)
#     plt.xlabel("BESS size [kWh]")
#     plt.ylabel("PV ratio")
#     plt.title(title)
#     plt.tight_layout()
#     plt.show()
#
# plot_heatmap(sc_profile, "SCI – Profile Scenario (Two Users)")
# plot_heatmap(ss_profile, "SSI – Profile Scenario (Two Users)")
# plot_heatmap(sc_optimal, "SCI – Optimal Scenario (Two Users)")
# plot_heatmap(ss_optimal, "SSI – Optimal Scenario (Two Users)")
