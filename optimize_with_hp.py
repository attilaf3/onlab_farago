import time
import numpy as np
import pulp
from pulp import LpStatusOptimal
import pandas as pd


def optimize_with_hp(p_pv, p_consumed, p_ut,
             T_env_vector, cop_hp_vector,
             dt=1,
             size_elh=None, size_bess=None, size_hss=None,
             size_hp=5,
             vol_hss_water=None,
             run_lp=False,
             T_set=21.0, T_init=21.0,
             **kwargs):
    """
    Extended CSC optimization including:
    - PV, ELH, BESS, HSS, grid
    - 3 kW levegő-víz hőszivattyú bináris vezérléssel (d_hp)
    - 5R/2C-s zónahőmérséklet-dinamika (T_zone, T_mass)
    - COP függés külső hőmérséklettől (cop_hp_vector)
    - Komfortfeltétel: T_zone >= T_set
    """
    # bemenet validálás
    assert (n_timestep := len(p_pv)) == len(p_consumed) == len(p_ut)

    print(f"Min COP: {cop_hp_vector.min()}, Max COP: {cop_hp_vector.max()}")

    time_set = range(n_timestep)
    n_timesteps_in_a_day = round(n_timestep / 365.)

    # Eredeti modell paraméterek
    eta_bess_in = kwargs.get('eta_bess_in', 0.98)
    eta_bess_out = kwargs.get('eta_bess_out', 0.96)
    eta_bess_stor = kwargs.get('eta_bess_stor', 0.995)
    t_bess_min = kwargs.get('t_bess_min', 2)
    soc_bess_min = kwargs.get('soc_bess_min', 0.2)
    soc_bess_max = kwargs.get('soc_bess_max', 1)
    t_hss_min_in = kwargs.get('t_hss_min', 4)
    t_hss_min_out = kwargs.get('t_hss_min', 1)
    c_hss = kwargs.get('c_hss', 0.00116667)  # kWh/kgK
    a_hss = kwargs.get('a_hss', 0.01275)
    T_env = kwargs.get('T_env', 20)  # °C
    T_max = kwargs.get('T_max', 55)
    T_in = kwargs.get('T_in', 12)  # °C
    T_min = T_in
    vol_hss_water = kwargs.get('vol_hss_water', 120)  # l/kg
    eta_elh = kwargs.get('eta_elh', 1)
    c_with = kwargs.get('c_with', 0.092)
    c_cl = kwargs.get('c_cl', 0.061)
    c_inj = kwargs.get('c_inj', 0.013)
    c_sh = kwargs.get('c_sh', 0.0024)
    objective = kwargs.get('objective', 'economic')

    # --- Paraméterek ---
    dt = 1  # [h]
    C_zone = 4.16  # [kWh/K]
    C_mass = 37.97  # [kWh/K]
    R_conv = 0.75  # [K/kW]
    R_rad = 0.2  # [K/kW]
    R_ve = 14.95  # [K/kW]
    R_ea = 80.13  # [K/kW]
    window_area = 15  # [m^2]
    g_window = 0.6  # [1]
    solar_gain_factor = 0.3
    C_zone = 2.5  # kWh/K
    C_mass = 20.0  # kWh/K
    R_conv = 0.4  # K/kW
    R_rad = 0.1  # K/kW
    R_ve = 7.0  # K/kW
    R_ea = 40.0  # K/kW
    # Napenergia‐nyereséghez szükséges vektorok
    solar_dir = kwargs.get('solar_radiation_direct')
    solar_dif = kwargs.get('solar_radiation_diffuse')

    # Globális horizontális irradiancia [kW/m2]
    G_hor = (solar_dir + solar_dif) / 1000.0
    # Sorozat [kW]
    Q_solar = list(g_window * window_area * G_hor * solar_gain_factor)

    # Solver parameters
    msg = kwargs.get('msg', True)  # print messages during optimization
    gapRel = kwargs.get('gapRel', None)  # relative gap values
    timeLimit = kwargs.get('timeLimit', None)  # time limit for optimization
    objective = kwargs.get('objective', "economic")  # objective function: economic or environmental
    assert objective in ("economic", "environmental"), "Objective must be either economic or environmental"

    # Get some useful variables
    elh_flag = size_elh is not None and size_elh > 0  # presence of ELH
    bess_flag = size_bess is not None and size_bess > 0  # presence of BESS
    cl_flag = p_ut is not None and any(p_ut > 0)  # presence of CL ---> presence of HSS
    hss_flag = size_hss is not None

    # LpProblem indítása
    prob = pulp.LpProblem("Opt_HP_Model", pulp.LpMinimize)

    # Useful parameters
    # fake-big-M parameter for maximum grid injections from the building
    M_grid_in = 1.1 * p_pv.max()  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    if bess_flag:
        M_grid_in += size_bess / t_bess_min
    # fake-big-M parameter for maximum grid withdrawals
    M_grid_out = 1.1 * p_consumed.max()
    M_cl_rec = p_pv.max()
    if bess_flag:
        M_grid_out += size_bess / t_bess_min
        M_cl_rec += size_bess / t_bess_min
    if cl_flag:
        # Good approximation of charge, discharge and storage efficiency of heater and heat storage
        M_grid_out += p_ut.max() * 1.1
    # big-M parameter needed to evaluate shared energy
    M_shared = 1.1 * max(M_grid_in, M_grid_out)
    M_hp = size_hp / min(cop_hp_vector)
    # ------------------------------------------------------------
    # ÚJ változók: hőszivattyú és zóna modell
    # d_hp = [pulp.LpVariable(f'd_hp_heat_{t}', cat=pulp.LpBinary) for t in time_set]
    p_hp_th = [pulp.LpVariable(f'p_hp_th{t}', lowBound=-size_hp, upBound=size_hp) for t in time_set]
    p_hp_el = [pulp.LpVariable(f'p_hp_el{t}', lowBound=0) for t in time_set]
    T_zone = [pulp.LpVariable(f'T_zone_{t}', lowBound=10, upBound=30) for t in time_set]
    T_mass = [pulp.LpVariable(f'T_mass_{t}', lowBound=0, upBound=50) for t in time_set]

    # ÚJ:
    d_hp_heat = [pulp.LpVariable(f'd_hp_heat_{t}', cat=pulp.LpBinary) for t in time_set]
    d_hp_cool = [pulp.LpVariable(f'd_hp_cool_{t}', cat=pulp.LpBinary) for t in time_set]
    # ------------------------------------------------------------

    # Eredeti változók (PV, CL, ELH, HSS, BESS, grid stb.)
    p_cl_with = [pulp.LpVariable(f'Pcl_with_{t}', lowBound=0) if cl_flag else 0 for t in time_set]
    p_cl_grid = [pulp.LpVariable(f'Pcl_grid_{t}', lowBound=0) if cl_flag else 0 for t in time_set]
    p_cl_rec = [pulp.LpVariable(f'Pcl_rec_{t}', lowBound=0) if cl_flag else 0 for t in time_set]
    d_cl = [pulp.LpVariable(f'Dcl_{t}', cat=pulp.LpBinary) if cl_flag and not run_lp else 0 for t in time_set]
    y_middle_day = [0] * 10 + [1] * 6 + [0] * 8

    p_elh_in = [pulp.LpVariable(f'Pelh_in_{t}', lowBound=0) if elh_flag else 0 for t in time_set]
    p_elh_out = [pulp.LpVariable(f'Pelh_out_{t}', lowBound=0) if elh_flag else 0 for t in time_set]

    p_hss_in = [pulp.LpVariable(f'Phss_in_{t}', lowBound=0) if hss_flag else 0 for t in time_set]
    p_hss_out = [pulp.LpVariable(f'Phss_out_{t}', lowBound=0) if hss_flag else 0 for t in time_set]
    e_hss_stor = [pulp.LpVariable(f'Ehss_stor_{t}', lowBound=0) if hss_flag else 0 for t in time_set]
    t_hss = [pulp.LpVariable(f'Thss_{t}', lowBound=T_min, upBound=T_max) if hss_flag else 0 for t in time_set]

    p_bess_in = [pulp.LpVariable(f'Pbess_in_{t}', lowBound=0) if bess_flag else 0 for t in time_set]
    p_bess_out = [pulp.LpVariable(f'Pbess_out_{t}', lowBound=0) if bess_flag else 0 for t in time_set]
    e_bess_stor = [pulp.LpVariable(f'Ebess_stor_{t}', lowBound=0) if bess_flag else 0 for t in time_set]
    d_bess = [pulp.LpVariable(f'Dbess_{t}', cat=pulp.LpBinary) if bess_flag and not run_lp else 0 for t in time_set]

    p_inj = [pulp.LpVariable(f'Pinj_{t}', lowBound=0) for t in time_set]
    p_with = [pulp.LpVariable(f'Pwith_{t}', lowBound=0) for t in time_set]
    d_grid = [pulp.LpVariable(f'Dgrid_{t}', cat=pulp.LpBinary) if not run_lp else 0 for t in time_set]
    p_grid_in = [pulp.LpVariable(f'Pgrid_in_{t}', lowBound=0) for t in time_set]
    p_grid_out = [pulp.LpVariable(f'Pgrid_out_{t}', lowBound=0) for t in time_set]
    # p_shared = [pulp.LpVariable(f'Psh_{t}', lowBound=0) for t in time_set]
    # y_shared = [pulp.LpVariable(f'Ysh_{t}', cat=pulp.LpBinary) if not run_lp else 0 for t in time_set]

    # Megszorítások minden időlépésre
    for t in time_set:
        k = (t + 1) % n_timestep

        # --- eredeti elektromos egyensúly --- (kiegészítve HP fogyasztással)
        prob += (p_pv[t] + p_grid_out[t] + p_bess_out[t]
                 == p_grid_in[t] + p_bess_in[t]
                 + p_consumed[t] + p_cl_with[t] + p_hp_el[t])
        prob += p_inj[t] == p_pv[t] + p_bess_out[t] - p_bess_in[t]
        prob += (p_with[t] == p_cl_with[t] + p_hp_el[t]
                 + p_consumed[t])

        # --------------------------------------------------------
        # HP-COP kapcsolat
        cop_cool = 2.8
        prob += p_hp_el[t] >= (1 / cop_hp_vector[t]) * p_hp_th[t]
        prob += p_hp_el[t] >= -(1 / cop_cool) * p_hp_th[t]

        # HP kapacitás bináris vezérlés
        # prob += p_hp_th[t] <= size_hp * d_hp[t]
        # prob += p_hp_th[t] >= -size_hp * d_hp[t]
        # prob += p_hp_el[t] <= M_hp * d_hp[t]

        # irányok kizárják egymást
        prob += d_hp_heat[t] + d_hp_cool[t] <= 1

        # teljesítmény irányok: fűtés és hűtés külön
        prob += p_hp_th[t] <= size_hp * d_hp_heat[t]
        prob += p_hp_th[t] >= -size_hp * d_hp_cool[t]

        # elektromos teljesítmény korlát mindkét irányban
        prob += p_hp_el[t] <= M_hp * (d_hp_heat[t] + d_hp_cool[t])

        # Komfort-feltétel: zónahőmérséklet
        prob += T_zone[t] >= T_set - 2.0  # Pl. 19 °C is elfogadható minimálisan
        prob += T_zone[t] <= T_set + 2.0  # Pl. 23 °C is elfogadható minimálisan

        # Fűtési tiltás meleg időben (pl. T_env > T_zone + 2 → ne fűtsön)
        prob += T_env_vector[t] <= T_zone[t] + 2 + (1 - d_hp_heat[t]) * 1000  # ha fűteni akar, legyen hidegebb

        # Hűtési tiltás hideg időben (pl. T_env < T_zone - 2 → ne hűtsön)
        prob += T_env_vector[t] >= T_zone[t] - 2 - (1 - d_hp_cool[t]) * 1000  # ha hűteni akar, legyen melegebb

        # Zóna és tömeg dinamikák (Euler diszkrét)
        if t < n_timestep - 1:
            # Zóna energiamérleg: belső tömeg, HP, szoláris nyereség
            prob += C_zone * (T_zone[k] - T_zone[t]) == dt * (
                    (T_mass[t] - T_zone[t]) * (1 / R_conv) + p_hp_th[t] + (T_mass[t] - T_zone[t]) * (1 / R_rad)
                    + (T_env_vector[t] - T_zone[t]) * (1 / R_ve) + Q_solar[t])
            prob += C_mass * (T_mass[k] - T_mass[t]) == dt * (
                    (T_zone[t] - T_mass[t]) * (1 / R_conv) + (T_env_vector[t] - T_mass[t]) * (1 / R_ea) + (
                    T_zone[t] - T_mass[t]) * (1 / R_rad))

            # # jobb oldal résztagjai külön változókban
            # conv_heat = (T_mass[t] - T_zone[t]) * (1 / R_conv)
            # rad_heat = (T_mass[t] - T_zone[t]) * (1 / R_rad)
            # infil_heat = (T_env_vector[t] - T_zone[t]) * (1 / R_ve)
            # solar_heat = Q_solar[t]
            # hp_heat = p_hp_th[t]
            #
            # # zóna energiamérleg
            # prob += C_zone * (T_zone[k] - T_zone[t]) == dt * (
            #         conv_heat + rad_heat + infil_heat + solar_heat + hp_heat
            # )
            # conv_rev = (T_zone[t] - T_mass[t]) * (1 / R_conv)
            # rad_rev = (T_zone[t] - T_mass[t]) * (1 / R_rad)
            # wall_loss = (T_env_vector[t] - T_mass[t]) * (1 / R_ea)
            #
            # prob += C_mass * (T_mass[k] - T_mass[t]) == dt * (
            #         conv_rev + rad_rev + wall_loss
            # )

        # --------------------------------------------------------

        # --- eredeti BESS, HSS, CL, grid megszorítások ---
        # battery energy storage system
        if bess_flag:
            # energy balance between time steps
            prob += e_bess_stor[k] - e_bess_stor[t] * eta_bess_stor == (
                    p_bess_in[t] * eta_bess_in - p_bess_out[t] * (1 / eta_bess_out)) * dt
            # maximum input and output power and mutual exclusivity
            if run_lp:
                prob += p_bess_in[t] <= (size_bess / t_bess_min)
                prob += p_bess_out[t] <= (size_bess / t_bess_min)
            else:
                prob += p_bess_in[t] <= d_bess[t] * (size_bess / t_bess_min)
                prob += p_bess_out[t] <= (1 - d_bess[t]) * (size_bess / t_bess_min)
            # maximum and minimum storable energy
            prob += e_bess_stor[t] <= (size_bess * soc_bess_max)
            prob += e_bess_stor[t] >= (size_bess * soc_bess_min)

        # heat storage system
        if hss_flag:
            prob += vol_hss_water * c_hss * (t_hss[k] - t_hss[t]) == p_hss_in[t] - p_hss_out[t] - a_hss * dt * (
                    t_hss[t] - T_env)

            # constitutive equation for electric heater
            prob += p_elh_out[t] == p_elh_in[t] * eta_elh

            # maximum power output
            prob += p_elh_in[t] <= size_elh
            # prob += p_elh_in[t] <= size_elh * (T_set - t_hss[t]) / (T_set - T_in)

            # prob += p_elh_in[t] >= size_elh * (T_set - t_hss[t]) / (T_set - T_in) * 0.8
            # prob += p_elh_in[t] >= size_elh * d_cl[t] * (T_set - T_min) / (T_set - T_in) * 0.8

            # heat-energy conversion
            prob += p_cl_with[t] == p_elh_in[t]
            prob += p_hss_in[t] == p_elh_out[t]
            prob += p_hss_out[t] == p_ut[t]
            prob += p_hss_out[k] <= vol_hss_water * c_hss * (t_hss[t] - T_in)

        # controlled load
        if cl_flag:
            # thermal hub, energy balance
            prob += p_cl_with[t] == p_cl_grid[t] + p_cl_rec[t]

            prob += p_cl_rec[t] <= p_pv[t] + p_bess_out[t]
            prob += p_cl_grid[t] <= p_grid_out[t]

            # maximum power of controlled loads
            if not run_lp and hss_flag:
                prob += p_cl_grid[t] <= M_grid_out
                prob += p_cl_rec[t] <= M_cl_rec * d_cl[t]
                prob += p_cl_with[t] <= M_grid_out + M_cl_rec

                # Controlled load switching: must be on for at least two consecutive time steps
                if t != time_set[0] and t != time_set[-1]:
                    # Switched on for at least two consecutive time steps
                    prob += d_cl[t + 1] >= d_cl[t] - d_cl[t - 1]

            if not hss_flag:
                prob += p_cl_with[t] == p_ut[t]

        # grid
        # maximum injected and withdrawn power and mutual exclusivity
        if not run_lp:
            prob += p_grid_in[t] <= d_grid[t] * M_grid_in
            prob += p_grid_out[t] <= (1 - d_grid[t]) * M_grid_out

        # linearization of shared energy definition
        # constraint on the shared energy, that must be smaller than both
        # the injections and the withdrawals of the virtual users
        # prob += p_shared[t] <= p_inj[t]
        # prob += p_shared[t] <= p_with[t]
        # constraint on the shared energy, that must also be equal to the
        # minimum between the two values.
        # when y_shared == 1: shared_power = p_inj, thanks to
        # this constraint and smaller-equal for the previous one.
        # when y_shared == 0, the other way around.
        # if not run_lp:
        #     prob += p_shared[t] >= p_inj[t] - M_shared * (1 - y_shared[t])
        #     prob += p_shared[t] >= p_with[t] - M_shared * y_shared[t]

        # Kezdeti feltételek a zóna modellhez
    prob += T_zone[0] == T_init
    prob += T_mass[0] == T_init

    # Minimum 2 órás működési ciklus hőszivattyúnál
    for t in range(1, n_timestep - 1):
        prob += d_hp_heat[t + 1] >= d_hp_heat[t] - d_hp_heat[t - 1]
        prob += d_hp_cool[t + 1] >= d_hp_cool[t] - d_hp_cool[t - 1]

    # Napi maximum 20 óra hőszivattyú működés (összesen fűtés + hűtés)
    for j in range(0, n_timestep, n_timesteps_in_a_day):
        prob += pulp.lpSum([d_hp_heat[t] + d_hp_cool[t] for t in range(j, j + n_timesteps_in_a_day)]) <= 20

    # --- eredeti napi CL megszorítások ---
    if cl_flag:
        for j in range(0, n_timestep, n_timesteps_in_a_day):
            if not run_lp:
                # At least (exactly) 8 hours of on-time for the switching
                prob += pulp.lpSum([d_cl[t] for t in range(j, j + n_timesteps_in_a_day)]) <= 12
                # At least 4 of it is in the middle of the day
                prob += pulp.lpSum([d_cl[t] * y_middle_day[t - j] for t in range(j, j + n_timesteps_in_a_day)]) >= 4

    # Célfüggvény kiegészítése: minimalizáljuk a hálózati villamos energiafelvételt
    if objective == 'economic':
        # az eredeti költségfüggvény kiegészítve HP fogyasztással (p_with tartalmazza p_hp_el)
        prob += pulp.lpSum([c_with * p_grid_out[t]
                            - c_inj * p_grid_in[t]
                            for t in time_set])
    else:
        prob += pulp.lpSum([p_grid_out[t] + p_grid_in[t]
                            for t in time_set])

    # LP debug
    if run_lp:
        prob.writeLP("debug_hp.lp")

    # Solve
    status = prob.solve(pulp.GUROBI_CMD(msg=True, gapRel=gapRel, timeLimit=timeLimit))
    print("Solver status:", pulp.LpStatus[status])
    if status != LpStatusOptimal:
        print("Optimalizáció sikertelen. Ellenőrizd a bemeneteket, COP értékeket, kapacitásokat stb.")
        return None, status, None, None, None

    # Extract values from pulp variables
    for t in time_set:
        p_inj[t] = pulp.value(p_inj[t])
        p_with[t] = pulp.value(p_with[t])
        p_bess_in[t] = pulp.value(p_bess_in[t])
        p_bess_out[t] = pulp.value(p_bess_out[t])
        e_bess_stor[t] = pulp.value(e_bess_stor[t])
        p_hss_in[t] = pulp.value(p_hss_in[t])
        p_hss_out[t] = pulp.value(p_hss_out[t])
        t_hss[t] = pulp.value(t_hss[t])
        # p_shared[t] = pulp.value(p_shared[t])
        p_cl_grid[t] = pulp.value(p_cl_grid[t])
        p_cl_rec[t] = pulp.value(p_cl_rec[t])
        p_cl_with[t] = pulp.value(p_cl_with[t])
        d_cl[t] = pulp.value(d_cl[t])
        p_grid_in[t] = pulp.value(p_grid_in[t])
        p_grid_out[t] = pulp.value(p_grid_out[t])
        # check
        d_bess[t] = pulp.value(d_bess[t])
        d_grid[t] = pulp.value(d_grid[t])
        # y_shared[t] = pulp.value(y_shared[t])
        d_hp_heat[t] = pulp.value(d_hp_heat[t])
        d_hp_cool[t] = pulp.value(d_hp_cool[t])
        T_zone[t] = pulp.value(T_zone[t])
        T_mass[t] = pulp.value(T_mass[t])
        # p_elh_in[t] = pulp.value(p_elh_in[t]) if isinstance(p_elh_in[t], pulp.LpVariable) else 0
        p_elh_in[t] = pulp.value(p_elh_in[t])
        w = pulp.value(p_elh_in[t])
        p_elh_in[t] = 0 if w is None else w
        # p_elh_out[t] = pulp.value(p_elh_out[t]) if isinstance(p_elh_out[t], pulp.LpVariable) else 0
        p_elh_out[t] = pulp.value(p_elh_out[t])
        v = pulp.value(p_elh_out[t])
        p_elh_out[t] = 0 if v is None else v
        p_hp_th[t] = pulp.value(p_hp_th[t])
        x = pulp.value(p_hp_th[t])
        p_hp_th[t] = 0 if x is None else x
        p_hp_el[t] = pulp.value(p_hp_el[t])
        y = pulp.value(p_hp_el[t])
        p_hp_el[t] = 0 if y is None else y

    e_hss_stor = vol_hss_water * c_hss * (np.array(t_hss) - T_env)
    p_shared = np.minimum(np.array(p_pv) + np.array(p_bess_out) - np.array(p_bess_in),
                          np.array(p_consumed) + np.array(p_cl_with) + np.array(p_hp_el))

    results = dict(p_inj=np.array(p_inj), p_with=np.array(p_with),
                   p_bess_in=np.array(p_bess_in), p_bess_out=np.array(p_bess_out), e_bess_stor=np.array(e_bess_stor),
                   p_elh_in=np.array(p_elh_in), p_elh_out=np.array(p_elh_out),
                   p_hss_in=np.array(p_hss_in), p_hss_out=np.array(p_hss_out), e_hss_stor=np.array(e_hss_stor),
                   t_hss=np.array(t_hss), p_cl_grid=np.array(p_cl_grid), p_cl_rec=np.array(p_cl_rec),
                   p_cl_with=np.array(p_cl_with),
                   p_shared=np.array(p_shared), p_grid_out=np.array(p_grid_out),
                   p_grid_in=np.array(p_grid_in), d_cl=np.array(d_cl), p_ue=np.array(p_consumed),
                   p_hp_th=np.array(p_hp_th), p_hp_el=np.array(p_hp_el), d_hp_heat=np.array(d_hp_heat),
                   d_hp_cool=np.array(d_hp_cool), T_zone=np.array(T_zone),
                   T_mass=np.array(T_mass), p_pv=p_pv, p_consumed=p_consumed, p_ut=p_ut)
    return results, status, pulp.value(prob.objective), prob.numVariables(), prob.numConstraints()


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
                                                                 size_elh=2, size_bess=5, size_hss=4, run_lp=False,
                                                                 objective="environmental", T_env_vector=T_env,
                                                                 cop_hp_vector=COP,
                                                                 solar_radiation_direct=solar_dir,
                                                                 solar_radiation_diffuse=solar_dif, gapRel=0.005)