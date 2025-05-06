import time
import numpy as np
import pulp
from pulp import LpStatusOptimal
import pandas as pd


def compute_thermal_params(
        floor_area, volume,
        wall_area, roof_area, window_area,
        U_wall, U_roof, U_window,
        ACH, wall_thickness, roof_thickness,
        wall_density, roof_density,
        c_air, air_density, c_wall
):
    """
    Számolja a zóna (5R/2C) paramétereit a geometriából és anyagadatokból.
    Visszaadja:
      C_zone  - beltéri levegő és berendezések hőtárolási kapacitása [J/K]
      C_mass  - szerkezeti tömeg hőtárolási kapacitása [J/K]
      R_conv  - belső felületés levegő közti konvekciós ellenállás [K/kW]
      R_ve    - zóna és külső környezet összesített hőátbocsátása (ablak+infiltráció) [K/kW]
      R_ea    - szerkezet és külső környezet vezetési ellenállása (fal+tető) [K/kW]
      R_rad   - sugárzási ellenállás felületek között [K/kW]
    """
    # beltéri levegő + bútorok kapacitás
    mass_air = air_density * volume  # kg
    C_zone = mass_air * c_air  # J/K
    # szerkezeti tömeg kapacitás (falak + födém)
    wall_mass = wall_density * wall_area * wall_thickness  # kg
    roof_mass = roof_density * roof_area * roof_thickness  # kg
    C_mass = (wall_mass + roof_mass) * c_wall  # J/K
    # hőellenállások
    R_wall = 1.0 / (U_wall * wall_area)  # K/kW
    R_roof = 1.0 / (U_roof * roof_area)
    R_window = 1.0 / (U_window * window_area)
    # szellőzés és ablakinfiltráció
    m_dot = ACH * mass_air / 3600.0  # kg/s
    U_infil = m_dot * c_air / 1000.0  # kW/K
    R_infil = 1.0 / U_infil if U_infil > 0 else 1e6
    # belső konvekció
    interior_area = wall_area + roof_area + floor_area  # m2
    h_int = 3.0  # W/(m2 K)
    R_conv = 1.0 / (h_int * interior_area)  # K/kW
    # zóna <-> környezet (ablak+infiltráció párhuzamosan)
    R_ve = (R_window * R_infil) / (R_window + R_infil)
    # szerkezet <-> külső (fal+tető sorosan)
    R_ea = R_wall + R_roof
    # sugárzási ellenállás vélelmezett értéke
    R_rad = 0.20
    return C_zone, C_mass, R_conv, R_ve, R_ea, R_rad


def cop_hp(T_env):
    return 0.05 * T_env + 2.5


def optimize(p_pv, p_consumed, p_ut,
             T_env_vector, cop_hp_vector,
             dt=1,
             size_elh=None, size_bess=None, size_hss=None,
             size_hp=6.0,
             vol_hss_water=None,
             run_lp=False,
             T_set=21.0, T_init=21.0, absorption_coeff=0.7,
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
    dt_s = dt * 3600.0

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
    vol_hss_water = kwargs.get('vol_hss_water', 3200)  # l/kg
    eta_elh = kwargs.get('eta_elh', 1)
    c_with = kwargs.get('c_with', 0.092)
    c_cl = kwargs.get('c_cl', 0.061)
    c_inj = kwargs.get('c_inj', 0.013)
    c_sh = kwargs.get('c_sh', 0.0024)
    objective = kwargs.get('objective', 'economic')

    # Geometriai és anyagparaméterek lekérdezése
    floor_area = kwargs.get('floor_area', 100.0)  # m2
    volume = kwargs.get('volume', 250.0)  # m3
    wall_area = kwargs.get('wall_area', 100.0)  # m2
    roof_area = kwargs.get('roof_area', 100.0)  # m2
    window_area = kwargs.get('window_area', 15.0)  # m2
    U_wall = kwargs.get('U_wall', 0.25)  # W/(m2K)
    U_roof = kwargs.get('U_roof', 0.25)  # W/(m2K)
    U_window = kwargs.get('U_window', 1.20)  # W/(m2K)
    ACH = kwargs.get('ACH', 0.50)  # 1/h
    wall_thickness = kwargs.get('wall_thickness', 0.30)  # m
    roof_thickness = kwargs.get('roof_thickness', 0.20)  # m
    wall_density = kwargs.get('wall_density', 1800.0)  # kg/m3
    roof_density = kwargs.get('roof_density', 1800.0)  # kg/m3
    c_air = kwargs.get('c_air', 1005.0)  # J/(kgK)
    air_density = kwargs.get('air_density', 1.20)  # kg/m3
    c_wall = kwargs.get('c_wall', 840.0)  # J/(kgK)

    # 5R/2C paraméterek előállítása
    C_zone, C_mass, R_conv, R_ve, R_ea, R_rad = compute_thermal_params(
        floor_area, volume,
        wall_area, roof_area, window_area,
        U_wall, U_roof, U_window,
        ACH, wall_thickness, roof_thickness,
        wall_density, roof_density,
        c_air, air_density, c_wall
    )

    q_solar = absorption_coeff * np.array(p_pv)
    cop_hp_vector = cop_hp(np.array(T_env_vector))

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
    d_hp = [pulp.LpVariable(f'd_hp_heat_{t}', cat=pulp.LpBinary) for t in time_set]
    p_hp_th = [pulp.LpVariable(f'p_hp_th{t}', lowBound=0, upBound=size_hp) for t in time_set]
    p_hp_el = [pulp.LpVariable(f'p_hp_el{t}', lowBound=0) for t in time_set]
    T_zone = [pulp.LpVariable(f'T_zone_{t}', lowBound=10, upBound=30) for t in time_set]
    T_mass = [pulp.LpVariable(f'T_mass_{t}', lowBound=0, upBound=50) for t in time_set]

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
    p_shared = [pulp.LpVariable(f'Psh_{t}', lowBound=0) for t in time_set]
    y_shared = [pulp.LpVariable(f'Ysh_{t}', cat=pulp.LpBinary) if not run_lp else 0 for t in time_set]

    # Megszorítások minden időlépésre
    for t in time_set:
        k = (t + 1) % n_timestep

        # --- eredeti elektromos egyensúly --- (kiegészítve HP fogyasztással)
        prob += (p_pv[t] + p_grid_out[t] + p_bess_out[t]
                 == p_grid_in[t] + p_bess_in[t]
                 + p_consumed[t] + p_cl_with[t] + p_hp_el[t])
        prob += p_inj[t] == p_pv[t] + p_bess_out[t]
        prob += (p_with[t] == p_cl_with[t]
                 + p_consumed[t] + p_bess_in[t] + p_hp_el[t])

        # --------------------------------------------------------
        # HP-COP kapcsolat
        prob += p_hp_el[t] == p_hp_th[t] * (1 / cop_hp_vector[t])

        # HP kapacitás bináris vezérlés
        prob += p_hp_th[t] <= size_hp * d_hp[t]

        # Komfort-feltétel: zónahőmérséklet
        prob += T_zone[t] >= T_set - 2.0  # Pl. 19 °C is elfogadható minimálisan
        prob += T_zone[t] <= T_set + 2.0  # Pl. 23 °C is elfogadható minimálisan

        prob += p_hp_el[t] <= M_hp * d_hp[t]

        # Zóna és tömeg dinamikák (Euler diszkrét)
        if t < n_timestep - 1:
            # Zóna energiamérleg: belső tömeg, HP, szoláris nyereség
            # TODO: q_solar nélkül?
            prob += C_zone * (T_zone[k] - T_zone[t]) == dt_s * (
                    (T_mass[t] - T_zone[t]) / R_conv + p_hp_th[t] + q_solar[t]
            )
            prob += C_mass * (T_mass[k] - T_mass[t]) == dt_s * (
                    (T_zone[t] - T_mass[t]) / R_conv + (T_env_vector[t] - T_mass[t]) / R_ea)
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
        prob += p_shared[t] <= p_inj[t]
        prob += p_shared[t] <= p_with[t]
        # constraint on the shared energy, that must also be equal to the
        # minimum between the two values.
        # when y_shared == 1: shared_power = p_inj, thanks to
        # this constraint and smaller-equal for the previous one.
        # when y_shared == 0, the other way around.
        if not run_lp:
            prob += p_shared[t] >= p_inj[t] - M_shared * (1 - y_shared[t])
            prob += p_shared[t] >= p_with[t] - M_shared * y_shared[t]

        # Kezdeti feltételek a zóna modellhez
    prob += T_zone[0] == T_init
    prob += T_mass[0] == T_init

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
                            - c_sh * p_shared[t]
                            for t in time_set])
    else:
        prob += pulp.lpSum([p_grid_out[t] + p_grid_in[t]
                            for t in time_set])

    # LP debug
    if run_lp:
        prob.writeLP("debug_hp.lp")

    # Solve
    status = prob.solve(pulp.GUROBI_CMD(msg=True))
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
        p_elh_in[t] = pulp.value(p_elh_in[t])
        p_elh_out[t] = pulp.value(p_elh_out[t])
        t_hss[t] = pulp.value(t_hss[t])
        p_shared[t] = pulp.value(p_shared[t])
        p_cl_grid[t] = pulp.value(p_cl_grid[t])
        p_cl_rec[t] = pulp.value(p_cl_rec[t])
        p_cl_with[t] = pulp.value(p_cl_with[t])
        d_cl[t] = pulp.value(d_cl[t])
        p_grid_in[t] = pulp.value(p_grid_in[t])
        p_grid_out[t] = pulp.value(p_grid_out[t])
        # check
        d_bess[t] = pulp.value(d_bess[t])
        d_grid[t] = pulp.value(d_grid[t])
        y_shared[t] = pulp.value(y_shared[t])
        p_elh_in[t] = pulp.value(p_elh_in[t]) if isinstance(p_elh_in[t], pulp.LpVariable) else 0
        p_elh_out[t] = pulp.value(p_elh_out[t]) if isinstance(p_elh_out[t], pulp.LpVariable) else 0
        p_hp_th[t] = pulp.value(p_hp_th[t])
        p_hp_el[t] = pulp.value(p_hp_el[t])
        d_hp[t] = pulp.value(d_hp[t])
        T_zone[t] = pulp.value(T_zone[t])
        T_mass[t] = pulp.value(T_mass[t])

    p_elh_in = [0 if v is None else v for v in p_elh_in]
    p_elh_out = [0 if v is None else v for v in p_elh_out]
    e_hss_stor = vol_hss_water * c_hss * (np.array(t_hss) - T_env) / dt

    results = dict(p_inj=np.array(p_inj), p_with=np.array(p_with),
                   p_bess_in=np.array(p_bess_in), p_bess_out=np.array(p_bess_out), e_bess_stor=np.array(e_bess_stor),
                   p_elh_in=np.array(p_elh_in), p_elh_out=np.array(p_elh_out),
                   p_hss_in=np.array(p_hss_in), p_hss_out=np.array(p_hss_out), e_hss_stor=np.array(e_hss_stor),
                   t_hss=np.array(t_hss), p_cl_grid=np.array(p_cl_grid), p_cl_rec=np.array(p_cl_rec),
                   p_cl_with=np.array(p_cl_with),
                   p_shared=np.array(p_shared), p_grid_out=np.array(p_grid_out),
                   p_grid_in=np.array(p_grid_in), d_cl=np.array(d_cl), p_ue=np.array(p_consumed),
                   p_hp_th=np.array(p_hp_th), p_hp_el=np.array(p_hp_el), d_hp=np.array(d_hp), T_zone=np.array(T_zone),
                   T_mass=np.array(T_mass))
    return results, status, pulp.value(prob.objective), prob.numVariables(), prob.numConstraints()


df = pd.read_csv('input.csv', sep=';', index_col=0, parse_dates=True)
# df_filtered = df[~df.index.astype(str).str.contains(":15|:30|:45", regex=True)]  # df.resample("1h").first()
df_filtered = df.resample("1h").sum()
na_values = df_filtered.values  # numpy array

np.random.seed(42)
hours = 8760
t = np.arange(hours)

T_env = (
        10 + 10 * np.sin(2 * np.pi * t / hours - np.pi / 2)
        + 3 * np.sin(2 * np.pi * t / 24)
        + np.random.normal(0, 1.0, hours)
)

COP = np.clip(6.0 - 0.25 * T_env, 1.5, 6.0)

df = pd.DataFrame({"T_env": T_env, "COP": COP})
df.to_csv("T_env_COP_8760.csv", index=False)

results, status, objective, num_vars, num_constraints = optimize(na_values[:, 1], na_values[:, 0], na_values[:, 2],
                                                                 size_elh=4, size_bess=20, size_hss=130, run_lp=True,
                                                                 objective="environmental", T_env_vector=T_env,
                                                                 cop_hp_vector=COP)

