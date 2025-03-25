import time

import numpy as np
import pulp

from utility.configuration import config


def optimize(p_pv, p_ue, p_ut, control_signal, size_bess, users, user_names, c_with, c_inj, c_sh, c_cl, msg, gapRel,
             timeLimit, objective, n_timestep, use_hss, dt, **kwargs):
    """
    Function to optimization energy flows in the CSC setting
    """

    # Assert data validity
    assert (n_timestep == p_pv.shape[0] == p_ue.shape[0] == p_ut.shape[0])
    "Input arrays of environmental data must have the same length."

    # Get time array
    time_set = range(n_timestep)

    # Get keyword arguments
    # Technical parameters
    eta_bess_in = kwargs.get('eta_bess_in', 0.98)  # charge eff. of BESS
    eta_bess_out = kwargs.get('eta_bess_out', 0.96)  # disch. eff. of BESS
    eta_bess_stor = kwargs.get('eta_bess_stor', 0.995)  # storage eff.
    t_bess_min = kwargs.get('t_bess_min', 2)  # minimum time of discharge
    soc_bess_min = kwargs.get('soc_bess_min', 0.2)  # minimum SOC of BESS
    soc_bess_max = kwargs.get('soc_bess_max', 1)  # maximum SOC of BESS

    T_env = kwargs.get('T_env', config.getint("physics", "room_temperature"))  # °C
    T_max = kwargs.get('T_max', config.getint("physics", "hot_water_set_temperature"))  # °C
    T_in = kwargs.get('T_in', kwargs.get('T_set', config.getint("physics", "cold_water_temperature")))  # °C
    T_min = T_in

    assert objective in ("economic", "environmental"), "Objective must be either economic or environmental"
    # Initialize optimization problem
    prob = pulp.LpProblem("CSCopt", pulp.LpMinimize)

    # Get some useful variables
    pv_flag = [user_name in p_pv.columns for u, user_name in enumerate(user_names)]
    cl_flag = [user_name in p_ut.columns for u, user_name in enumerate(user_names)]
    ue_flag = [user_name in p_ue.columns for u, user_name in enumerate(user_names)]
    bess_flag = size_bess is not None
    c = config.getfloat("physics", "energy_per_mass_ratio_water")

    # Parameters of the optimization
    # fake-big-M parameter for maximum grid injections
    M_cl = 1.3 * p_ut.total.max()
    M_grid_in = 1.1 * p_pv.total.max()
    if bess_flag:
        M_grid_in += size_bess / t_bess_min
    # fake-big-M parameter for maximum grid withdrawals
    M_grid_out = 1.1 * p_ue.total.max()
    M_cl_rec = p_pv.total.max()
    if bess_flag:
        M_grid_out += size_bess / t_bess_min
        M_cl_rec += size_bess / t_bess_min
    if cl_flag:
        # Good approximation of charge, discharge and storage efficiency of heater and heat storage
        M_grid_out += p_ut.total.max() * 1.1
    # big-M parameter needed to evaluate shared energy
    M_shared = 1.1 * max(M_grid_in, M_grid_out)

    p_pv = np.array([[p_pv.loc[t, user] if pv_flag[u] else 0 for t in time_set] for u, user in enumerate(user_names)])
    p_ue = np.array([[p_ue.loc[t, user] if ue_flag[u] else 0 for t in time_set] for u, user in enumerate(user_names)])
    p_ut = np.array([[p_ut.loc[t, user] if cl_flag[u] else 0 for t in time_set] for u, user in enumerate(user_names)])
    # d_cl = np.array([control_signal.loc[t] for t in time_set])
    vol_hss_water, a_hss, eta_elh, size_elh = None, None, None, None
    if use_hss:
        vol_hss_water = [users.loc[user, "vol_hss_water"] if cl_flag[u] else 0 for u, user in enumerate(user_names)]
        a_hss = [users.loc[user, "a_hss"] if cl_flag[u] else 0 for u, user in enumerate(user_names)]
        eta_elh = [users.loc[user, "eta_elh"] if cl_flag[u] else 0 for u, user in enumerate(user_names)]
        size_elh = [users.loc[user, "size_elh"] if cl_flag[u] else 0 for u, user in enumerate(user_names)]

    # Initialize variables for the optimization problem
    # controlled appliances (CL)
    # controlled load input
    # withdrawn electricity
    p_cl_with = np.array(
        [[pulp.LpVariable(f'Pcl_with_{u}_{t}', lowBound=0) if cl_flag[u] else 0 for t in time_set] for u, user in
         enumerate(user_names)])
    # controlled load input from the grid
    p_cl_grid = np.array(
        [[pulp.LpVariable(f'Pcl_grid_{u}_{t}', lowBound=0) if cl_flag[u] else 0 for t in time_set] for u, user in
         enumerate(user_names)])
    # controlled load input from the community
    p_cl_rec = np.array(
        [[pulp.LpVariable(f'Pcl_rec_{u}_{t}', lowBound=0) if cl_flag[u] else 0 for t in time_set] for u, user in
         enumerate(user_names)])
    # Control signal of the controlled load
    d_cl = [pulp.LpVariable(f'Dcl_{t}', cat=pulp.LpBinary) if cl_flag else 0 for t in time_set]
    # Mark periods in a day: middle of the day period is from 10 am to 16 pm
    # Controlled load control signal allows controlled load to be switched on for 4 hours in the middle of the day
    y_middle_day = [0] * 10 + [1] * 6 + [0] * 8

    # Electric heater load (ELH)
    # electric power input
    p_elh_in = np.array(
        [[pulp.LpVariable(f'Pelh_in_{u}_{t}', lowBound=0) if cl_flag[u] else 0 for t in time_set] for u, user in
         enumerate(user_names)])
    p_elh_out = np.array(
        [[pulp.LpVariable(f'Pelh_out_{u}_{t}', lowBound=0) if cl_flag[u] else 0 for t in time_set] for u, user in
         enumerate(user_names)])

    # heat storage system (HSS)
    # input thermal power
    p_hss_in = np.array(
        [[pulp.LpVariable(f'Phss_in_{u}_{t}', lowBound=0) if cl_flag[u] and use_hss else 0 for t in time_set] for
         u, user in enumerate(user_names)])
    # output thermal power
    p_hss_out = np.array(
        [[pulp.LpVariable(f'Phss_out_{u}_{t}', lowBound=0) if cl_flag[u] and use_hss else 0 for t in time_set] for
         u, user in enumerate(user_names)])
    # water temperature of the storage
    t_hss = np.array([[pulp.LpVariable(f'Thss_{u}_{t}', lowBound=T_in, upBound=T_max) if cl_flag[u] and use_hss else 0
                       for t in time_set] for u, user in enumerate(user_names)])

    # battery energy storage system (BESS)
    # input electric power
    p_bess_in = [pulp.LpVariable(f'Pbess_in_{t}', lowBound=0) if bess_flag else 0 for t in time_set]
    # output electric power
    p_bess_out = [pulp.LpVariable(f'Pbess_out_{t}', lowBound=0) if bess_flag else 0 for t in time_set]
    # stored electric energy
    e_bess_stor = [pulp.LpVariable(f'Ebess_stor_{t}', lowBound=0) if bess_flag else 0 for t in time_set]
    # state (charge/discharge)
    d_bess = [pulp.LpVariable(f'Dbess_{t}', cat=pulp.LpBinary) if bess_flag else 0 for t in time_set]

    # electricity grid (only building)
    # injected electricity
    p_inj_user = np.array(
        [[pulp.LpVariable(f'Pinj_{u}_{t}', lowBound=0) for t in time_set] for u, user in enumerate(user_names)])
    # withdrawn electricity
    p_with_user = np.array(
        [[pulp.LpVariable(f'Pwith_{u}_{t}', lowBound=0) for t in time_set] for u, user in enumerate(user_names)])
    # state (injection/withdrawal) of user
    d_user = np.array(
        [[pulp.LpVariable(f'Dwith_{u}_{t}', cat=pulp.LpBinary) for t in time_set] for u, user in enumerate(user_names)])

    # electricity grid
    # injected electricity
    p_grid_in = [pulp.LpVariable(f'Pgrid_in_{t}', lowBound=0) for t in time_set]
    # withdrawn electricity
    p_grid_out = [pulp.LpVariable(f'Pgrid_out_{t}', lowBound=0) for t in time_set]
    # state (injection/withdrawal)
    d_grid = [pulp.LpVariable(f'Dgrid_{t}', cat=pulp.LpBinary) for t in time_set]

    # community
    p_inj = [pulp.LpVariable(f'Prec_inj_{t}', lowBound=0) for t in time_set]
    # withdrawn electricity
    p_with = [pulp.LpVariable(f'Prec_with_{t}', lowBound=0) for t in time_set]

    # shared energy
    p_shared = [pulp.LpVariable(f'Psh_{t}', lowBound=0) for t in time_set]
    # auxiliary binary variables to evaluate shared energy
    y_shared = [pulp.LpVariable(f'Ysh_{t}', cat=pulp.LpBinary) for t in time_set]

    # Add the constraints to the problem
    # For each time step
    for t in time_set:
        k = (t + 1) % len(time_set)

        # battery energy storage system
        if bess_flag:
            # energy balance between time steps
            prob += e_bess_stor[k] - e_bess_stor[t] * eta_bess_stor == (
                    p_bess_in[t] * eta_bess_in - p_bess_out[t] * (1 / eta_bess_out)) * dt
            # maximum input and output power and mutual exclusivity
            prob += p_bess_in[t] <= d_bess[t] * (size_bess / t_bess_min)
            prob += p_bess_out[t] <= (1 - d_bess[t]) * (size_bess / t_bess_min)
            # maximum and minimum storable energy
            prob += e_bess_stor[t] <= (size_bess * soc_bess_max)
            prob += e_bess_stor[t] >= (size_bess * soc_bess_min)

        for u, _ in enumerate(user_names):
            # electric node, energy balance
            # energy exchange between the grid and the hub
            prob += p_inj_user[u, t] + p_cl_with[u, t] + p_ue[u, t] == p_pv[u, t] + p_with_user[u, t]
            prob += p_with_user[u, t] <= d_user[u, t] * (p_ue[u, t] + M_cl)
            prob += p_inj_user[u, t] <= (1 - d_user[u, t]) * p_pv[u, t]

            # controlled load
            if not cl_flag[u]:
                continue

            prob += p_cl_with[u, t] == p_cl_grid[u, t] + p_cl_rec[u, t]

            if use_hss:
                prob += vol_hss_water[u] * c * (t_hss[u, k] - t_hss[u, t]) == p_hss_in[u, t] - p_hss_out[u, t] \
                        - a_hss[u] * dt * (t_hss[u, t] - T_env)
#
                # # constitutive equation for electric heater
                prob += p_elh_out[u, t] == p_elh_in[u, t] * eta_elh[u]
#
                # # maximum power output
                prob += p_elh_in[u, t] <= size_elh[u]
#
                # # heat-energy conversion
                prob += p_elh_in[u, t] == p_cl_with[u, t]
                prob += p_hss_in[u, t] == p_elh_out[u, t]
                prob += p_hss_out[u, t] == p_ut[u, t]
                prob += p_hss_out[u, k] <= vol_hss_water[u] * c * (t_hss[u, t] - T_in)

            else:
                prob += p_cl_with[u, t] == p_ut[u, t]

        # cl energy balance
        prob += pulp.lpSum(p_cl_rec[u, t] for u, _ in enumerate(user_names) if cl_flag[u]) <= pulp.lpSum(p_pv[u, t] for u, _ in enumerate(user_names)) + p_bess_out[t]
        prob += pulp.lpSum(p_cl_grid[u, t] for u, _ in enumerate(user_names) if cl_flag[u]) <= p_grid_out[t]

        # maximum power of controlled loads
        prob += pulp.lpSum(p_cl_rec[u, t] for u, _ in enumerate(user_names) if cl_flag[u]) <= M_cl_rec
        prob += pulp.lpSum(p_cl_grid[u, t] for u, _ in enumerate(user_names) if cl_flag[u]) <= M_grid_out
        prob += pulp.lpSum(p_cl_with[u, t] for u, _ in enumerate(user_names) if cl_flag[u]) <= (M_cl_rec + M_grid_out)

        # grid
        # energy balance
        prob += p_inj[t] + p_grid_out[t] == p_grid_in[t] + p_with[t]
        prob += p_inj[t] == pulp.lpSum(p_inj_user[u, t] for u, _ in enumerate(user_names)) + p_bess_out[t]
        prob += p_with[t] == pulp.lpSum(p_with_user[u, t] for u, _ in enumerate(user_names)) + p_bess_in[t]

        # maximum injected and withdrawn power and mutual exclusivity
        prob += p_grid_in[t] <= d_grid[t] * M_grid_in
        prob += p_grid_out[t] <= (1 - d_grid[t]) * M_grid_out

        # linearization of shared energy definition
        # constraint on the shared energy, that must be smaller than both
        # the injections and the withdrawals of the virtual users
        prob += p_shared[t] <= p_inj[t]
        prob += p_shared[t] <= p_with[t]
        # constraint on the shared energy, that must also be equal to the
        # minimum between the two values.
        # when y_shared == 1: shared_power = p_inj_user, thanks to
        # this constraint and smaller-equal for the previous one.
        # when y_shared == 0, the other way around.
        prob += p_shared[t] >= p_inj[t] - M_shared * (1 - y_shared[t])
        prob += p_shared[t] >= p_with[t] - M_shared * y_shared[t]

        # Controlled load switching: must be on for at least two consecutive time steps
        if t != time_set[0] and t != time_set[-1]:
            # Switched on for at least two consecutive time steps
            prob += d_cl[t + 1] >= d_cl[t] - d_cl[t - 1]

    n_timesteps_in_a_day = 24
    if cl_flag:
        for j in range(0, n_timestep, n_timesteps_in_a_day):
            # At least (exactly) 8 hours of on-time for the switching
            prob += pulp.lpSum([d_cl[t] for t in range(j, j + n_timesteps_in_a_day)]) <= 12
            # At least 4 of it is in the middle of the day
            prob += pulp.lpSum([d_cl[t] * y_middle_day[t - j] for t in range(j, j + n_timesteps_in_a_day)]) >= 4

    # add the objective of the optimisation
    if objective == 'economic':
        if use_hss:
            prob += pulp.lpSum([c_with * p_grid_out[t] - c_inj * p_grid_in[t] - c_sh * p_shared[t] for t in time_set])
        else:
            prob += pulp.lpSum([c_with * (p_grid_out[t] - p_cl_with[t]) + c_cl * p_cl_with[t] - c_inj * p_grid_in[
                t] - c_sh * p_shared[t] for t in time_set])
    else:
        prob += pulp.lpSum([p_grid_out[t] + p_grid_in[t] for t in time_set])

    # Solve the problem
    t = time.time()
    prob.solve(pulp.GUROBI_CMD(msg=True, gapRel=gapRel, timeLimit=timeLimit))
    print(f"Time to solve: {time.time() - t:.3}")
    status = pulp.LpStatus[prob.status]
    objective = pulp.value(prob.objective)

    # Process the results
    # Extract values from pulp variables
    e_hss_stor = np.zeros(shape=(len(user_names), n_timestep), dtype=float)
    for t in time_set:
        p_bess_in[t] = pulp.value(p_bess_in[t])
        p_bess_out[t] = pulp.value(p_bess_out[t])
        e_bess_stor[t] = pulp.value(e_bess_stor[t])
        p_grid_in[t] = pulp.value(p_grid_in[t])
        p_grid_out[t] = pulp.value(p_grid_out[t])
        d_bess[t] = pulp.value(d_bess[t])
        d_grid[t] = pulp.value(d_grid[t])
        y_shared[t] = pulp.value(y_shared[t])
        p_shared[t] = pulp.value(p_shared[t])
        d_cl[t] = pulp.value(d_cl[t])
        p_inj[t] = pulp.value(p_inj[t])
        p_with[t] = pulp.value(p_with[t])

        for u, _ in enumerate(user_names):
            p_inj_user[u, t] = pulp.value(p_inj_user[u, t])
            p_with_user[u, t] = pulp.value(p_with_user[u, t])
            if use_hss:
                p_hss_in[u, t] = pulp.value(p_hss_in[u, t])
                p_hss_out[u, t] = pulp.value(p_hss_out[u, t])
                e_hss_stor[u, t] = vol_hss_water[u] * c * (pulp.value(t_hss[u, t]) - T_in) / dt
                t_hss[u, t] = pulp.value(t_hss[u, t])
            p_elh_in[u, t] = pulp.value(p_elh_in[u, t])
            p_elh_out[u, t] = pulp.value(p_elh_out[u, t])
            p_cl_grid[u, t] = pulp.value(p_cl_grid[u, t])
            p_cl_rec[u, t] = pulp.value(p_cl_rec[u, t])
            p_cl_with[u, t] = pulp.value(p_cl_with[u, t])
    p_selfcons = p_pv - p_inj_user

    # Store in results
    results = dict(p_inj_user=np.array(p_inj_user), p_with_user=np.array(p_with_user),
                   p_inj=np.array(p_inj), p_with=np.array(p_with),
                   p_bess_in=np.array(p_bess_in), p_bess_out=np.array(p_bess_out), e_bess_stor=np.array(e_bess_stor),
                   p_elh_in=np.array(p_elh_in), p_elh_out=np.array(p_elh_out),
                   p_hss_in=np.array(p_hss_in), p_hss_out=np.array(p_hss_out), e_hss_stor=np.array(e_hss_stor),
                   t_hss=np.array(t_hss), p_cl_grid=np.array(p_cl_grid), p_cl_rec=np.array(p_cl_rec),
                   p_cl_with=np.array(p_cl_with),
                   p_shared=np.array(p_shared), p_grid_out=np.array(p_grid_out),
                   p_grid_in=np.array(p_grid_in), d_cl=np.array(d_cl), p_selfcons=p_selfcons)

    return results, status, objective, user_names, prob.numVariables(), prob.numConstraints()
