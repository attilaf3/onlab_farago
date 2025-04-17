import time

import numpy as np
import pulp
from pulp import LpStatusOptimal



def optimize(p_pv, p_consumed, p_ut, dt=1, size_elh=None, size_bess=None, size_hss=None, vol_hss_water=None, run_lp=False,
             **kwargs):
    """
    Function to optimization energy flows in the CSC setting
    """

    # Assert data validity
    assert (n_timestep := len(p_pv)) == len(p_consumed) == len(p_ut), \
        "Input arrays of environmental data must have the same length."

    # Get time array
    time_set = range(n_timestep)

    # Daily time steps
    n_timesteps_in_a_day = round(n_timestep / 365.)

    # Get keyword arguments

    # Technical parameters
    eta_bess_in = kwargs.get('eta_bess_in', 0.98)  # charge eff. of BESS
    eta_bess_out = kwargs.get('eta_bess_out', 0.96)  # disch. eff. of BESS
    eta_bess_stor = kwargs.get('eta_bess_stor', 0.995)  # storage eff.
    t_bess_min = kwargs.get('t_bess_min', 2)  # minimum time of discharge
    soc_bess_min = kwargs.get('soc_bess_min', 0.2)  # minimum SOC of BESS
    soc_bess_max = kwargs.get('soc_bess_max', 1)  # maximum SOC of BESS
    t_hss_min_in = kwargs.get('t_hss_min', 4)  # minimum time of charge
    t_hss_min_out = kwargs.get('t_hss_min', 1)  # minimum time of discharge
    c_hss = kwargs.get('c_hss', 0.00116667)  # kWh/kg/°C
    a_hss = kwargs.get('a_hss', 0.01275)  # kW/°C
    T_env = kwargs.get('T_env', 20)  # °C
    T_max = kwargs.get('T_max', 55)  # °C
    T_in = kwargs.get('T_in', 12)  # °C
    T_min = T_in
    vol_hss_water = kwargs.get('vol_hss_water', 3200)  # l/kg
    eta_elh = kwargs.get('eta_elh', 1)  # electric heater efficiency

    # Costs for the objective function
    c_with = kwargs.get('c_with', 0.092)  # price of electricity from the grid
    c_cl = kwargs.get('c_with', 0.061)  # price of controlled electricity
    c_inj = kwargs.get('c_inj', 0.013)  # price for electricity in the grid
    c_sh = kwargs.get('c_sh', 0.0024)  # network usage tariff for shared electricity

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
    # Initialize optimization problem
    prob = pulp.LpProblem("CSCopt", pulp.LpMinimize)

    # Parameters of the optimization

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

    # Initialize variables for the optimization problem
    # controlled appliances (CL)
    # controlled load input
    p_cl_with = [pulp.LpVariable(f'Pcl_with_{t}', lowBound=0) if cl_flag else 0 for t in time_set]
    # controlled load input from the grid
    p_cl_grid = [pulp.LpVariable(f'Pcl_grid_{t}', lowBound=0) if cl_flag else 0 for t in time_set]
    # controlled load input from the community
    p_cl_rec = [pulp.LpVariable(f'Pcl_rec_{t}', lowBound=0) if cl_flag else 0 for t in time_set]
    # Control signal of the controlled load
    d_cl = [pulp.LpVariable(f'Dcl_{t}', cat=pulp.LpBinary) if cl_flag and not run_lp else 0 for t in time_set]

    # Mark periods in a day: middle of the day period is from 10 am to 16 pm
    # Controlled load control signal allows controlled load to be switched on for 4 hours in the middle of the day
    y_middle_day = [0] * 10 + [1] * 6 + [0] * 8

    # Electric heater load (ELH)
    # electric power input
    p_elh_in = [pulp.LpVariable(f'Pelh_in_{t}', lowBound=0) if elh_flag else 0 for t in time_set]
    # thermal power output
    p_elh_out = [pulp.LpVariable(f'Pelh_out_{t}', lowBound=0) if elh_flag else 0 for t in time_set]

    # heat storage system (HSS)
    # input thermal power
    p_hss_in = [pulp.LpVariable(f'Phss_in_{t}', lowBound=0) if hss_flag else 0 for t in time_set]
    # output thermal power
    p_hss_out = [pulp.LpVariable(f'Phss_out_{t}', lowBound=0) if hss_flag else 0 for t in time_set]
    # stored thermal energy
    e_hss_stor = [pulp.LpVariable(f'Ehss_stor_{t}', lowBound=0) if hss_flag else 0 for t in time_set]
    # water temperature of the storage
    t_hss = [pulp.LpVariable(f'Thss_{t}', lowBound=T_min, upBound=T_max) if hss_flag else 0 for t in time_set]
    # state (charge/discharge)
    # d_hss = [pulp.LpVariable(f'Dhss_{t}', cat=pulp.LpBinary) if hss_flag and not run_lp else 0 for t in time_set]

    # battery energy storage system (BESS)
    # input electric power
    p_bess_in = [pulp.LpVariable(f'Pbess_in_{t}', lowBound=0) if bess_flag else 0 for t in time_set]
    # output electric power
    p_bess_out = [pulp.LpVariable(f'Pbess_out_{t}', lowBound=0) if bess_flag else 0 for t in time_set]
    # stored electric energy
    e_bess_stor = [pulp.LpVariable(f'Ebess_stor_{t}', lowBound=0) if bess_flag else 0 for t in time_set]
    # state (charge/discharge)
    d_bess = [pulp.LpVariable(f'Dbess_{t}', cat=pulp.LpBinary) if bess_flag and not run_lp else 0 for t in time_set]

    # electricity grid (only building, not end users inside: ?)
    # injected electricity
    p_inj = [pulp.LpVariable(f'Pinj_{t}', lowBound=0) for t in time_set]
    # withdrawn electricity
    p_with = [pulp.LpVariable(f'Pwith_{t}', lowBound=0) for t in time_set]
    # state (injection/withdrawal)
    d_grid = [pulp.LpVariable(f'Dgrid_{t}', cat=pulp.LpBinary) if not run_lp else 0 for t in time_set]
    # electricity grid (only building, not end users inside: ?)
    # injected electricity
    p_grid_in = [pulp.LpVariable(f'Pgrid_in_{t}', lowBound=0) for t in time_set]
    # withdrawn electricity
    p_grid_out = [pulp.LpVariable(f'Pgrid_out_{t}', lowBound=0) for t in time_set]

    # shared energy
    p_shared = [pulp.LpVariable(f'Psh_{t}', lowBound=0) for t in time_set]
    # auxiliary binary variables to evaluate shared energy
    y_shared = [pulp.LpVariable(f'Ysh_{t}', cat=pulp.LpBinary) if not run_lp else 0 for t in time_set]

    # Add the constraints to the problem
    # For each time step
    for t in time_set:

        # subsequent time-step (0 for the last one, i.e. cyclic)
        k = (t + 1) % len(time_set)

        # electric node, energy balance
        prob += p_pv[t] + p_grid_out[t] + p_bess_out[t] == p_grid_in[t] + p_bess_in[t] + p_consumed[t] + p_cl_with[t]
        # energy exchange between the grid and the hubs
        # prob += p_inj[t] - p_with[t] == p_grid_in[t] - p_grid_out[t]
        prob += p_inj[t] == p_pv[t] + p_bess_out[t]
        prob += p_with[t] == p_cl_with[t] + p_consumed[t] + p_bess_in[t]

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

    if cl_flag:
        for j in range(0, n_timestep, n_timesteps_in_a_day):
            if not run_lp:
                # At least (exactly) 8 hours of on-time for the switching
                prob += pulp.lpSum([d_cl[t] for t in range(j, j + n_timesteps_in_a_day)]) <= 12
                # At least 4 of it is in the middle of the day
                prob += pulp.lpSum([d_cl[t] * y_middle_day[t - j] for t in range(j, j + n_timesteps_in_a_day)]) >= 4

    # add the objective of the optimisation
    if objective == 'economic':
        if hss_flag:
            prob += pulp.lpSum([c_with * p_grid_out[t] - c_inj * p_grid_in[t] - c_sh * p_shared[t] for t in time_set])
        else:
            prob += pulp.lpSum([c_with * (p_grid_out[t] - p_cl_with[t]) + c_cl * p_cl_with[t] - c_inj * p_grid_in[
                t] - c_sh * p_shared[t] for t in time_set])
    else:
        prob += pulp.lpSum([p_grid_out[t] + p_grid_in[t] for t in time_set])

    # Solve the problem
    t = time.time()
    status = prob.solve(pulp.GUROBI_CMD(msg=True, gapRel=gapRel, timeLimit=timeLimit))
    if status != LpStatusOptimal:
        raise RuntimeError("Unable to solve the problem!")
    print(f"Time to solve: {time.time() - t:.3}")
    objective = pulp.value(prob.objective)

    # Process the results

    # Extract values from pulp variables
    for t in time_set:
        p_inj[t] = pulp.value(p_inj[t])
        p_with[t] = pulp.value(p_with[t])
        p_bess_in[t] = pulp.value(p_bess_in[t])
        p_bess_out[t] = pulp.value(p_bess_out[t])
        e_bess_stor[t] = pulp.value(e_bess_stor[t])
        p_hss_in[t] = pulp.value(p_hss_in[t])
        p_hss_out[t] = pulp.value(p_hss_out[t])
        # p_elh_in[t] = pulp.value(p_elh_in[t])
        # p_elh_out[t] = pulp.value(p_elh_out[t])
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

    p_elh_in = [0 if v is None else v for v in p_elh_in]
    p_elh_out = [0 if v is None else v for v in p_elh_out]
    e_hss_stor = vol_hss_water * c_hss * (np.array(t_hss) - T_env) / dt

    # Store in results
    results = dict(p_inj=np.array(p_inj), p_with=np.array(p_with),
                   p_bess_in=np.array(p_bess_in), p_bess_out=np.array(p_bess_out), e_bess_stor=np.array(e_bess_stor),
                   p_elh_in=np.array(p_elh_in), p_elh_out=np.array(p_elh_out),
                   p_hss_in=np.array(p_hss_in), p_hss_out=np.array(p_hss_out), e_hss_stor=np.array(e_hss_stor),
                   t_hss=np.array(t_hss), p_cl_grid=np.array(p_cl_grid), p_cl_rec=np.array(p_cl_rec),
                   p_cl_with=np.array(p_cl_with),
                   p_shared=np.array(p_shared), p_grid_out=np.array(p_grid_out),
                   p_grid_in=np.array(p_grid_in), d_cl=np.array(d_cl), p_ue=np.array(p_consumed))

    return results, status, objective, prob.numVariables(), prob.numConstraints()
