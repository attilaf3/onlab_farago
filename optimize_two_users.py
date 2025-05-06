import time
import pandas as pd
import numpy as np
import pulp
from pulp import LpStatusOptimal, LpStatus


def optimize_two_users(p_pv, p_consumed, p_ut, dt=1, size_elh=None, size_bess=None, size_hss=None, vol_hss_water=None,
                       run_lp=False, **kwargs):

    # # Bemenetek ellenőrzése
    # print("Input shapes inside function:")
    # print("p_pv shape:", p_pv.shape)
    # print("p_consumed shape:", p_consumed.shape)
    # print("p_ut shape:", p_ut.shape)
    n_timestep = p_pv.shape[0]

    # n_timestep definiálása
    n_timestep = p_pv.shape[0]
    assert n_timestep == p_consumed.shape[0] == p_ut.shape[0], "Input arrays must have the same length."
    # print("n_timestep:", n_timestep)
    time_set = range(n_timestep)
    # print("time_set length:", len(time_set))

    # Parameters
    eta_bess_in = kwargs.get('eta_bess_in', 0.98)
    eta_bess_out = kwargs.get('eta_bess_out', 0.96)
    eta_bess_stor = kwargs.get('eta_bess_stor', 0.995)
    t_bess_min = kwargs.get('t_bess_min', 2)
    soc_bess_min = kwargs.get('soc_bess_min', 0.2)
    soc_bess_max = kwargs.get('soc_bess_max', 1)
    c_hss = kwargs.get('c_hss', 0.00116667)
    a_hss = kwargs.get('a_hss', 0.01275)
    T_env = kwargs.get('T_env', 20)
    T_max = kwargs.get('T_max', 55)
    T_in = kwargs.get('T_in', 12)
    T_min = T_in
    vol_hss_water = kwargs.get('vol_hss_water', [1000, 1200])
    eta_elh = kwargs.get('eta_elh', [1, 1])

    c_with = kwargs.get('c_with', 0.092)
    c_cl = kwargs.get('c_cl', 0.061)
    c_inj = kwargs.get('c_inj', 0.013)
    c_sh = kwargs.get('c_sh', 0.0024)

    msg = kwargs.get('msg', True)
    gapRel = kwargs.get('gapRel', None)
    timeLimit = kwargs.get('timeLimit', None)
    objective = kwargs.get('objective', "environmental")

    assert objective in ("economic", "environmental"), "Objective must be either economic or environmental"

    hss_flag = size_hss is not None
    user_ids = [0, 1]

    M_cl = 1.3 * (p_ut[:, 0] + p_ut[:, 1]).max()
    M_grid_in = 1.1 * (p_pv[:, 0] + p_pv[:, 1]).max() + size_bess / t_bess_min
    M_grid_out = 1.1 * (p_consumed[:, 0] + p_consumed[:, 1]).max() + size_bess / t_bess_min + (
                p_ut[:, 0] + p_ut[:, 1]).max() * 1.1
    M_cl_rec = (p_pv[:, 0] + p_pv[:, 1]).max() + size_bess / t_bess_min
    M_shared = 1.1 * max(M_grid_in, M_grid_out)

    prob = pulp.LpProblem("CSCopt", pulp.LpMinimize)

    # Define decision variables here
    p_cl_with = np.array([[pulp.LpVariable(f'Pcl_with_{t}_{u}', lowBound=0) for u in user_ids] for t in time_set])
    p_cl_grid = np.array([[pulp.LpVariable(f'Pcl_grid_{t}_{u}', lowBound=0) for u in user_ids] for t in time_set])
    p_cl_rec = np.array([[pulp.LpVariable(f'Pcl_rec_{t}_{u}', lowBound=0) for u in user_ids] for t in time_set])
    d_cl = [pulp.LpVariable(f'Dcl_{t}', cat=pulp.LpBinary) if not run_lp else 0 for t in time_set]

    p_elh_in = np.array([[pulp.LpVariable(f'Pelh_in_{t}_{u}', lowBound=0) for u in user_ids] for t in time_set])
    p_elh_out = np.array([[pulp.LpVariable(f'Pelh_out_{t}_{u}', lowBound=0) for u in user_ids] for t in time_set])

    p_hss_in = np.array([[pulp.LpVariable(f'Phss_in_{t}_{u}', lowBound=0) if hss_flag else 0 for u in user_ids] for t in time_set])
    p_hss_out = np.array([[pulp.LpVariable(f'Phss_out_{t}_{u}', lowBound=0) if hss_flag else 0 for u in user_ids] for t in time_set])
    # e_hss_stor = np.zeros(shape=(len(user_ids), len(time_set)), dtype=float)
    e_hss_stor = np.array(
        [[pulp.LpVariable(f'Ehss_stor_{t}_{u}', lowBound=0) if hss_flag else 0 for u in user_ids] for t in time_set])
    t_hss = np.array([[pulp.LpVariable(f'Thss_{t}_{u}', lowBound=T_in, upBound=T_max) if hss_flag else 0 for u in user_ids] for t in time_set])

    p_bess_in = [pulp.LpVariable(f'Pbess_in_{t}', lowBound=0) for t in time_set]
    p_bess_out = [pulp.LpVariable(f'Pbess_out_{t}', lowBound=0) for t in time_set]
    e_bess_stor = [pulp.LpVariable(f'Ebess_stor_{t}', lowBound=0) for t in time_set]
    d_bess = [pulp.LpVariable(f'Dbess_{t}', cat=pulp.LpBinary) if not run_lp else 0 for t in time_set]

    p_inj_user = np.array([[pulp.LpVariable(f'Pinj_{t}_{u}', lowBound=0) for u in user_ids] for t in time_set])
    p_with_user = np.array([[pulp.LpVariable(f'Pwith_{t}_{u}', lowBound=0) for u in user_ids] for t in time_set])
    p_selfcons = np.array([[pulp.LpVariable(f'Pselfcons_{t}_{u}') for u in user_ids] for t in time_set])
    d_user = np.array([[pulp.LpVariable(f'Dwith_{t}_{u}', cat=pulp.LpBinary) for u in user_ids] for t in time_set])
    d_grid = [pulp.LpVariable(f'Dgrid_{t}', cat=pulp.LpBinary) for t in time_set]

    p_grid_in = [pulp.LpVariable(f'Pgrid_in_{t}', lowBound=0) for t in time_set]
    p_grid_out = [pulp.LpVariable(f'Pgrid_out_{t}', lowBound=0) for t in time_set]
    p_inj = [pulp.LpVariable(f'Prec_inj_{t}', lowBound=0) for t in time_set]
    p_with = [pulp.LpVariable(f'Prec_with_{t}', lowBound=0) for t in time_set]
    p_shared = [pulp.LpVariable(f'Psh_{t}', lowBound=0) for t in time_set]
    y_shared = [pulp.LpVariable(f'Ysh_{t}', cat=pulp.LpBinary) for t in time_set]

    for t in time_set:
        k = (t + 1) % len(time_set)

        # BESS
        prob += e_bess_stor[k] - e_bess_stor[t] * eta_bess_stor == (
                p_bess_in[t] * eta_bess_in - p_bess_out[t] * (1 / eta_bess_out)) * dt
        prob += p_bess_in[t] <= d_bess[t] * (size_bess / t_bess_min)
        prob += p_bess_out[t] <= (1 - d_bess[t]) * (size_bess / t_bess_min)
        prob += e_bess_stor[t] <= size_bess * soc_bess_max
        prob += e_bess_stor[t] >= size_bess * soc_bess_min

        for u in user_ids:
            # Node balance
            prob += p_inj_user[t, u] + p_cl_with[t, u] + p_consumed[t, u] == p_pv[t, u] + p_with_user[t, u]
            prob += p_with_user[t, u] <= d_user[t, u] * (p_consumed[t, u] + M_cl)
            prob += p_inj_user[t, u] <= (1 - d_user[t, u]) * p_pv[t, u]

            # Controlled load
            prob += p_cl_with[t, u] == p_cl_grid[t, u] + p_cl_rec[t, u]

            if hss_flag:
                prob += vol_hss_water[u] * c_hss * (t_hss[k, u] - t_hss[t, u]) == (p_hss_in[t, u] -
                                                                                   p_hss_out[t, u] - a_hss * dt * (
                                                                                           t_hss[t, u] - T_env))
                prob += p_elh_out[t, u] == p_elh_in[t, u] * eta_elh[u]
                prob += p_elh_in[t, u] <= size_elh[u]
                prob += p_elh_in[t, u] == p_cl_with[t, u]
                prob += p_hss_in[t, u] == p_elh_out[t, u]
                prob += p_hss_out[t, u] == p_ut[t, u]
                prob += p_hss_out[k, u] <= vol_hss_water[u] * c_hss * (t_hss[t, u] - T_in)

        # # DEBUG: Kényszerített megosztott energia teszthez egy időpillanatban
        # t_test = 100  # tetszőleges időpillanat
        # prob += p_inj[t_test] >= 1
        # prob += p_with[t_test] >= 1

        # cl energy balance
        prob += pulp.lpSum(p_cl_rec[t, u] for u in user_ids) <= pulp.lpSum(p_pv[t, u] for u in user_ids) + p_bess_out[t]
        prob += pulp.lpSum(p_cl_grid[t, u] for u in user_ids) <= p_grid_out[t]

        # maximum power of controlled loads
        prob += pulp.lpSum(p_cl_rec[t, u] for u in user_ids) <= M_cl_rec
        prob += pulp.lpSum(p_cl_grid[t, u] for u in user_ids) <= M_grid_out
        prob += pulp.lpSum(p_cl_with[t, u] for u in user_ids) <= (M_cl_rec + M_grid_out)

        # Shared energy logic
        prob += p_inj[t] == pulp.lpSum(p_inj_user[t, u] for u in user_ids) + p_bess_out[t]
        prob += p_with[t] == pulp.lpSum(p_with_user[t, u] for u in user_ids) + p_bess_in[t]
        prob += p_grid_in[t] + p_with[t] == p_grid_out[t] + p_inj[t]
        prob += p_grid_in[t] <= d_grid[t] * M_grid_in
        prob += p_grid_out[t] <= (1 - d_grid[t]) * M_grid_out

        prob += p_shared[t] <= p_inj[t]
        prob += p_shared[t] <= p_with[t]
        prob += p_shared[t] >= p_inj[t] - M_shared * (1 - y_shared[t])
        prob += p_shared[t] >= p_with[t] - M_shared * y_shared[t]

    if t != time_set[0] and t != time_set[-1]:
        prob += d_cl[t + 1] >= d_cl[t] - d_cl[t - 1]
    n_timesteps_in_a_day = 24
    # d_cl időzítési megszorítás
    y_middle_day = [0] * 10 + [1] * 6 + [0] * 8
    for j in range(0, n_timestep - n_timesteps_in_a_day + 1, n_timesteps_in_a_day):
        prob += pulp.lpSum([d_cl[t] for t in range(j, j + n_timesteps_in_a_day)]) <= 12
        prob += pulp.lpSum([d_cl[t] * y_middle_day[t - j] for t in range(j, j + n_timesteps_in_a_day)]) >= 4

    # Objective
    if objective == "economic":
        prob += pulp.lpSum([c_with * p_grid_out[t] - c_inj * p_grid_in[t] - c_sh * p_shared[t] for t in time_set])
    else:
        prob += pulp.lpSum([p_grid_out[t] + p_grid_in[t] for t in time_set])

    # Write LP for debugging
    if run_lp:
        prob.writeLP("debug_two_user.lp")

    # Solve
    t = time.time()
    prob.solve(pulp.GUROBI_CMD(msg=True, gapRel=gapRel, timeLimit=timeLimit))
    print(f"Time to solve: {time.time() - t:.3}")
    status = pulp.LpStatus[prob.status]
    objective = pulp.value(prob.objective)

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

        for u in user_ids:
            p_inj_user[t, u] = pulp.value(p_inj_user[t, u])
            p_with_user[t, u] = pulp.value(p_with_user[t, u])
            if hss_flag:
                p_hss_in[t, u] = pulp.value(p_hss_in[t, u])
                p_hss_out[t, u] = pulp.value(p_hss_out[t, u])
                e_hss_stor[t, u] = vol_hss_water[u] * c_hss * (pulp.value(t_hss[t, u]) - T_in) / dt
                t_hss[t, u] = pulp.value(t_hss[t, u])
            p_elh_in[t, u] = pulp.value(p_elh_in[t, u])
            p_elh_out[t, u] = pulp.value(p_elh_out[t, u])
            p_cl_grid[t, u] = pulp.value(p_cl_grid[t, u])
            p_cl_rec[t, u] = pulp.value(p_cl_rec[t, u])
            p_cl_with[t, u] = pulp.value(p_cl_with[t, u])
            p_selfcons[t, u] = p_pv[t, u] - pulp.value(p_inj_user[t, u])

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

    return results, status, objective, user_ids, prob.numVariables(), prob.numConstraints()


# Load input
df = pd.read_csv('input_tobb_haztartas.csv', sep=';', index_col=0, parse_dates=True)
df_filtered = df.resample("1h").sum()
na_p_consumed = df_filtered[["consumer1", "consumer2"]].to_numpy()
na_p_pv = df_filtered[["pv1", "pv2"]].to_numpy()
na_p_ut = df_filtered[["thermal_user1", "thermal_user2"]].to_numpy()

# # Bemenetek ellenőrzése
# print("na_p_consumed shape:", na_p_consumed.shape)
# print("na_p_pv shape:", na_p_pv.shape)
# print("na_p_ut shape:", na_p_ut.shape)

# Run optimization
results, status, objective, user_ids, num_vars, num_constraints = optimize_two_users(p_pv=na_p_pv, p_ut=na_p_ut,
                                                                                     p_consumed=na_p_consumed,
                                                                                   size_elh=np.array([2, 2.5]),
                                                                                     size_bess=20)
# # Kimeneti méretek
# print("Final results shapes:")
# print("p_inj_user shape:", results['p_inj_user'].shape)
# print("p_with_user shape:", results['p_with_user'].shape)
# print("p_cl_with shape:", results['p_cl_with'].shape)
# print("p_grid_in shape:", results['p_grid_in'].shape)
# print("p_bess_in shape:", results['p_bess_in'].shape)
# print("Objective value:", objective)
# print("Number of variables:", num_vars)
# print("Number of constraints:", num_constraints)
is_shared_used = int(np.any(results["p_shared"] > 0))
print("Megosztott energia használt:", is_shared_used)

import matplotlib.pyplot as plt
import numpy as np

p_shared = results["p_shared"]

plt.figure(figsize=(10, 4))
plt.plot(p_shared, label="Megosztott energia (p_shared)", linewidth=1.5)
plt.title("Megosztott energia időbeli alakulása")
plt.xlabel("Időlépés")
plt.ylabel("Teljesítmény [kW]")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
