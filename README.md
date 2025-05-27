# ONLAB – Renewable Energy Communities Modeling and Optimization

This repository contains the code and documentation for a BSc thesis project focused on modeling and optimizing different types of renewable energy community (REC) systems. The models simulate and optimize energy flows for residential-scale systems involving photovoltaic (PV) generation, battery energy storage (BESS), thermal storage, and controllable thermal loads such as electric boilers and heat pumps.

The goal is to improve the environmental performance of local energy systems through increased self-consumption and self-sufficiency, while reducing dependency on the external grid.

## Structure

| File / Folder | Description |
|---------------|-------------|
| `optimize.py` | Optimization model for a single-user system (PV + ELH + BESS + HSS) |
| `optimize_two_users.py` | Optimization model for a two-user community with shared BESS and individual boilers |
| `optimize_with_hp.py` | OpOptimization model for a single-user system with heatpump (PV + ELH + BESS + HSS + HP) |
| `optimizecaller.py` | Visualization of results of the one-user system |
| `optimize_two_user_visualization.py` | Visualization of results of the two-user system |
| `optimize_with_hp_visualization.py` | Visualization of results of the one-user system with HP|
| `input.csv` | Hourly input time series data: PV generation, demand, DHW, weather |
| `input_tobb_haztartas.csv` | Hourly input time series data for the two-user system: PV generation, demand |

## System Configurations Modeled

- **1 household (PV + ELH)** – basic scenario with controllable electric boiler and thermal storage
- **2 households (PV + ELH)** – community model with shared battery and energy exchange
- **1 household (PV + ELH + HP)** – extended scenario with heat pump and building dynamics (5R2C model)

The optimization is solved using **Gurobi Optimizer** via mixed-integer linear programming (MILP).

## Notes

- All time series simulations use hourly resolution for the year 2023
- Thermal dynamics are modeled to support state-aware heating decisions
- The project supports environmental objective functions, focusing on maximizing local renewable utilization

