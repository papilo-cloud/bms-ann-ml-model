# Extract and merge discharging cycles from B0005.mat
# Computes SoC, SoH, SoP and saves to CSV
import scipy.io
import numpy as np
import pandas as pd

# Load dataset
mat_data = scipy.io.loadmat("B0005.mat")
cycles = mat_data["B0005"][0, 0]["cycle"][0]

nominal_capacity = 2.0  # adjust to your dataset

rows = []

for cycle_idx, cycle in enumerate(cycles, start=1):
    cycle_type = cycle["type"][0]
    if cycle_type != "discharge": # only keep discharging data, skip others, e.g. 'charge'
        continue

    data = cycle["data"][0, 0]

    t = data["Time"].flatten()
    I = data["Current_measured"].flatten()
    V = data["Voltage_measured"].flatten()
    T = data["Temperature_measured"].flatten()

    # capacity series
    if "Capacity" in data.dtype.names:
        cap_series = data["Capacity"].flatten()
    else:
        cap_series = np.cumsum(I * np.gradient(t)) / 3600.0
    cap_series = np.maximum(cap_series, 0)

    # SoC per sample
    soc = cap_series / nominal_capacity

    # SoP (instantaneous power / nominal power)
    sop = (V * I) / (nominal_capacity * np.mean(V))

    # cycle-level stats
    cycle_capacity = np.max(cap_series)
    soh = cycle_capacity / nominal_capacity

    # merge per-sample data with per-cycle stats
    for ti, ci, vi, tiC, capi, soci, sopi in zip(t, I, V, T, cap_series, soc, sop):
        rows.append({
            "cycle": cycle_idx,
            "time_s": ti,
            "current_A": ci,
            "voltage_V": vi,
            "temperature_C": tiC,
            "capacity_Ah": capi,
            "SoC": soci,
            "SoH": soh,               # same across rows in the cycle
            "SoP": sopi,              # per-sample
            "cycle_capacity_Ah": cycle_capacity  # cycle summary merged in
        })

# Save merged CSV
df = pd.DataFrame(rows)
df.to_csv("B0005_discharging.csv", index=False)
print("✅ Merged file saved: B0005_discharging.csv")
# Note: Adjust nominal_capacity based on your dataset specifications
