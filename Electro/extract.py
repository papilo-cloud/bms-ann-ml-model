import scipy.io
import pandas as pd
import numpy as np

# MAT_PATH = "B0005.mat"  # Put your downloaded file here
# CSV_OUT = "B0005_charging_PINN2.csv"

# mat = scipy.io.loadmat(MAT_PATH)
# battery = mat['B0005'][0, 0]
# cycles = battery['cycle'][0]

# rows = []
# for cycle in cycles:
#     typ = cycle['type'][0]
#     if typ != 'charge':
#         continue  # Only keep charging data

#     data = cycle['data'][0, 0]
#     t = data['Time'].flatten()
#     I = data['Current_measured'].flatten()
#     V = data['Voltage_measured'].flatten()
#     Temp = data['Temperature_measured'].flatten()
#     cap = data['Capacity'].flatten()[0] if 'Capacity' in data.dtype.names else np.nan
    
#     # If Capacity not per-time, estimate via coulomb count:
#     if np.isnan(cap):
#         cap_series = np.cumsum(I * np.gradient(t)) / 3600.0
#     else:
#         cap_series = cap

#     for ti, ii, vv, tm, ca in zip(t, I, V, Temp, cap_series):
#         rows.append({
#             'time_s': ti,
#             'current_A': ii,
#             'voltage_V': vv,
#             'temperature_C': tm,
#             'capacity_Ah': ca
#         })

# df = pd.DataFrame(rows)
# df.to_csv(CSV_OUT, index=False)
# print(f"Saved cleaned charging data to: {CSV_OUT}")


### Normalization of capacity_sh to SoC
# import scipy.io
# import pandas as pd
# import numpy as np

# MAT_PATH = "B0005.mat"  # Path to your downloaded file
# CSV_OUT = "B0005_charging_PINN.csv"

# # Load .mat file
# mat = scipy.io.loadmat(MAT_PATH)
# battery = mat['B0005'][0, 0]
# cycles = battery['cycle'][0]

# rows = []
# all_capacities = []

# # First pass: collect all capacities for normalization
# for cycle in cycles:
#     if cycle['type'][0] != 'charge':
#         continue
#     data = cycle['data'][0, 0]
#     if 'Capacity' in data.dtype.names:
#         all_capacities.append(data['Capacity'][0][0])

# nominal_capacity = max(all_capacities) if all_capacities else 2.0  # Fallback to 2Ah

# # Second pass: extract and normalize
# for cycle in cycles:
#     if cycle['type'][0] != 'charge':
#         continue

#     data = cycle['data'][0, 0]
#     t = data['Time'].flatten()
#     I = data['Current_measured'].flatten()
#     V = data['Voltage_measured'].flatten()
#     Temp = data['Temperature_measured'].flatten()

#     if 'Capacity' in data.dtype.names:
#         cap_series = data['Capacity'].flatten()
#     else:
#         cap_series = np.cumsum(I * np.gradient(t)) / 3600.0  # Coulomb counting

#     # Normalize to SoC (0 → empty, 1 → full)
#     soc_series = cap_series / nominal_capacity

#     for ti, ii, vv, tm, ca, soc in zip(t, I, V, Temp, cap_series, soc_series):
#         rows.append({
#             'time_s': ti,
#             'current_A': ii,
#             'voltage_V': vv,
#             'temperature_C': tm,
#             'capacity_Ah': ca,
#             'SoC': min(max(soc, 0), 1)  # Clamp between 0 and 1
#         })

# df = pd.DataFrame(rows)
# df.to_csv(CSV_OUT, index=False)
# print(f"✅ Saved cleaned charging data with SoC to: {CSV_OUT}")
# print(f"Nominal capacity used for normalization: {nominal_capacity:.3f} Ah")


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



# import scipy.io
# import pandas as pd
# import numpy as np

# MAT_PATH = "B0005.mat"  # Path to your downloaded file
# CSV_OUT = "B0005_charging_PINN1.csv"

# # Load .mat file
# mat = scipy.io.loadmat(MAT_PATH)
# battery = mat['B0005'][0, 0]
# cycles = battery['cycle'][0]

# rows = []
# all_capacities = []

# # First pass: collect all capacities for normalization (Ah)
# for cycle in cycles:
#     if cycle['type'][0] != 'charge':
#         continue
#     data = cycle['data'][0, 0]
#     if 'Capacity' in data.dtype.names:
#         all_capacities.append(data['Capacity'][0][0])

# nominal_capacity = max(all_capacities) if all_capacities else 2.0  # fallback to 2Ah

# # Function to estimate internal resistance from current/voltage
# def estimate_r_internal(V, I):
#     dV = np.diff(V)
#     dI = np.diff(I)
#     mask = np.abs(dI) > 0.01  # avoid division by near-zero
#     if not np.any(mask):
#         return np.nan
#     r_estimates = -dV[mask] / dI[mask]  # negative sign because V drops when I rises
#     return np.nanmedian(r_estimates)

# # Second pass: extract and compute SoH & SoP
# for cycle_idx, cycle in enumerate(cycles, start=1):
#     if cycle['type'][0] != 'charge':
#         continue

#     data = cycle['data'][0, 0]
#     t = data['Time'].flatten()
#     I = data['Current_measured'].flatten()
#     V = data['Voltage_measured'].flatten()
#     Temp = data['Temperature_measured'].flatten()

#     if 'Capacity' in data.dtype.names:
#         cap_series = data['Capacity'].flatten()
#         cycle_capacity = cap_series[-1]  # last value = full cycle capacity
#     else:
#         cap_series = np.cumsum(I * np.gradient(t)) / 3600.0  # coulomb counting
#         cycle_capacity = cap_series[-1]

#     # State of Health (SoH) for this cycle
#     SoH = cycle_capacity / nominal_capacity

#     # Estimate internal resistance (R_int)
#     R_int = estimate_r_internal(V, I)

#     # Estimate SoP: P = V^2 / (4R_int), normalized by first cycle
#     P_available = (np.mean(V) ** 2) / (4 * R_int) if R_int and R_int > 0 else np.nan
#     SoP = P_available / ((np.mean(V) ** 2) / (4 * R_int) if cycle_idx == 1 else P_available)

#     # Normalize SoC
#     soc_series = cap_series / nominal_capacity

#     for ti, ii, vv, tm, ca, soc in zip(t, I, V, Temp, cap_series, soc_series):
#         rows.append({
#             'cycle': cycle_idx,
#             'time_s': ti,
#             'current_A': ii,
#             'voltage_V': vv,
#             'temperature_C': tm,
#             'capacity_Ah': ca,
#             'SoC': min(max(soc, 0), 1),
#             'SoH': SoH,
#             'SoP': SoP
#         })

# df = pd.DataFrame(rows)
# df.to_csv(CSV_OUT, index=False)
# print(f"✅ Saved cleaned charging data with SoC, SoH, SoP to: {CSV_OUT}")
# print(f"Nominal capacity used for normalization: {nominal_capacity:.3f} Ah")


# import scipy.io
# import pandas as pd
# import numpy as np

# MAT_PATH = "B0005.mat"  # Path to your downloaded file
# CSV_OUT = "B0005_charging_PINN_extended.csv"

# # Load .mat file
# mat = scipy.io.loadmat(MAT_PATH)
# battery = mat['B0005'][0, 0]
# cycles = battery['cycle'][0]

# rows = []
# all_capacities = []
# all_powers = []

# # First pass: collect all capacities and powers for normalization
# for cycle in cycles:
#     if cycle['type'][0] != 'charge':
#         continue
#     data = cycle['data'][0, 0]
    
#     if 'Capacity' in data.dtype.names:
#         all_capacities.append(data['Capacity'][0][0])
    
#     # Calculate instantaneous power for SoP reference
#     I = data['Current_measured'].flatten()
#     V = data['Voltage_measured'].flatten()
#     all_powers.extend(np.abs(I * V))  # Absolute power values

# nominal_capacity = max(all_capacities) if all_capacities else 2.0  # Fallback to 2Ah
# max_power = max(all_powers) if all_powers else 1.0  # Fallback to 1W

# # Second pass: extract and compute all metrics
# for cycle in cycles:
#     if cycle['type'][0] != 'charge':
#         continue

#     data = cycle['data'][0, 0]
#     cycle_number = cycle['type'][0] + str(cycle['cycle_index'][0][0])
    
#     t = data['Time'].flatten()
#     I = data['Current_measured'].flatten()
#     V = data['Voltage_measured'].flatten()
#     Temp = data['Temperature_measured'].flatten()

#     # Calculate instantaneous power
#     P = I * V
    
#     if 'Capacity' in data.dtype.names:
#         cap_series = data['Capacity'].flatten()
#     else:
#         cap_series = np.cumsum(I * np.gradient(t)) / 3600.0  # Coulomb counting

#     # Calculate metrics
#     soc_series = cap_series / nominal_capacity
#     sop_series = np.abs(P) / max_power  # State of Power (normalized 0-1)
#     soh_series = cap_series / nominal_capacity  # State of Health (capacity ratio)

#     for ti, ii, vv, tm, ca, soc, sop, soh in zip(t, I, V, Temp, cap_series, soc_series, sop_series, soh_series):
#         rows.append({
#             'time_s': ti,
#             'current_A': ii,
#             'voltage_V': vv,
#             'temperature_C': tm,
#             'capacity_Ah': ca,
#             'SoC': min(max(soc, 0), 1),  # Clamp between 0 and 1
#             'SoP': min(max(sop, 0), 1),   # Clamp between 0 and 1
#             'SoH': min(max(soh, 0), 1),   # Clamp between 0 and 1
#             'cycle': cycle_number
#         })

# df = pd.DataFrame(rows)
# df.to_csv(CSV_OUT, index=False)
# print(f"✅ Saved extended data with SoC/SoP/SoH to: {CSV_OUT}")
# print(f"Reference values - Nominal capacity: {nominal_capacity:.3f} Ah, Max power: {max_power:.3f} W")