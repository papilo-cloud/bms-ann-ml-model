import pandas as pd
import numpy as np

# After loading the CSV with computed states
df = pd.read_csv("B0005_with_corrected_states.csv")

# Filter for discharge segments only
# Keep only rows where current is negative (discharge) with some margin
discharge_mask = df['current_A'] < -0.1  # At least 100mA discharge

df_discharge = df[discharge_mask].copy()

# Reset time to start from 0
df_discharge['time_s'] = df_discharge['time_s'] - df_discharge['time_s'].min()

# Recalculate SoC_CC after filtering
df_discharge = df_discharge.reset_index(drop=True)

print(f"Filtered: {len(df_discharge)} discharge samples from {len(df)} total")
print(f"SoC range: {df_discharge['SoC_CC'].min():.3f} - {df_discharge['SoC_CC'].max():.3f}")

# Save filtered data
df_discharge.to_csv("B0005_discharge_only.csv", index=False)