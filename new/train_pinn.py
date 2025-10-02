# train_pinn_soc_soh_sop_fixed.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ----------------------------
# 0) Config / constants
# ----------------------------
C_NOMINAL_AH = 2.3           
P_REF = 100.0                
LR = 1e-3
EPOCHS = 4000
W_V = 1.0                    
W_SOH_SUP = 0.5              
W_SOP_SUP = 0.5              
W_SOH_RES = 1.0              
W_SOP_RES = 1.0              
DEVICE = "cpu"

# ----------------------------
# Dataset split
# ----------------------------
def split_dataset(data, train_ratio=0.7, val_ratio=0.15):
    df = data.copy()
    if df["SoC"].max() > 1.0: df["SoC"] /= df["SoC"].max()
    if "SoH" in df and df["SoH"].max() > 1.0: df["SoH"] /= df["SoH"].max()
    if "SoP" in df and df["SoP"].max() > 1.0: df["SoP"] /= df["SoP"].max()

    N = len(df)
    N_train = int(N * train_ratio)
    N_val   = int(N * val_ratio)

    train_df = df.iloc[:N_train]
    val_df   = df.iloc[N_train:N_train+N_val]
    test_df  = df.iloc[N_train+N_val:]

    def to_tensor(dfx):
        return {
            "time":    torch.tensor(dfx["time_s"].values, dtype=torch.float32, device=DEVICE),
            "current": torch.tensor(dfx["current_A"].values, dtype=torch.float32, device=DEVICE),
            "voltage": torch.tensor(dfx["voltage_V"].values, dtype=torch.float32, device=DEVICE),
            "soc":     torch.tensor(dfx["SoC"].values, dtype=torch.float32, device=DEVICE),
            "soh":     torch.tensor(dfx["SoH"].values, dtype=torch.float32, device=DEVICE) if "SoH" in dfx else None,
            "sop":     torch.tensor(dfx["SoP"].values, dtype=torch.float32, device=DEVICE) if "SoP" in dfx else None,
        }
    return to_tensor(train_df), to_tensor(val_df), to_tensor(test_df)

# ----------------------------
# OCV polynomial fitting
# ----------------------------
class OCVPoly(nn.Module):
    def __init__(self, coeffs):
        super().__init__()
        self.coeffs = nn.Parameter(torch.tensor(coeffs, dtype=torch.float32), requires_grad=False)

    def forward(self, soc):
        y = torch.zeros_like(soc)
        for c in self.coeffs:
            y = y * soc + c
        return y

def fit_ocv_coeffs_from_df(df, deg=3, low_I_threshold=0.05):
    mask = np.abs(df["current_A"].values) < low_I_threshold
    s = df["SoC"].values[mask]
    v = df["voltage_V"].values[mask]
    if len(s) < deg+1:
        s = df["SoC"].values
        v = df["voltage_V"].values
    coeffs = np.polyfit(s, v, deg=deg)
    return coeffs

# ----------------------------
# PINN model
# ----------------------------
class PINN(nn.Module):
    def __init__(self, ocv_fn, hidden=64):
        super().__init__()
        self.ocv = ocv_fn
        self._R = nn.Parameter(torch.tensor(0.01))
        self.body = nn.Sequential(
            nn.Linear(3, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 2)
        )
        self.a0 = nn.Parameter(torch.tensor(0.0))
        self.a1 = nn.Parameter(torch.tensor(0.0))
        self.a2 = nn.Parameter(torch.tensor(0.0))
        self.a3 = nn.Parameter(torch.tensor(0.0))
        self._tau = nn.Parameter(torch.tensor(1.0))

    @property
    def R(self): return torch.nn.functional.softplus(self._R) + 1e-6
    @property
    def tau(self): return torch.nn.functional.softplus(self._tau) + 1e-3

    def forward(self, t, I, soc):
        V_pred = self.ocv(soc) - I * self.R
        x = torch.stack([t, I, soc], dim=-1)
        out = self.body(x)
        soh_pred = torch.sigmoid(out[:,0])
        sop_pred = torch.sigmoid(out[:,1])
        return V_pred, soh_pred, sop_pred

    def soh_deg_rate(self, I, V):
        sp = torch.nn.functional.softplus
        return sp(self.a0 + self.a1*I.abs() + self.a2*(I.abs()*(V-3.7).abs()) + self.a3*(I.abs()**2))

# ----------------------------
# Load dataset
# ----------------------------
df = pd.read_csv("B0005_charging_with_SOH_SOP.csv")
if df["SoC"].max() > 1.0: df["SoC"] /= df["SoC"].max()
coeffs = fit_ocv_coeffs_from_df(df)
ocv_fn = OCVPoly(coeffs).to(DEVICE)

train, val, test = split_dataset(df)

# ----------------------------
# Training
# ----------------------------
model = PINN(ocv_fn).to(DEVICE)
opt = optim.Adam(model.parameters(), lr=LR)
mse = nn.MSELoss()

for epoch in range(EPOCHS):
    model.train()
    opt.zero_grad()

    time = train["time"].clone().detach().requires_grad_(True)
    I    = train["current"]
    V    = train["voltage"]
    soc  = train["soc"]

    V_pred, soh_pred, sop_pred = model(time, I, soc)

    # supervised losses
    loss_v = mse(V_pred, V) * W_V
    loss_soh_sup = mse(soh_pred, train["soh"]) * W_SOH_SUP if train["soh"] is not None else 0.0
    loss_sop_sup = mse(sop_pred, train["sop"]) * W_SOP_SUP if train["sop"] is not None else 0.0

    # SoH degradation residual
    dsoh_dt = torch.autograd.grad(soh_pred, time, torch.ones_like(soh_pred), create_graph=True)[0]
    rho = model.soh_deg_rate(I, V)
    loss_soh_res = ((dsoh_dt + rho)**2).mean() * W_SOH_RES

    # SoP first-order residual
    dsop_dt = torch.autograd.grad(sop_pred, time, torch.ones_like(sop_pred), create_graph=True)[0]
    y_target = (V**2 / (4.0 * model.R * P_REF)).clamp(0.0, 1.0)
    loss_sop_res = ((model.tau * dsop_dt + sop_pred - y_target)**2).mean() * W_SOP_RES

    loss = loss_v + loss_soh_sup + loss_sop_sup + loss_soh_res + loss_sop_res
    loss.backward()
    opt.step()

    if epoch % 500 == 0:
        with torch.no_grad():
            model.eval()
            tval = val["time"].clone().detach().requires_grad_(True)
            Vp, sohp, sopp = model(tval, val["current"], val["soc"])
            lv = mse(Vp, val["voltage"]).item()
            print(f"Epoch {epoch:4d} | Loss {loss.item():.5e} | Val V {lv:.5e} | R={model.R.item():.4f}Ω, tau={model.tau.item():.3f}s")

# ----------------------------
# Evaluation + metrics
# ----------------------------
model.eval()
with torch.no_grad():
    Vt, soh_t, sop_t = model(test["time"], test["current"], test["soc"])

def metrics(y_true, y_pred, name):
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    r2   = r2_score(y_true, y_pred)
    print(f"{name}: RMSE={rmse:.4f}, MAE={mae:.4f}, MAPE={mape:.2f}%, R²={r2:.4f}")

metrics(test["soc"], test["soc"], "SoC (baseline)")   # reference
if test["soh"] is not None: metrics(test["soh"], soh_t, "SoH")
if test["sop"] is not None: metrics(test["sop"], sop_t, "SoP")

# ----------------------------
# Plots (separate pages)
# ----------------------------
# SoC
plt.figure(figsize=(8,5))
plt.plot(test["time"].cpu(), test["soc"].cpu(), label="SoC (measured)")
plt.xlabel("Time (s)"); plt.ylabel("SoC")
plt.title("State of Charge (SoC)")
plt.legend(); plt.show()

# SoH
plt.figure(figsize=(8,5))
if test["soh"] is not None:
    plt.plot(test["time"].cpu(), test["soh"].cpu(), label="True SoH")
plt.plot(test["time"].cpu(), soh_t.cpu(), '--', label="PINN SoH")
plt.xlabel("Time (s)"); plt.ylabel("SoH")
plt.title("State of Health (SoH)")
plt.legend(); plt.show()

# SoP
plt.figure(figsize=(8,5))
if test["sop"] is not None:
    plt.plot(test["time"].cpu(), test["sop"].cpu(), label="True SoP")
plt.plot(test["time"].cpu(), sop_t.cpu(), '--', label="PINN SoP")
plt.xlabel("Time (s)"); plt.ylabel("SoP")
plt.title("State of Power (SoP)")
plt.legend(); plt.show()
