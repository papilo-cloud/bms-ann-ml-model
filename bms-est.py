import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 1. Generate Synthetic Battery Data
np.random.seed(42)
n_samples = 500
voltage = np.random.uniform(3.0, 4.2, n_samples)
current = np.random.uniform(0.0, 2.0, n_samples)
temperature = np.random.uniform(20, 45, n_samples)
soc = np.random.uniform(0.1, 1.0, n_samples)
soh = np.random.uniform(0.8, 1.0, n_samples)
sop = np.random.uniform(0.5, 1.0, n_samples)
rul = np.random.uniform(50, 1000, n_samples)

df = pd.DataFrame({
    'Voltage': voltage,
    'Current': current,
    'Temperature': temperature,
    'SOC': soc,
    'SOH': soh,
    'SOP': sop,
    'RUL': rul
})

# 2. Features and Targets
X = df[['Voltage', 'Current', 'Temperature']]
y = df[['SOC', 'SOH', 'SOP', 'RUL']]

# 3. Normalize
scaler_X = MinMaxScaler()
scaler_Y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
Y_scaled = scaler_Y.fit_transform(y)

# 4. Train-Test Split
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y_scaled, test_size=0.2, random_state=42)

# 5. Convert to Tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
Y_train = torch.tensor(Y_train, dtype=torch.float32)
Y_test = torch.tensor(Y_test, dtype=torch.float32)

# 6. Define ANN
class BatteryANN(nn.Module):
    def __init__(self):
        super(BatteryANN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )

    def forward(self, x):
        return self.net(x)

model = BatteryANN()
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 7. Train Model
losses = []
epochs = 200
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = loss_fn(outputs, Y_train)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

# 8. Evaluate
model.eval()
with torch.no_grad():
    predictions = model(X_test).numpy()
    predictions_inv = scaler_Y.inverse_transform(predictions)
    Y_test_inv = scaler_Y.inverse_transform(Y_test.numpy())

    mae = mean_absolute_error(Y_test_inv, predictions_inv, multioutput='raw_values')
    rmse = np.sqrt(mean_squared_error(Y_test_inv, predictions_inv, multioutput='raw_values'))
    r2 = r2_score(Y_test_inv, predictions_inv, multioutput='raw_values')

    metrics = ['SOC', 'SOH', 'SOP', 'RUL']
    print("\nModel Evaluation:")
    for i, metric in enumerate(metrics):
        print(f"{metric} => MAE: {mae[i]:.4f}, RMSE: {rmse[i]:.4f}, R2: {r2[i]:.4f}")

# 9. Plot Loss Curve
plt.plot(losses)
plt.title("Training Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.grid(True)
plt.tight_layout()
plt.show()
