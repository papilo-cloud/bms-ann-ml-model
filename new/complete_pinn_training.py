# complete_pinn_training.py - Multi-State Battery Estimation
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Configuration
C_NOMINAL_AH = 1.86
LR = 5e-5
EPOCHS = 3000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}\n")

def load_battery_data(filename="B0005_discharge_only.csv"):
    """Load and prepare battery data"""
    try:
        df = pd.read_csv(filename)
        print(f"Loaded {len(df)} samples")
        
        # Data quality checks
        print(f"Data ranges:")
        print(f"  SoC: {df['SoC'].min():.3f} - {df['SoC'].max():.3f}")
        print(f"  SoH: {df['SoH'].min():.3f} - {df['SoH'].max():.3f}")
        print(f"  SoP: {df['SoP'].min():.3f} - {df['SoP'].max():.3f}")
        print(f"  Current: {df['current_A'].min():.3f} - {df['current_A'].max():.3f} A")
        print(f"  Voltage: {df['voltage_V'].min():.3f} - {df['voltage_V'].max():.3f} V\n")
        
        # Filter outliers
        df = df[
            (df['SoC'] >= 0.0) & (df['SoC'] <= 1.0) &
            (df['voltage_V'] > 2.5) & (df['voltage_V'] < 4.5) &
            (df['current_A'] < 0.5) &
            (df['SoH'] > 0.0) & (df['SoH'] <= 1.0) &
            (df['SoP'] >= 0.0) & (df['SoP'] <= 1.0)
        ].reset_index(drop=True)
        
        print(f"After filtering: {len(df)} samples\n")
        return df
        
    except FileNotFoundError:
        print(f"Error: {filename} not found")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def prepare_data_splits(df):
    """Prepare train/val/test splits with normalization"""
    df_sorted = df.sort_values(['cycle', 'time_s']).reset_index(drop=True)
    
    n = len(df_sorted)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    
    train_df = df_sorted.iloc[:n_train].copy()
    val_df = df_sorted.iloc[n_train:n_train+n_val].copy()
    test_df = df_sorted.iloc[n_train+n_val:].copy()
    
    # Compute normalization stats from training data
    norm_stats = {
        'time_min': train_df['time_s'].min(),
        'time_max': train_df['time_s'].max(),
        'current_mean': train_df['current_A'].mean(),
        'current_std': train_df['current_A'].std(),
        'voltage_mean': train_df['voltage_V'].mean(),
        'voltage_std': train_df['voltage_V'].std(),
        'temp_mean': train_df['temperature_C'].mean() if 'temperature_C' in train_df else 25.0,
        'temp_std': train_df['temperature_C'].std() if 'temperature_C' in train_df else 5.0
    }
    
    def tensorize(data_df):
        # Normalize inputs
        t_norm = (data_df['time_s'] - norm_stats['time_min']) / (norm_stats['time_max'] - norm_stats['time_min'] + 1e-8)
        i_norm = (data_df['current_A'] - norm_stats['current_mean']) / (norm_stats['current_std'] + 1e-8)
        v_norm = (data_df['voltage_V'] - norm_stats['voltage_mean']) / (norm_stats['voltage_std'] + 1e-8)
        
        if 'temperature_C' in data_df:
            temp_norm = (data_df['temperature_C'] - norm_stats['temp_mean']) / (norm_stats['temp_std'] + 1e-8)
            temp_values = temp_norm.values
        else:
            # temp_norm = np.zeros(len(data_df))
            temp_values = np.zeros(len(data_df))
        
        return {
            'time': torch.tensor(t_norm.values, dtype=torch.float32, device=DEVICE),
            'current': torch.tensor(i_norm.values, dtype=torch.float32, device=DEVICE),
            'voltage': torch.tensor(v_norm.values, dtype=torch.float32, device=DEVICE),
            'temperature': torch.tensor(temp_values, dtype=torch.float32, device=DEVICE),
            'soc': torch.tensor(data_df['SoC'].values, dtype=torch.float32, device=DEVICE),
            'soh': torch.tensor(data_df['SoH'].values, dtype=torch.float32, device=DEVICE),
            'sop': torch.tensor(data_df['SoP'].values, dtype=torch.float32, device=DEVICE),
            'time_raw': torch.tensor(data_df['time_s'].values, dtype=torch.float32, device=DEVICE),
            'current_raw': torch.tensor(data_df['current_A'].values, dtype=torch.float32, device=DEVICE)
        }
    
    return tensorize(train_df), tensorize(val_df), tensorize(test_df), norm_stats

class MultiStatePINN(nn.Module):
    """PINN for simultaneous SoC, SoH, and SoP estimation"""
    
    def __init__(self, hidden_dim=128):
        super().__init__()
        
        # Shared feature extractor
        self.shared_net = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Dropout(0.1)
        )
        
        # SoC prediction head
        self.soc_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        # SoH prediction head
        self.soh_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        # SoP prediction head
        self.sop_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)
    
    def forward(self, time, current, voltage, temperature):
        # Stack inputs
        x = torch.stack([time, current, voltage, temperature], dim=-1)
        
        # Shared features
        features = self.shared_net(x)
        
        # Individual predictions
        soc_logit = self.soc_head(features).squeeze(-1)
        soh_logit = self.soh_head(features).squeeze(-1)
        sop_logit = self.sop_head(features).squeeze(-1)
        
        # Apply sigmoid to keep in [0,1] range
        soc = torch.sigmoid(soc_logit)
        soh = torch.sigmoid(soh_logit)
        sop = torch.sigmoid(sop_logit)
        
        return soc, soh, sop

def compute_physics_loss(model, time, current, current_raw, norm_stats):
    """Compute physics loss: dSoC/dt = -|I|/(3600*C)"""
    time_grad = time.clone().detach().requires_grad_(True)
    
    try:
        # Dummy inputs for other variables
        current_dummy = current[:len(time_grad)]
        voltage_dummy = torch.zeros_like(time_grad)
        temp_dummy = torch.zeros_like(time_grad)
        
        soc, _, _ = model(time_grad, current_dummy, voltage_dummy, temp_dummy)
        
        dsoc_dt = torch.autograd.grad(
            outputs=soc.sum(),
            inputs=time_grad,
            create_graph=True,
            allow_unused=True
        )[0]
        
        if dsoc_dt is not None:
            time_scale = norm_stats['time_max'] - norm_stats['time_min']
            dsoc_dt_real = dsoc_dt / (time_scale + 1e-8)
            
            expected_dsoc_dt = -torch.abs(current_raw) / (3600.0 * C_NOMINAL_AH)
            
            physics_residual = dsoc_dt_real - expected_dsoc_dt
            return (physics_residual ** 2).mean()
        else:
            return torch.tensor(0.0, device=DEVICE)
    except:
        return torch.tensor(0.0, device=DEVICE)

def train_model(train_data, val_data, norm_stats):
    """Train the multi-state PINN"""
    model = MultiStatePINN(hidden_dim=128).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1500, factor=0.7)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("Starting training...\n")
    
    best_val_loss = float('inf')
    patience = 0
    history = {'train': [], 'val': [], 'physics': []}
    
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        soc_pred, soh_pred, sop_pred = model(
            train_data['time'], train_data['current'],
            train_data['voltage'], train_data['temperature']
        )
        
        # Data losses
        loss_soc = nn.functional.mse_loss(soc_pred, train_data['soc'])
        loss_soh = nn.functional.mse_loss(soh_pred, train_data['soh'])
        loss_sop = nn.functional.mse_loss(sop_pred, train_data['sop'])
        
        # Physics loss with curriculum learning
        physics_weight = min(0.05, (epoch / 5000.0) * 0.05)
        loss_physics = compute_physics_loss(
            model, train_data['time'], train_data['current'],
            train_data['current_raw'], norm_stats
        ) * physics_weight
        
        # Total loss
        total_loss = loss_soc + 0.5 * loss_soh + 0.5 * loss_sop + loss_physics
        
        # Backward
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Validation
        if epoch % 500 == 0:
            model.eval()
            with torch.no_grad():
                soc_val, soh_val, sop_val = model(
                    val_data['time'], val_data['current'],
                    val_data['voltage'], val_data['temperature']
                )
                
                val_loss = (nn.functional.mse_loss(soc_val, val_data['soc']) +
                           0.5 * nn.functional.mse_loss(soh_val, val_data['soh']) +
                           0.5 * nn.functional.mse_loss(sop_val, val_data['sop']))
                
                print(f"Epoch {epoch:5d} | Train: {total_loss.item():.6f} | "
                      f"Val: {val_loss.item():.6f} | "
                      f"SoC: {loss_soc.item():.6f} | Physics: {loss_physics.item():.6f}")
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience = 0
                    torch.save(model.state_dict(), 'best_multistate_pinn.pth')
                else:
                    patience += 500
                
                if patience >= 3000:
                    print(f"Early stopping at epoch {epoch}")
                    break
                
                history['train'].append(total_loss.item())
                history['val'].append(val_loss.item())
                history['physics'].append(loss_physics.item())
            
            scheduler.step(val_loss)
    
    model.load_state_dict(torch.load('best_multistate_pinn.pth'))
    print("\nTraining complete. Best model loaded.\n")
    return model, history

def calculate_metrics(y_true, y_pred, name=""):
    """Calculate performance metrics"""
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.cpu().numpy()
    
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-10))
    
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100
    max_error = np.max(np.abs(y_true - y_pred))
    
    return {'MAE': mae, 'RMSE': rmse, 'R²': r2, 'MAPE': mape, 'Max_Error': max_error}

def evaluate_and_visualize(model, train_data, val_data, test_data):
    """Evaluate model and create visualizations"""
    model.eval()
    
    with torch.no_grad():
        # Predictions
        soc_train, soh_train, sop_train = model(
            train_data['time'], train_data['current'],
            train_data['voltage'], train_data['temperature']
        )
        soc_val, soh_val, sop_val = model(
            val_data['time'], val_data['current'],
            val_data['voltage'], val_data['temperature']
        )
        soc_test, soh_test, sop_test = model(
            test_data['time'], test_data['current'],
            test_data['voltage'], test_data['temperature']
        )
    
    # Calculate metrics for all datasets and states
    print("="*70)
    print("PERFORMANCE METRICS")
    print("="*70)
    
    for dataset_name, true_data, pred_soc, pred_soh, pred_sop in [
        ("TRAIN", train_data, soc_train, soh_train, sop_train),
        ("VALIDATION", val_data, soc_val, soh_val, sop_val),
        ("TEST", test_data, soc_test, soh_test, sop_test)
    ]:
        print(f"\n{dataset_name} SET:")
        
        soc_metrics = calculate_metrics(true_data['soc'], pred_soc, "SoC")
        print(f"  SoC  - RMSE: {soc_metrics['RMSE']:.4f}, MAE: {soc_metrics['MAE']:.4f}, "
              f"R²: {soc_metrics['R²']:.4f}, MAPE: {soc_metrics['MAPE']:.2f}%")
        
        soh_metrics = calculate_metrics(true_data['soh'], pred_soh, "SoH")
        print(f"  SoH  - RMSE: {soh_metrics['RMSE']:.4f}, MAE: {soh_metrics['MAE']:.4f}, "
              f"R²: {soh_metrics['R²']:.4f}, MAPE: {soh_metrics['MAPE']:.2f}%")
        
        sop_metrics = calculate_metrics(true_data['sop'], pred_sop, "SoP")
        print(f"  SoP  - RMSE: {sop_metrics['RMSE']:.4f}, MAE: {sop_metrics['MAE']:.4f}, "
              f"R²: {sop_metrics['R²']:.4f}, MAPE: {sop_metrics['MAPE']:.2f}%")
    
    # Create visualization (only for test set)
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    time_test = test_data['time_raw'].cpu().numpy()
    
    # SoC plot
    axes[0].plot(time_test, test_data['soc'].cpu().numpy(), 'b-', 
                label='True SoC', linewidth=2.5, alpha=0.8)
    axes[0].plot(time_test, soc_test.cpu().numpy(), 'r--', 
                label='PINN SoC', linewidth=2.5)
    axes[0].set_title(f'State of Charge Estimation (R² = {soc_metrics["R²"]:.3f})', fontsize=14)
    axes[0].set_ylabel('SoC', fontsize=12)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # SoH plot
    axes[1].plot(time_test, test_data['soh'].cpu().numpy(), 'b-', 
                label='True SoH', linewidth=2.5, alpha=0.8)
    axes[1].plot(time_test, soh_test.cpu().numpy(), 'r--', 
                label='PINN SoH', linewidth=2.5)
    axes[1].set_title(f'State of Health Estimation (R² = {soh_metrics["R²"]:.3f})', fontsize=14)
    axes[1].set_ylabel('SoH', fontsize=12)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    # SoP plot
    axes[2].plot(time_test, test_data['sop'].cpu().numpy(), 'b-', 
                label='True SoP', linewidth=2.5, alpha=0.8)
    axes[2].plot(time_test, sop_test.cpu().numpy(), 'r--', 
                label='PINN SoP', linewidth=2.5)
    axes[2].set_title(f'State of Power Estimation (R² = {sop_metrics["R²"]:.3f})', fontsize=14)
    axes[2].set_xlabel('Time (s)', fontsize=12)
    axes[2].set_ylabel('SoP', fontsize=12)
    axes[2].legend(fontsize=11)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pinn_multistate_results.png', dpi=300, bbox_inches='tight')
    print("\nPlot saved as 'pinn_multistate_results.png'")
    plt.show()
    
    return soc_metrics, soh_metrics, sop_metrics

# Main execution
if __name__ == "__main__":
    print("Multi-State PINN for Battery Estimation")
    print("="*70)
    
    # Load data
    df = load_battery_data("B0005_discharge_only.csv")
    
    if df is not None:
        # Prepare data
        train_data, val_data, test_data, norm_stats = prepare_data_splits(df)
        print(f"Train: {len(train_data['time'])}, Val: {len(val_data['time'])}, Test: {len(test_data['time'])}\n")
        
        # Train model
        model, history = train_model(train_data, val_data, norm_stats)
        
        # Evaluate and visualize
        soc_metrics, soh_metrics, sop_metrics = evaluate_and_visualize(
            model, train_data, val_data, test_data
        )
        
        # Final summary
        print("\n" + "="*70)
        print("THESIS SUMMARY")
        print("="*70)
        print("Physics-Informed Neural Network for Multi-State Battery Estimation")
        print(f"  Dataset: NASA B0005 lithium-ion battery discharge data")
        print(f"  Physics: dSoC/dt = -|I|/(3600*C)")
        print(f"\nTest Set Performance:")
        print(f"  SoC: RMSE={soc_metrics['RMSE']:.4f}, R²={soc_metrics['R²']:.3f}")
        print(f"  SoH: RMSE={soh_metrics['RMSE']:.4f}, R²={soh_metrics['R²']:.3f}")
        print(f"  SoP: RMSE={sop_metrics['RMSE']:.4f}, R²={sop_metrics['R²']:.3f}")
        print("="*70)