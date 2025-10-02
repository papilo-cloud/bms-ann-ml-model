# nasa_b0005_pinn.py - PINN model for NASA B0005 discharge data
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Configuration
C_NOMINAL_AH = 1.86  # NASA B0005 nominal capacity
LR = 1e-4
EPOCHS = 3000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

def load_and_prepare_nasa_data(filename="B0005_processed.csv"):
    """Load and prepare NASA B0005 discharge data for PINN training"""
    try:
        df = pd.read_csv(filename)
        print(f"Loaded {len(df)} samples from NASA B0005 discharge data")
        
        # Data quality checks
        print(f"SoC range: {df['SoC'].min():.3f} - {df['SoC'].max():.3f}")
        print(f"Voltage range: {df['voltage_V'].min():.3f} - {df['voltage_V'].max():.3f} V")
        print(f"Current range: {df['current_A'].min():.3f} - {df['current_A'].max():.3f} A")
        
        # Remove any remaining outliers
        df = df[
            (df['SoC'] >= 0.0) & (df['SoC'] <= 1.0) &
            (df['voltage_V'] > 2.5) & (df['voltage_V'] < 4.5) &
            (df['current_A'] > -5.0) & (df['current_A'] < 0.5) &
            (df['temperature_C'] > 15) & (df['temperature_C'] < 45)
        ].reset_index(drop=True)
        
        print(f"After filtering: {len(df)} samples")
        return df
        
    except FileNotFoundError:
        print(f"File {filename} not found. Please run the data extraction script first.")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def prepare_training_data(df):
    """Prepare data for PINN training with proper normalization"""
    
    # Sort by time to ensure proper temporal ordering
    df_sorted = df.sort_values(['cycle', 'time_s']).reset_index(drop=True)
    
    # Split data: 70% train, 15% val, 15% test
    n_total = len(df_sorted)
    n_train = int(0.7 * n_total)
    n_val = int(0.15 * n_total)
    
    train_df = df_sorted.iloc[:n_train].copy()
    val_df = df_sorted.iloc[n_train:n_train+n_val].copy()
    test_df = df_sorted.iloc[n_train+n_val:].copy()
    
    # Compute normalization parameters from training data only
    time_stats = {
        'min': train_df['time_s'].min(),
        'max': train_df['time_s'].max()
    }
    
    current_stats = {
        'min': train_df['current_A'].min(),
        'max': train_df['current_A'].max(),
        'mean': train_df['current_A'].mean(),
        'std': train_df['current_A'].std()
    }
    
    def normalize_and_tensorize(data_df, split_name):
        # Normalize time to [0, 1]
        time_norm = (data_df['time_s'] - time_stats['min']) / (time_stats['max'] - time_stats['min'] + 1e-8)
        
        # Standardize current (z-score normalization)
        current_norm = (data_df['current_A'] - current_stats['mean']) / (current_stats['std'] + 1e-8)
        
        # Add temperature as additional input (normalized)
        temp_mean = train_df['temperature_C'].mean()
        temp_std = train_df['temperature_C'].std()
        temp_norm = (data_df['temperature_C'] - temp_mean) / (temp_std + 1e-8)
        
        print(f"{split_name} - Time range: [{time_norm.min():.3f}, {time_norm.max():.3f}]")
        print(f"{split_name} - Current range: [{current_norm.min():.3f}, {current_norm.max():.3f}]")
        print(f"{split_name} - SoC range: [{data_df['SoC'].min():.3f}, {data_df['SoC'].max():.3f}]")
        
        return {
            'time': torch.tensor(time_norm.values, dtype=torch.float32, device=DEVICE),
            'current': torch.tensor(current_norm.values, dtype=torch.float32, device=DEVICE),
            'temperature': torch.tensor(temp_norm.values, dtype=torch.float32, device=DEVICE),
            'voltage': torch.tensor(data_df['voltage_V'].values, dtype=torch.float32, device=DEVICE),
            'soc': torch.tensor(data_df['SoC'].values, dtype=torch.float32, device=DEVICE),
            'time_raw': torch.tensor(data_df['time_s'].values, dtype=torch.float32, device=DEVICE),
            'current_raw': torch.tensor(data_df['current_A'].values, dtype=torch.float32, device=DEVICE),
            'cycle': torch.tensor(data_df['cycle'].values, dtype=torch.long, device=DEVICE)
        }
    
    train_data = normalize_and_tensorize(train_df, "Train")
    val_data = normalize_and_tensorize(val_df, "Validation") 
    test_data = normalize_and_tensorize(test_df, "Test")
    
    # Store normalization stats for physics calculations
    norm_stats = {
        'time': time_stats,
        'current': current_stats
    }
    
    return train_data, val_data, test_data, norm_stats

class NASA_PINN(nn.Module):
    """Physics-Informed Neural Network for NASA B0005 SoC estimation"""
    
    def __init__(self, hidden_dim=128):
        super().__init__()
        
        # Main SoC prediction network: time + current + temperature -> SoC
        self.feature_net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.Tanh(),
            nn.Linear(hidden_dim//2, 1)
        )
        
        # Physics-informed initialization
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for stable training"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)
    
    def forward(self, time, current, temperature):
        """Forward pass: predict SoC from inputs"""
        # Stack inputs
        x = torch.stack([time, current, temperature], dim=-1)
        
        # Get SoC logits
        soc_logit = self.feature_net(x).squeeze(-1)
        
        # Apply sigmoid to keep SoC in [0, 1] range
        soc = torch.sigmoid(soc_logit)
        
        return soc

def compute_physics_loss(model, time, current, current_raw, norm_stats):
    """Compute physics loss: dSoC/dt = -|I|/(3600*C)"""
    
    # Create time tensor that requires gradients
    time_grad = time.clone().detach().requires_grad_(True)
    
    try:
        # Forward pass with gradient tracking
        # Note: we use the mean temperature for physics loss calculation
        temp_mean = torch.zeros_like(time_grad)  # Simplified - use zero normalized temp
        soc_pred = model(time_grad, current, temp_mean)
        
        # Compute dSoC/dt
        dsoc_dt = torch.autograd.grad(
            outputs=soc_pred.sum(),
            inputs=time_grad,
            create_graph=True,
            allow_unused=True
        )[0]
        
        if dsoc_dt is not None:
            # Convert time derivative back to real units
            time_scale = norm_stats['time']['max'] - norm_stats['time']['min']
            dsoc_dt_real = dsoc_dt / time_scale  # dSoC per second
            
            # Physics equation: dSoC/dt = -|I|/(3600*C)
            # Note: current_raw is already in Amperes
            expected_dsoc_dt = -torch.abs(current_raw) / (3600.0 * C_NOMINAL_AH)
            
            # Compute physics residual
            physics_residual = dsoc_dt_real - expected_dsoc_dt
            return (physics_residual ** 2).mean()
        
        else:
            return torch.tensor(0.0, device=DEVICE)
            
    except Exception as e:
        print(f"Physics loss computation error: {e}")
        return torch.tensor(0.0, device=DEVICE)

def train_nasa_pinn(train_data, val_data, norm_stats):
    """Train the PINN model"""
    
    model = NASA_PINN(hidden_dim=128).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.8, patience=1000
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("Starting NASA B0005 PINN training...")
    
    best_val_loss = float('inf')
    patience_counter = 0
    train_history = {'loss': [], 'val_loss': [], 'physics_loss': []}
    
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        soc_pred = model(train_data['time'], train_data['current'], train_data['temperature'])
        
        # Data loss (MSE between predicted and true SoC)
        data_loss = nn.functional.mse_loss(soc_pred, train_data['soc'])
        
        # Physics loss with curriculum learning
        physics_weight = min(0.1, (epoch / 4000.0) * 0.1)  # Gradually increase
        physics_loss_val = compute_physics_loss(
            model, train_data['time'], train_data['current'], 
            train_data['current_raw'], norm_stats
        )
        physics_loss = physics_loss_val * physics_weight
        
        # Total loss
        total_loss = data_loss + physics_loss
        
        # Backward pass
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Validation every 500 epochs
        if epoch % 500 == 0:
            model.eval()
            with torch.no_grad():
                val_soc_pred = model(val_data['time'], val_data['current'], val_data['temperature'])
                val_loss = nn.functional.mse_loss(val_soc_pred, val_data['soc'])
                
                print(f"Epoch {epoch:5d} | Train: {total_loss.item():.6f} | "
                      f"Val: {val_loss.item():.6f} | Physics: {physics_loss.item():.6f} | "
                      f"LR: {optimizer.param_groups[0]['lr']:.2e}")
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    torch.save(model.state_dict(), 'nasa_b0005_pinn_best.pth')
                else:
                    patience_counter += 500
                
                # Early stopping
                if patience_counter >= 3000:
                    print(f"Early stopping at epoch {epoch}")
                    break
                
                # Store history
                train_history['loss'].append(total_loss.item())
                train_history['val_loss'].append(val_loss.item())
                train_history['physics_loss'].append(physics_loss.item())
            
            scheduler.step(val_loss)
    
    # Load best model
    model.load_state_dict(torch.load('nasa_b0005_pinn_best.pth'))
    print("Loaded best model weights")
    
    return model, train_history

def evaluate_model(model, train_data, val_data, test_data):
    """Comprehensive model evaluation"""
    
    def calculate_metrics(y_true, y_pred, dataset_name):
        if torch.is_tensor(y_true):
            y_true = y_true.cpu().numpy()
        if torch.is_tensor(y_pred):
            y_pred = y_pred.cpu().numpy()
        
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        
        # R² calculation
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-10))
        
        mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100
        
        print(f"{dataset_name} Metrics:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  R²:   {r2:.4f}")
        print(f"  MAPE: {mape:.2f}%")
        
        return {'RMSE': rmse, 'MAE': mae, 'R²': r2, 'MAPE': mape}
    
    model.eval()
    with torch.no_grad():
        # Predictions on all datasets
        train_pred = model(train_data['time'], train_data['current'], train_data['temperature'])
        val_pred = model(val_data['time'], val_data['current'], val_data['temperature'])
        test_pred = model(test_data['time'], test_data['current'], test_data['temperature'])
        
        # Calculate metrics
        train_metrics = calculate_metrics(train_data['soc'], train_pred, "TRAIN")
        val_metrics = calculate_metrics(val_data['soc'], val_pred, "VALIDATION")
        test_metrics = calculate_metrics(test_data['soc'], test_pred, "TEST")
        
        return {
            'train': {'pred': train_pred, 'metrics': train_metrics},
            'val': {'pred': val_pred, 'metrics': val_metrics},
            'test': {'pred': test_pred, 'metrics': test_metrics}
        }

def visualize_results(train_data, val_data, test_data, results, history):
    """Create comprehensive visualizations"""
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    
    # Plot 1: Test set SoC prediction
    time_test = test_data['time_raw'].cpu().numpy()
    soc_true_test = test_data['soc'].cpu().numpy()
    soc_pred_test = results['test']['pred'].cpu().numpy()
    
    axes[0,0].plot(time_test, soc_true_test, 'b-', label='True SoC', linewidth=2, alpha=0.8)
    axes[0,0].plot(time_test, soc_pred_test, 'r--', label='PINN SoC', linewidth=2)
    axes[0,0].set_title(f'SoC Prediction (Test Set)\nR² = {results["test"]["metrics"]["R²"]:.3f}')
    axes[0,0].set_xlabel('Time (s)')
    axes[0,0].set_ylabel('State of Charge')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Plot 2: Error analysis
    error_test = soc_pred_test - soc_true_test
    axes[0,1].plot(time_test, error_test, 'g-', linewidth=2)
    axes[0,1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[0,1].fill_between(time_test, error_test, 0, alpha=0.3, color='green')
    axes[0,1].set_title(f'Prediction Error\nMAE = {results["test"]["metrics"]["MAE"]:.4f}')
    axes[0,1].set_xlabel('Time (s)')
    axes[0,1].set_ylabel('Error (Predicted - True)')
    axes[0,1].grid(True, alpha=0.3)
    
    # Plot 3: Correlation plot
    axes[1,0].scatter(soc_true_test, soc_pred_test, alpha=0.6, s=20)
    axes[1,0].plot([0, 1], [0, 1], 'r--', linewidth=2)
    axes[1,0].set_xlabel('True SoC')
    axes[1,0].set_ylabel('Predicted SoC')
    axes[1,0].set_title(f'Correlation Analysis\nR² = {results["test"]["metrics"]["R²"]:.3f}')
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].set_xlim(0, 1)
    axes[1,0].set_ylim(0, 1)
    
    # Plot 4: Training curves
    epochs = np.arange(len(history['loss'])) * 500
    axes[1,1].semilogy(epochs, history['loss'], 'b-', label='Training Loss', linewidth=2)
    axes[1,1].semilogy(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[1,1].set_title('Training Convergence')
    axes[1,1].set_xlabel('Epoch')
    axes[1,1].set_ylabel('Loss (log scale)')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    # Plot 5: Physics loss evolution
    axes[2,0].plot(epochs, history['physics_loss'], 'g-', linewidth=2)
    axes[2,0].set_title('Physics Loss Evolution')
    axes[2,0].set_xlabel('Epoch')
    axes[2,0].set_ylabel('Physics Loss')
    axes[2,0].grid(True, alpha=0.3)
    
    # Plot 6: Cycle-wise performance
    cycles = test_data['cycle'].cpu().numpy()
    cycle_mae = []
    for cycle in np.unique(cycles):
        mask = cycles == cycle
        if np.sum(mask) > 0:
            cycle_error = np.abs(soc_pred_test[mask] - soc_true_test[mask])
            cycle_mae.append(np.mean(cycle_error))
        else:
            cycle_mae.append(0)
    
    axes[2,1].bar(np.unique(cycles), cycle_mae, alpha=0.7, color='purple')
    axes[2,1].set_title('Per-Cycle MAE Performance')
    axes[2,1].set_xlabel('Cycle Number')
    axes[2,1].set_ylabel('Mean Absolute Error')
    axes[2,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('nasa_b0005_pinn_results.png', dpi=300, bbox_inches='tight')
    plt.show()

# Main execution
if __name__ == "__main__":
    print("NASA B0005 PINN for SoC Estimation")
    print("="*50)
    
    # Load NASA B0005 discharge data
    df = pd.read_csv("B0005_discharge_only.csv") # load_and_prepare_nasa_data("B0005_processed.csv")
    
    if df is not None:
        # Prepare training data
        train_data, val_data, test_data, norm_stats = prepare_training_data(df)
        
        # Train PINN model
        model, history = train_nasa_pinn(train_data, val_data, norm_stats)
        
        # Evaluate model
        print("\n" + "="*50)
        print("MODEL EVALUATION RESULTS")
        print("="*50)
        
        results = evaluate_model(model, train_data, val_data, test_data)
        
        # Create visualizations
        visualize_results(train_data, val_data, test_data, results, history)
        
        # Summary for thesis
        test_metrics = results['test']['metrics']
        print("\n" + "="*60)
        print("THESIS SUMMARY - NASA B0005 PINN RESULTS")
        print("="*60)
        print(f"Physics-Informed Neural Network for NASA B0005 SoC Estimation:")
        print(f"• Dataset: Real NASA B0005 lithium-ion battery discharge cycles")
        print(f"• Physics constraint: dSoC/dt = -|I|/(3600*C)")
        print(f"• Test set performance:")
        print(f"  - RMSE: {test_metrics['RMSE']:.4f}")
        print(f"  - MAE:  {test_metrics['MAE']:.4f}")
        print(f"  - R²:   {test_metrics['R²']:.4f}")
        print(f"  - MAPE: {test_metrics['MAPE']:.2f}%")
        
        if test_metrics['R²'] > 0.85:
            print(f"\n✅ EXCELLENT: R² > 0.85 demonstrates superior performance")
        elif test_metrics['R²'] > 0.70:
            print(f"\n✅ GOOD: R² > 0.70 shows strong predictive capability")
        elif test_metrics['R²'] > 0.50:
            print(f"\n✅ ACCEPTABLE: R² > 0.50 indicates reasonable performance")
        else:
            print(f"\n⚠️ Results may need improvement for publication quality")
        
        print("="*60)
        
    else:
        print("Could not load NASA B0005 data. Please ensure:")
        print("1. Run the data extraction script first")
        print("2. B0005_discharge_corrected.csv file exists")
        print("3. Data contains valid discharge cycles")