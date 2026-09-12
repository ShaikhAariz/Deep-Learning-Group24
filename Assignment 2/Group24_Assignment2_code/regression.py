import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

np.random.seed(0)

PLOT_DIR = "plots/regression"
os.makedirs(PLOT_DIR, exist_ok=True)

class FCNNReg:
    def __init__(self, layer_sizes, hidden_act='tanh', lr=0.01, epochs=1200, tol=1e-5):
        self.layer_sizes = layer_sizes
        self.hidden_act = hidden_act
        self.lr = lr
        self.epochs = epochs
        self.tol = tol
        self.loss_history = []
        
        self.W = [np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(1.0 / layer_sizes[i])
                  for i in range(len(layer_sizes) - 1)]
        self.b = [np.zeros(layer_sizes[i + 1]) for i in range(len(layer_sizes) - 1)]

    def _act(self, z, kind):
        if kind == 'tanh':
            return np.tanh(z)
        elif kind == 'logistic':
            z_clipped = np.clip(z, -250, 250)
            return 1.0 / (1.0 + np.exp(-z_clipped))
        return z

    def _dact(self, a, kind):
        if kind == 'tanh':
            return 1.0 - a**2
        elif kind == 'logistic':
            return a * (1.0 - a)
        return np.ones_like(a)

    def _forward(self, X):
        As = [X]
        A = X
        for i in range(len(self.W)):
            Z = A @ self.W[i] + self.b[i]
            kind = 'linear' if i == len(self.W) - 1 else self.hidden_act
            A = self._act(Z, kind)
            As.append(A)
        return As

    def compute_loss(self, X, y):
        y = y.reshape(-1, 1) if y.ndim == 1 else y
        As = self._forward(X)
        err = y - As[-1]
        return np.mean(0.5 * err**2)

    def train_sgd(self, X, y):
        y = y.reshape(-1, 1) if y.ndim == 1 else y
        n = len(X)
        self.loss_history = []
        prev_loss = float('inf')
        
        for epoch in range(self.epochs):
            idx = np.random.permutation(n)
            for j in idx:
                x = X[j:j+1]
                t = y[j:j+1]
                As = self._forward(x)
                dA = As[-1] - t
                
                for i in range(len(self.W) - 1, -1, -1):
                    kind = 'linear' if i == len(self.W) - 1 else self.hidden_act
                    dZ = dA * self._dact(As[i + 1], kind)
                    dW = As[i].T @ dZ
                    db = dZ[0]
                    if i > 0:
                        dA = dZ @ self.W[i].T
                    self.W[i] -= self.lr * dW
                    self.b[i] -= self.lr * db
                    
            epoch_loss = self.compute_loss(X, y)
            self.loss_history.append(epoch_loss)
            
            if abs(prev_loss - epoch_loss) < self.tol:
                break
            prev_loss = epoch_loss

    def predict(self, X):
        return self._forward(X)[-1].ravel()

    def count_parameters(self):
        return sum(w.size for w in self.W) + sum(b.size for b in self.b)

def split_60_20_20(X, y, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n = len(X)
    n_tr, n_val = int(n * 0.6), int(n * 0.2)
    return X[idx[:n_tr]], y[idx[:n_tr]], X[idx[n_tr:n_tr+n_val]], y[idx[n_tr:n_tr+n_val]], X[idx[n_tr+n_val:]], y[idx[n_tr+n_val:]]

def standardize_train_val_test(Xtr, Xval, Xte):
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0) + 1e-8
    return (Xtr - mu) / sd, (Xval - mu) / sd, (Xte - mu) / sd

def rmse_metrics(y_true, y_pred):
    err = y_true - y_pred
    rmse = np.sqrt(np.mean(err**2))
    pct_rmse = (rmse / (np.abs(np.mean(y_true)) + 1e-8)) * 100.0
    return rmse, pct_rmse

def resolve_path(candidates):
    for p in candidates:
        if os.path.exists(p): return p
    raise FileNotFoundError(f"Checked: {candidates}")

p_uni = resolve_path(['Regression/UnivariateData/24.csv', 'data/Regression/UnivariateData/24.csv'])
p_bi = resolve_path(['Regression/BivariateData/24.csv', 'data/Regression/BivariateData/24.csv'])

data_uni = pd.read_csv(p_uni, header=None).values
X_uni, y_uni = data_uni[:, :-1], data_uni[:, -1]

data_bi = pd.read_csv(p_bi, header=None).values
X_bi, y_bi = data_bi[:, :-1], data_bi[:, -1]

Xtr_u, ytr_u, Xval_u, yval_u, Xte_u, yte_u = split_60_20_20(X_uni, y_uni)
Xtr_b, ytr_b, Xval_b, yval_b, Xte_b, yte_b = split_60_20_20(X_bi, y_bi)

Xtr_u_s, Xval_u_s, Xte_u_s = standardize_train_val_test(Xtr_u, Xval_u, Xte_u)
Xtr_b_s, Xval_b_s, Xte_b_s = standardize_train_val_test(Xtr_b, Xval_b, Xte_b)

print("=" * 70)
print("CS601T PROGRAMMING ASSIGNMENT II: REGRESSION EXPERIMENTS")
print("=" * 70)
print(f"Univariate Dataset -> Train: {len(Xtr_u)}, Val: {len(Xval_u)}, Test: {len(Xte_u)}")
print(f"Bivariate Dataset  -> Train: {len(Xtr_b)}, Val: {len(Xval_b)}, Test: {len(Xte_b)}\n")

# ----------------- DATASET 1: UNIVARIATE REGRESSION -----------------
print("----------------------------------------------------------------------")
print("1. EVALUATING ALL ARCHITECTURES ON VALIDATION SET (DATASET 1: UNIVARIATE)")
print("----------------------------------------------------------------------")
uni_configs_1h = [[1, 2, 1], [1, 4, 1], [1, 8, 1], [1, 16, 1]]
uni_results = []

for cfg in uni_configs_1h:
    np.random.seed(0)
    m = FCNNReg(cfg, hidden_act='tanh', lr=0.01, epochs=1200, tol=1e-5)
    m.train_sgd(Xtr_u_s, ytr_u)
    r_tr, p_tr = rmse_metrics(ytr_u, m.predict(Xtr_u_s))
    r_val, p_val = rmse_metrics(yval_u, m.predict(Xval_u_s))
    uni_results.append((cfg, m, r_tr, p_tr, r_val, p_val))
    print(f"Architecture {cfg} -> Train RMSE: {r_tr:.4f} ({p_tr:.2f}%) | Val RMSE: {r_val:.4f} ({p_val:.2f}%)")

best_uni_item = min(uni_results, key=lambda x: (x[4], x[1].count_parameters()))
best_uni_cfg, best_uni_model = best_uni_item[0], best_uni_item[1]

print("=" * 70)
print(f"SELECTED BEST UNIVARIATE ARCHITECTURE: {best_uni_cfg} (Val RMSE: {best_uni_item[4]:.4f})")
print("=" * 70)

summary_uni = []
for item in uni_results:
    cfg, m, r_tr, p_tr, r_val, p_val = item
    summary_uni.append({
        "Architecture": str(cfg),
        "Parameters": m.count_parameters(),
        "Train RMSE": f"{r_tr:.4f}",
        "Train %RMSE": f"{p_tr:.2f}%",
        "Val RMSE": f"{r_val:.4f}",
        "Val %RMSE": f"{p_val:.2f}%"
    })
print("\nValidation Summary Table (Univariate Regression):")
print(pd.DataFrame(summary_uni).to_string(index=False))

# Learning Rate Comparison
plt.figure(figsize=(7, 4))
for lr in [0.01, 0.05, 0.08]:
    np.random.seed(0)
    m = FCNNReg(best_uni_cfg, hidden_act='tanh', lr=lr, epochs=1200, tol=1e-5)
    m.train_sgd(Xtr_u_s, ytr_u)
    plt.plot(m.loss_history, label=f'lr = {lr} (Stopped at epoch {len(m.loss_history)})')
plt.title(f'Univariate Best {best_uni_cfg}: Learning Rate Comparison')
plt.xlabel('Epochs')
plt.ylabel('Average Squared Error')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "Uni_lr_comparison.png"), dpi=300)
plt.close()

# Average Error vs Epochs for Best Model
plt.figure(figsize=(6, 4))
plt.plot(best_uni_model.loss_history, color='purple', lw=2)
plt.title(f'Univariate Best {best_uni_cfg}: Average Error vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Average Squared Error')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "Uni_best_error_vs_epochs.png"), dpi=300)
plt.close()

# Superimposed Plots (Train, Val, Test)
for X, y, split in [(Xtr_u_s, ytr_u, 'Train'), (Xval_u_s, yval_u, 'Val'), (Xte_u_s, yte_u, 'Test')]:
    plt.figure(figsize=(6, 4))
    plt.scatter(X[:, 0], y, c='r', s=15, label='Target Output', alpha=0.6)
    xs = np.linspace(X[:, 0].min() - 0.2, X[:, 0].max() + 0.2, 300)[:, None]
    plt.plot(xs[:, 0], best_uni_model.predict(xs), 'b-', lw=2, label='Model Output')
    plt.xlabel('x (Standardized)')
    plt.ylabel('y')
    plt.title(f'Univariate Best {best_uni_cfg} - Superimposed ({split} Data)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"Uni_superimposed_{split}.png"), dpi=300)
    plt.close()

# Scatter Plots (Train, Val, Test)
for X, y, split in [(Xtr_u_s, ytr_u, 'Train'), (Xval_u_s, yval_u, 'Val'), (Xte_u_s, yte_u, 'Test')]:
    plt.figure(figsize=(5, 5))
    preds = best_uni_model.predict(X)
    plt.scatter(y, preds, alpha=0.5, s=15, c='teal')
    lo, hi = min(y.min(), preds.min()), max(y.max(), preds.max())
    plt.plot([lo, hi], [lo, hi], 'k--', lw=1.5, label='Ideal')
    plt.xlabel('Target Output (True y)')
    plt.ylabel('Model Output (Predicted y)')
    plt.title(f'Univariate Best {best_uni_cfg} - Scatter ({split})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"Uni_scatter_{split}.png"), dpi=300)
    plt.close()

# Hidden and Output Node Plots (1D Input -> 2D Curves)
for split_name, X_split in [('Train', Xtr_u_s), ('Val', Xval_u_s), ('Test', Xte_u_s)]:
    xs = np.linspace(X_split[:, 0].min() - 0.2, X_split[:, 0].max() + 0.2, 300)[:, None]
    As = best_uni_model._forward(xs)
    n_hidden_nodes = best_uni_model.layer_sizes[1]
    for node in range(n_hidden_nodes):
        plt.figure(figsize=(5, 4))
        plt.plot(xs[:, 0], As[1][:, node], 'b-', lw=2)
        plt.scatter(X_split[:, 0], best_uni_model._forward(X_split)[1][:, node], c='k', s=8, alpha=0.4)
        plt.xlabel('x (Standardized)')
        plt.ylabel(f'H1N{node} Output')
        plt.title(f'Univariate Best ({split_name}) - Hidden 1 Node {node}')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, f"Uni_{split_name}_H1_N{node}.png"), dpi=200)
        plt.close()

r_te_u, p_te_u = rmse_metrics(yte_u, best_uni_model.predict(Xte_u_s))
print(f"\nBest Univariate Architecture {best_uni_cfg} -> Test RMSE: {r_te_u:.4f} ({p_te_u:.2f}%)")

# ----------------- DATASET 2: BIVARIATE REGRESSION -----------------
print("\n----------------------------------------------------------------------")
print("2. EVALUATING ALL ARCHITECTURES ON VALIDATION SET (DATASET 2: BIVARIATE)")
print("----------------------------------------------------------------------")
bi_configs = [
    [2, 2, 1], [2, 4, 1], [2, 8, 1], [2, 16, 1],
    [2, 4, 4, 1], [2, 8, 4, 1], [2, 8, 8, 1], [2, 16, 8, 1]
]
bi_results = []

for cfg in bi_configs:
    np.random.seed(0)
    m = FCNNReg(cfg, hidden_act='tanh', lr=0.01, epochs=1200, tol=1e-5)
    m.train_sgd(Xtr_b_s, ytr_b)
    r_tr, p_tr = rmse_metrics(ytr_b, m.predict(Xtr_b_s))
    r_val, p_val = rmse_metrics(yval_b, m.predict(Xval_b_s))
    bi_results.append((cfg, m, r_tr, p_tr, r_val, p_val))
    print(f"Architecture {cfg} -> Train RMSE: {r_tr:.4f} ({p_tr:.2f}%) | Val RMSE: {r_val:.4f} ({p_val:.2f}%)")

best_bi_item = min(bi_results, key=lambda x: (x[4], x[1].count_parameters()))
best_bi_cfg, best_bi_model = best_bi_item[0], best_bi_item[1]

print("=" * 70)
print(f"SELECTED BEST BIVARIATE ARCHITECTURE: {best_bi_cfg} (Val RMSE: {best_bi_item[4]:.4f})")
print("=" * 70)

summary_bi = []
for item in bi_results:
    cfg, m, r_tr, p_tr, r_val, p_val = item
    summary_bi.append({
        "Architecture": str(cfg),
        "Hidden Layers": len(cfg) - 2,
        "Parameters": m.count_parameters(),
        "Train RMSE": f"{r_tr:.4f}",
        "Train %RMSE": f"{p_tr:.2f}%",
        "Val RMSE": f"{r_val:.4f}",
        "Val %RMSE": f"{p_val:.2f}%"
    })
print("\nValidation Summary Table (Bivariate Regression):")
print(pd.DataFrame(summary_bi).to_string(index=False))

# Learning Rate Comparison
plt.figure(figsize=(7, 4))
for lr in [0.01, 0.05, 0.08]:
    np.random.seed(0)
    m = FCNNReg(best_bi_cfg, hidden_act='tanh', lr=lr, epochs=1200, tol=1e-5)
    m.train_sgd(Xtr_b_s, ytr_b)
    plt.plot(m.loss_history, label=f'lr = {lr} (Stopped at epoch {len(m.loss_history)})')
plt.title(f'Bivariate Best {best_bi_cfg}: Learning Rate Comparison')
plt.xlabel('Epochs')
plt.ylabel('Average Squared Error')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "Bi_lr_comparison.png"), dpi=300)
plt.close()

# Average Error vs Epochs
plt.figure(figsize=(6, 4))
plt.plot(best_bi_model.loss_history, color='darkgreen', lw=2)
plt.title(f'Bivariate Best {best_bi_cfg}: Average Error vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Average Squared Error')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "Bi_best_error_vs_epochs.png"), dpi=300)
plt.close()

# 3D Superimposed Outputs (Train, Val, Test)
for X, y, split in [(Xtr_b_s, ytr_b, 'Train'), (Xval_b_s, yval_b, 'Val'), (Xte_b_s, yte_b, 'Test')]:
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], y, c='r', s=10, label='Target Output', alpha=0.7)
    gx, gy = np.meshgrid(np.linspace(X[:, 0].min(), X[:, 0].max(), 25), np.linspace(X[:, 1].min(), X[:, 1].max(), 25))
    gz = best_bi_model.predict(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)
    ax.plot_surface(gx, gy, gz, alpha=0.5, cmap='viridis')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('y')
    ax.set_title(f'Bivariate Best {best_bi_cfg} - Superimposed ({split})')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"Bi_superimposed_{split}.png"), dpi=200)
    plt.close()

# Scatter Plots (Train, Val, Test)
for X, y, split in [(Xtr_b_s, ytr_b, 'Train'), (Xval_b_s, yval_b, 'Val'), (Xte_b_s, yte_b, 'Test')]:
    plt.figure(figsize=(5, 5))
    preds = best_bi_model.predict(X)
    plt.scatter(y, preds, alpha=0.5, s=15, c='darkorange')
    lo, hi = min(y.min(), preds.min()), max(y.max(), preds.max())
    plt.plot([lo, hi], [lo, hi], 'k--', lw=1.5, label='Ideal')
    plt.xlabel('Target Output (True y)')
    plt.ylabel('Model Output (Predicted y)')
    plt.title(f'Bivariate Best {best_bi_cfg} - Scatter ({split})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"Bi_scatter_{split}.png"), dpi=300)
    plt.close()

# 3D Node Surface Plots for Bivariate (2D Input -> 3D Surface)
for split_name, X_split, y_split in [('Train', Xtr_b_s, ytr_b), ('Val', Xval_b_s, yval_b), ('Test', Xte_b_s, yte_b)]:
    gx, gy = np.meshgrid(np.linspace(X_split[:, 0].min(), X_split[:, 0].max(), 25), np.linspace(X_split[:, 1].min(), X_split[:, 1].max(), 25))
    grid = np.c_[gx.ravel(), gy.ravel()]
    n_hidden_layers = len(best_bi_model.layer_sizes) - 2
    for h in range(1, n_hidden_layers + 1):
        for node in range(best_bi_model.layer_sizes[h]):
            fig = plt.figure(figsize=(5, 4))
            ax = fig.add_subplot(111, projection='3d')
            gz = best_bi_model._forward(grid)[h][:, node].reshape(gx.shape)
            ax.plot_surface(gx, gy, gz, alpha=0.6, cmap='viridis')
            ax.scatter(X_split[:, 0], X_split[:, 1], best_bi_model._forward(X_split)[h][:, node], c='k', s=8, alpha=0.5)
            ax.set_xlabel('x1')
            ax.set_ylabel('x2')
            ax.set_zlabel(f'H{h}N{node}')
            ax.set_title(f'Bivariate Best ({split_name}) - Hidden {h} Node {node}')
            plt.tight_layout()
            plt.savefig(os.path.join(PLOT_DIR, f"Bi_{split_name}_H{h}_N{node}.png"), dpi=200)
            plt.close()

r_te_b, p_te_b = rmse_metrics(yte_b, best_bi_model.predict(Xte_b_s))
print(f"\nBest Bivariate Architecture {best_bi_cfg} -> Test RMSE: {r_te_b:.4f} ({p_te_b:.2f}%)")

# ----------------- COMPARISON WITH SINGLE-NEURON LINEAR BASELINE -----------------
print("\n" + "=" * 70)
print("3. COMPARISON WITH SINGLE-NEURON MODEL BASELINE (ASSIGNMENT 1)")
print("=" * 70)

class SingleNeuronLinearPerceptron:
    def __init__(self, d, lr=0.01, epochs=1200):
        self.w = np.random.randn(d + 1) * 0.01
        self.lr = lr
        self.epochs = epochs
    def train(self, X, y):
        n = len(X); Xa = np.c_[np.ones(n), X]
        for _ in range(self.epochs):
            idx = np.random.permutation(n)
            for j in idx:
                self.w += self.lr * (y[j] - Xa[j] @ self.w) * Xa[j]
    def predict(self, X):
        return np.c_[np.ones(len(X)), X] @ self.w

sn_uni = SingleNeuronLinearPerceptron(1); sn_uni.train(Xtr_u_s, ytr_u)
r_sn_u, p_sn_u = rmse_metrics(yte_u, sn_uni.predict(Xte_u_s))

sn_bi = SingleNeuronLinearPerceptron(2); sn_bi.train(Xtr_b_s, ytr_b)
r_sn_b, p_sn_b = rmse_metrics(yte_b, sn_bi.predict(Xte_b_s))

reg_comp_df = pd.DataFrame([
    {
        "Dataset": "Univariate",
        "Single-Neuron Baseline Test RMSE": f"{r_sn_u:.4f} ({p_sn_u:.2f}%)",
        "Best FCNN Architecture": str(best_uni_cfg),
        "FCNN Test RMSE": f"{r_te_u:.4f} ({p_te_u:.2f}%)"
    },
    {
        "Dataset": "Bivariate",
        "Single-Neuron Baseline Test RMSE": f"{r_sn_b:.4f} ({p_sn_b:.2f}%)",
        "Best FCNN Architecture": str(best_bi_cfg),
        "FCNN Test RMSE": f"{r_te_b:.4f} ({p_te_b:.2f}%)"
    }
])
print(reg_comp_df.to_string(index=False))
print(f"\nAll regression figures saved in: {PLOT_DIR}")