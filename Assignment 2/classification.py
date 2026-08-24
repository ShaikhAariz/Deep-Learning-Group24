# Generated from classification.ipynb
# Run from the Assignment 2 directory so relative data paths resolve.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — needed for 3d projection
np.random.seed(0)

class FCNN:
    def __init__(self, layer_sizes, hidden_act='tanh', out_act='logistic', lr=0.01, epochs=500):
        self.layer_sizes = layer_sizes
        self.hidden_act = hidden_act
        self.out_act = out_act
        self.lr = lr
        self.epochs = epochs
        self.loss_history = []
        # Xavier-like init
        self.W = []
        self.b = []
        for i in range(len(layer_sizes)-1):
            fan_in = layer_sizes[i]
            self.W.append(np.random.randn(fan_in, layer_sizes[i+1]) * np.sqrt(1/fan_in))
            self.b.append(np.zeros(layer_sizes[i+1]))

    def _act(self, z, kind):
        if kind == 'tanh':
            return np.tanh(z)
        if kind == 'logistic':
            return 1/(1+np.exp(-z))
        return z  # linear

    def _dact(self, a, kind):
        if kind == 'tanh':
            return 1 - a**2
        if kind == 'logistic':
            return a*(1-a)
        return np.ones_like(a)

    def _forward(self, x):
        # x: (1, D) or (N, D); returns list of activations, last is output
        As = [x]
        Zs = []
        A = x
        for i in range(len(self.W)):
            Z = A @ self.W[i] + self.b[i]
            Zs.append(Z)
            kind = self.out_act if i == len(self.W)-1 else self.hidden_act
            A = self._act(Z, kind)
            As.append(A)
        return As, Zs

    def _forward_single(self, x):
        As, _ = self._forward(x[None, :])
        return [a[0] for a in As]

    def train_sgd(self, X, y_onehot):
        n = X.shape[0]
        self.loss_history = []
        for epoch in range(self.epochs):
            idx = np.random.permutation(n)
            for j in idx:
                x = X[j:j+1]  # (1, D)
                target = y_onehot[j:j+1]  # (1, C)
                As, Zs = self._forward(x)
                # squared error: 0.5*||target - out||^2 -> dL/da_out = (out - target)
                dA = As[-1] - target  # (1, C)
                # backprop
                for i in range(len(self.W)-1, -1, -1):
                    kind = self.out_act if i == len(self.W)-1 else self.hidden_act
                    dZ = dA * self._dact(As[i+1], kind)  # (1, size_out)
                    dW = As[i].T @ dZ  # (fan_in, fan_out)
                    db = dZ[0]
                    # update
                    self.W[i] -= self.lr * dW
                    self.b[i] -= self.lr * db
                    if i > 0:
                        dA = dZ @ self.W[i].T
            # average squared error for this epoch
            As_full, _ = self._forward(X)
            err = y_onehot - As_full[-1]
            self.loss_history.append(np.mean(0.5 * err**2))

    def predict(self, X):
        As, _ = self._forward(X)
        return np.argmax(As[-1], axis=1)

    def hidden_outputs(self, X, layer=1):
        # layer=1 -> first hidden, layer=2 -> second hidden (if exists)
        As, _ = self._forward(X)
        return As[layer]

def prepare_data_60_20_20(class_data_list):
    X_tr, y_tr, X_val, y_val, X_te, y_te = [], [], [], [], [], []
    for label, data in enumerate(class_data_list):
        n = data.shape[0]
        n_tr = int(n*0.6)
        n_val = int(n*0.2)
        X_tr.append(data[:n_tr]); y_tr.append(np.full(n_tr, label))
        X_val.append(data[n_tr:n_tr+n_val]); y_val.append(np.full(n_val, label))
        X_te.append(data[n_tr+n_val:]); y_te.append(np.full(n - n_tr - n_val, label))
    return (np.vstack(X_tr), np.concatenate(y_tr),
            np.vstack(X_val), np.concatenate(y_val),
            np.vstack(X_te), np.concatenate(y_te))

def to_onehot(y, C=3):
    oh = np.zeros((len(y), C)); oh[np.arange(len(y)), y.astype(int)] = 1; return oh

def plot_decision_regions(X, y, model, title):
    plt.figure(figsize=(6,5))
    x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
    y_min, y_max = X[:,1].min()-1, X[:,1].max()+1
    xx, yy = np.meshgrid(np.arange(x_min,x_max,0.05), np.arange(y_min,y_max,0.05))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid).reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    plt.scatter(X[:,0], X[:,1], c=y, edgecolor='k', cmap='coolwarm', s=20)
    plt.title(title); plt.xlabel('Feature 1'); plt.ylabel('Feature 2'); plt.tight_layout(); plt.show()

def evaluate_metrics(y_true, y_pred, C=3):
    cm = np.zeros((C,C), int)
    for t,p in zip(y_true, y_pred): cm[int(t), int(p)] += 1
    acc = np.trace(cm)/cm.sum()
    precs, recs, f1s = [], [], []
    for i in range(C):
        tp=cm[i,i]; fp=cm[:,i].sum()-tp; fn=cm[i,:].sum()-tp
        prec=tp/(tp+fp) if tp+fp>0 else 0; rec=tp/(tp+fn) if tp+fn>0 else 0
        f1=2*prec*rec/(prec+rec) if prec+rec>0 else 0
        precs.append(prec); recs.append(rec); f1s.append(f1)
    print(f"Confusion Matrix:\n{cm}")
    print(f"Accuracy: {acc*100:.2f}%")
    for i in range(C): print(f"Class {i} -> Precision: {precs[i]:.2f}, Recall: {recs[i]:.2f}, F1: {f1s[i]:.2f}")
    print(f"Mean Precision: {np.mean(precs):.2f}  Mean Recall: {np.mean(recs):.2f}  Mean F1: {np.mean(f1s):.2f}\n")
    return acc, cm

def plot_hidden_surfaces(model, X, y, split_name):
    # for each hidden layer, plot each node's 3D surface over input space (x1,x2 -> activation)
    # plus scatter of actual data points' activations
    n_hidden_layers = len(model.layer_sizes)-2  # e.g. [2,8,3] -> 1, [2,8,8,3] -> 2
    for h in range(1, n_hidden_layers+1):
        n_nodes = model.layer_sizes[h]
        for node in range(n_nodes):
            fig = plt.figure(figsize=(6,5))
            ax = fig.add_subplot(111, projection='3d')
            # surface over grid
            x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
            y_min, y_max = X[:,1].min()-1, X[:,1].max()+1
            gx, gy = np.meshgrid(np.linspace(x_min,x_max,30), np.linspace(y_min,y_max,30))
            grid = np.c_[gx.ravel(), gy.ravel()]
            As,_ = model._forward(grid)
            gz = As[h][:, node].reshape(gx.shape)
            ax.plot_surface(gx, gy, gz, alpha=0.6, cmap='viridis')
            # scatter actual points
            As_data,_ = model._forward(X)
            ax.scatter(X[:,0], X[:,1], As_data[h][:, node], c=y, cmap='coolwarm', s=10, edgecolor='k', alpha=0.8)
            ax.set_xlabel('Feature 1'); ax.set_ylabel('Feature 2'); ax.set_zlabel(f'H{h}N{node} output')
            ax.set_title(f'{split_name} — Hidden layer {h} node {node}')
            plt.tight_layout(); plt.show()
    # output nodes (one per class)
    for out in range(model.layer_sizes[-1]):
        fig = plt.figure(figsize=(6,5))
        ax = fig.add_subplot(111, projection='3d')
        x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
        y_min, y_max = X[:,1].min()-1, X[:,1].max()+1
        gx, gy = np.meshgrid(np.linspace(x_min,x_max,30), np.linspace(y_min,y_max,30))
        grid = np.c_[gx.ravel(), gy.ravel()]
        As,_ = model._forward(grid)
        gz = As[-1][:, out].reshape(gx.shape)
        ax.plot_surface(gx, gy, gz, alpha=0.6, cmap='coolwarm')
        As_data,_ = model._forward(X)
        ax.scatter(X[:,0], X[:,1], As_data[-1][:, out], c=y, cmap='coolwarm', s=10, edgecolor='k', alpha=0.8)
        ax.set_xlabel('Feature 1'); ax.set_ylabel('Feature 2'); ax.set_zlabel(f'Output {out}')
        ax.set_title(f'{split_name} — Output node {out}')
        plt.tight_layout(); plt.show()

base = 'data/Classification'
ls_c1 = np.loadtxt(f'{base}/LS_Group24/Class1.txt')
ls_c2 = np.loadtxt(f'{base}/LS_Group24/Class2.txt')
ls_c3 = np.loadtxt(f'{base}/LS_Group24/Class3.txt')
nls_data = np.loadtxt(f'{base}/NLS_Group24.txt', skiprows=1)
nls_c1, nls_c2, nls_c3 = nls_data[:300], nls_data[300:800], nls_data[800:1800]

Xtr_ls, ytr_ls, Xval_ls, yval_ls, Xte_ls, yte_ls = prepare_data_60_20_20([ls_c1, ls_c2, ls_c3])
Xtr_nls, ytr_nls, Xval_nls, yval_nls, Xte_nls, yte_nls = prepare_data_60_20_20([nls_c1, nls_c2, nls_c3])
print(f"LS  — train {Xtr_ls.shape} val {Xval_ls.shape} test {Xte_ls.shape}")
print(f"NLS — train {Xtr_nls.shape} val {Xval_nls.shape} test {Xte_nls.shape}")

ls_configs = [[2, h, 3] for h in [4, 8, 16]]
nls_configs = [[2, h1, h2, 3] for h1,h2 in [(4,4),(8,8),(16,8),(8,4)]]

# hyper-parameters (tweak these cells to experiment)
LR = 0.01
EPOCHS = 800
HIDDEN_ACT = 'tanh'  # tanh or logistic

print("LS configs:", ls_configs)
print("NLS configs:", nls_configs)
print(f"lr={LR} epochs={EPOCHS} hidden_act={HIDDEN_ACT}")

ls_results = []
for cfg in ls_configs:
    np.random.seed(0)
    m = FCNN(cfg, hidden_act=HIDDEN_ACT, out_act='logistic', lr=LR, epochs=EPOCHS)
    m.train_sgd(Xtr_ls, to_onehot(ytr_ls))
    ypred_val = m.predict(Xval_ls)
    acc, _ = evaluate_metrics(yval_ls, ypred_val)
    ls_results.append((cfg, m, acc))
    print(f"Config {cfg} -> val acc {acc*100:.2f}%  final train loss {m.loss_history[-1]:.4f}")

best_ls_cfg, best_ls_model, best_ls_acc = max(ls_results, key=lambda x: x[2])
print(f"\nBest LS config: {best_ls_cfg}  val acc {best_ls_acc*100:.2f}%")

nls_results = []
for cfg in nls_configs:
    np.random.seed(0)
    m = FCNN(cfg, hidden_act=HIDDEN_ACT, out_act='logistic', lr=LR, epochs=EPOCHS)
    m.train_sgd(Xtr_nls, to_onehot(ytr_nls))
    ypred_val = m.predict(Xval_nls)
    acc, _ = evaluate_metrics(yval_nls, ypred_val)
    nls_results.append((cfg, m, acc))
    print(f"Config {cfg} -> val acc {acc*100:.2f}%  final train loss {m.loss_history[-1]:.4f}")

best_nls_cfg, best_nls_model, best_nls_acc = max(nls_results, key=lambda x: x[2])
print(f"\nBest NLS config: {best_nls_cfg}  val acc {best_nls_acc*100:.2f}%")

plt.figure(figsize=(6,4))
plt.plot(best_ls_model.loss_history, label=f'LS {best_ls_cfg}')
plt.title('LS — Best Architecture: Average Squared Error vs Epochs (train)')
plt.xlabel('Epochs'); plt.ylabel('0.5 * mean((y - out)^2)'); plt.legend(); plt.show()

plt.figure(figsize=(6,4))
plt.plot(best_nls_model.loss_history, label=f'NLS {best_nls_cfg}')
plt.title('NLS — Best Architecture: Average Squared Error vs Epochs (train)')
plt.xlabel('Epochs'); plt.ylabel('0.5 * mean((y - out)^2)'); plt.legend(); plt.show()

plot_decision_regions(Xtr_ls, ytr_ls, best_ls_model, f'LS Decision Regions — Best {best_ls_cfg} (train)')
plot_decision_regions(Xtr_nls, ytr_nls, best_nls_model, f'NLS Decision Regions — Best {best_nls_cfg} (train)')

print("=== LS Best — Test ===")
evaluate_metrics(yte_ls, best_ls_model.predict(Xte_ls))
print("=== NLS Best — Test ===")
evaluate_metrics(yte_nls, best_nls_model.predict(Xte_nls))

print("LS — train")
plot_hidden_surfaces(best_ls_model, Xtr_ls, ytr_ls, 'LS train')
print("LS — val")
plot_hidden_surfaces(best_ls_model, Xval_ls, yval_ls, 'LS val')
print("LS — test")
plot_hidden_surfaces(best_ls_model, Xte_ls, yte_ls, 'LS test')

print("NLS — train")
plot_hidden_surfaces(best_nls_model, Xtr_nls, ytr_nls, 'NLS train')
print("NLS — val")
plot_hidden_surfaces(best_nls_model, Xval_nls, yval_nls, 'NLS val')
print("NLS — test")
plot_hidden_surfaces(best_nls_model, Xte_nls, yte_nls, 'NLS test')

# Single-neuron OVO baseline (from Assignment 1) — re-evaluated on the 60/20/20 split
class Perceptron:
    def __init__(self, d, act='logistic', lr=0.05, epochs=1000):
        self.w=np.random.randn(d+1)*0.01; self.act=act; self.lr=lr; self.epochs=epochs
    def _a(self,z): return 1/(1+np.exp(-z)) if self.act=='logistic' else np.tanh(z)
    def _da(self,a): return a*(1-a) if self.act=='logistic' else 1-a**2
    def train(self,X,y):
        n=len(X); Xa=np.c_[np.ones(n),X]
        for _ in range(self.epochs):
            z=Xa@self.w; a=self._a(z); e=y-a
            self.w -= self.lr*(Xa.T@(-e*self._da(a)))/n
    def predict(self,X):
        a=self._a(np.c_[np.ones(len(X)),X]@self.w)
        return np.where(a>=0.5,1,0) if self.act=='logistic' else np.where(a>=0,1,-1)
class OVO:
    def __init__(self, act='logistic'):
        self.act=act; self.models={}
    def train(self,X,y):
        for i in range(3):
            for j in range(i+1,3):
                idx=np.where((y==i)|(y==j))[0]; Xp,yp=X[idx],y[idx]
                yt=np.where(yp==i,1,0) if self.act=='logistic' else np.where(yp==i,1,-1)
                m=Perceptron(X.shape[1],self.act,lr=0.05,epochs=1000); m.train(Xp,yt); self.models[(i,j)]=m
    def predict(self,X):
        v=np.zeros((len(X),3))
        for (i,j),m in self.models.items():
            p=m.predict(X)
            for k in range(len(X)): v[k, i if p[k]==1 else j]+=1
        return np.argmax(v,axis=1)

for act in ['logistic','tanh']:
    print(f"\n--- Single-neuron OVO ({act}) — LS val ---")
    np.random.seed(0); m=OVO(act); m.train(Xtr_ls, ytr_ls); evaluate_metrics(yval_ls, m.predict(Xval_ls))
    print(f"--- Single-neuron OVO ({act}) — NLS val ---")
    np.random.seed(0); m=OVO(act); m.train(Xtr_nls, ytr_nls); evaluate_metrics(yval_nls, m.predict(Xval_nls))

print(f"FCNN best LS  val: {best_ls_acc*100:.2f}%  (config {best_ls_cfg})")
print(f"FCNN best NLS val: {best_nls_acc*100:.2f}%  (config {best_nls_cfg})")

