# Generated from regression.ipynb
# Run from the Assignment 2 directory so relative data paths resolve.

import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
np.random.seed(0)

class FCNNReg:
    def __init__(self, layer_sizes, hidden_act='tanh', lr=0.01, epochs=800):
        self.layer_sizes = layer_sizes
        self.hidden_act = hidden_act
        self.lr = lr
        self.epochs = epochs
        self.loss_history = []
        self.W = [np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(1/layer_sizes[i])
                for i in range(len(layer_sizes)-1)]
        self.b = [np.zeros(layer_sizes[i+1]) for i in range(len(layer_sizes)-1)]

    def _act(self, z, kind):
        if kind == 'tanh': return np.tanh(z)
        if kind == 'logistic': return 1/(1+np.exp(-z))
        return z
    def _dact(self, a, kind):
        if kind == 'tanh': return 1 - a**2
        if kind == 'logistic': return a*(1-a)
        return np.ones_like(a)

    def _forward(self, X):
        As=[X]; A=X
        for i in range(len(self.W)):
            Z = A @ self.W[i] + self.b[i]
            kind = 'linear' if i==len(self.W)-1 else self.hidden_act
            A = self._act(Z, kind)
            As.append(A)
        return As

    def train_sgd(self, X, y):
        y = y.reshape(-1,1) if y.ndim==1 else y
        n=len(X); self.loss_history=[]
        for epoch in range(self.epochs):
            idx=np.random.permutation(n)
            for j in idx:
                x=X[j:j+1]; t=y[j:j+1]
                As=self._forward(x)
                dA = As[-1] - t  # squared error gradient
                for i in range(len(self.W)-1,-1,-1):
                    kind='linear' if i==len(self.W)-1 else self.hidden_act
                    dZ = dA * self._dact(As[i+1], kind)
                    dW = As[i].T @ dZ
                    db = dZ[0]
                    self.W[i] -= self.lr * dW
                    self.b[i] -= self.lr * db
                    if i>0: dA = dZ @ self.W[i].T
            As_full=self._forward(X)
            err = y - As_full[-1]
            self.loss_history.append(np.mean(0.5*err**2))

    def predict(self, X):
        return self._forward(X)[-1].ravel()

    def hidden_outputs(self, X, layer=1):
        return self._forward(X)[layer]

def load_or_synthesize_regression(path_24, kind='uni'):
    if os.path.exists(path_24):
        data = pd.read_csv(path_24, header=None).values
        X, y = data[:,:-1], data[:,-1]
        print(f"Loaded {path_24}: {X.shape}")
        return X, y
    print(f"{path_24} not found — synthesizing placeholder {kind} data so notebook stays runnable. Replace with real 24.csv.")
    n=600; rng=np.random.default_rng(24)
    if kind=='uni':
        X=rng.uniform(-2,2,(n,1)); y=0.8*X[:,0]**3 - 1.5*X[:,0] + rng.normal(0,0.3,n)
    else:
        X=rng.uniform(-2,2,(n,2)); y=np.sin(X[:,0])*np.cos(X[:,1]) + 0.5*X[:,0] + rng.normal(0,0.2,n)
    return X, y

def split_60_20_20(X, y, seed=0):
    rng=np.random.default_rng(seed); idx=rng.permutation(len(X))
    n=len(X); n_tr=int(n*0.6); n_val=int(n*0.2)
    tr=idx[:n_tr]; val=idx[n_tr:n_tr+n_val]; te=idx[n_tr+n_val:]
    return X[tr], y[tr], X[val], y[val], X[te], y[te], idx

def rmse_metrics(y_true, y_pred):
    err=y_true - y_pred; r=np.sqrt(np.mean(err**2)); pct=r/(np.abs(np.mean(y_true))+1e-8)*100
    return r, pct

def plot_loss(history, title):
    plt.figure(figsize=(6,4)); plt.plot(history); plt.title(title)
    plt.xlabel('Epochs'); plt.ylabel('0.5 * mean((y - pred)^2)'); plt.show()

def plot_regression_fit_uni(Xtr,ytr,Xval,yval,Xte,yte, model, split_name=''):
    for X,y,name in [(Xtr,ytr,'train'),(Xval,yval,'val'),(Xte,yte,'test')]:
        plt.figure(figsize=(6,4))
        order=np.argsort(X[:,0]); plt.scatter(X[:,0], y, c='r', s=10, label='target', alpha=0.6)
        # dense line for model
        xs=np.linspace(X[:,0].min()-0.5, X[:,0].max()+0.5, 300)[:,None]
        # need original scale: if X was standardized, invert. For placeholder we keep raw.
        # Here X passed is already the model input scale, so plot in that scale.
        ys=model.predict(xs)
        # scatter model predictions on actual points lightly
        plt.plot(xs[:,0], ys, 'b-', lw=2, label='model')
        plt.xlabel('x'); plt.ylabel('y'); plt.title(f'Univariate {split_name} — {name}'); plt.legend(); plt.show()

def plot_regression_fit_bi(Xtr,ytr,Xval,yval,Xte,yte, model):
    for X,y,name in [(Xtr,ytr,'train'),(Xval,yval,'val'),(Xte,yte,'test')]:
        fig=plt.figure(figsize=(7,5)); ax=fig.add_subplot(111, projection='3d')
        ax.scatter(X[:,0], X[:,1], y, c='r', s=10, label='target', alpha=0.7)
        # surface of model
        gx, gy=np.meshgrid(np.linspace(X[:,0].min()-0.5,X[:,0].max()+0.5,25), np.linspace(X[:,1].min()-0.5,X[:,1].max()+0.5,25))
        gz=model.predict(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)
        ax.plot_surface(gx, gy, gz, alpha=0.5, cmap='viridis')
        ax.set_xlabel('x1'); ax.set_ylabel('x2'); ax.set_zlabel('y')
        ax.set_title(f'Bivariate — {name}'); plt.show()

def scatter_target_vs_pred(y_true, y_pred, title):
    plt.figure(figsize=(5,5)); plt.scatter(y_true, y_pred, alpha=0.5, s=12)
    lo=min(y_true.min(), y_pred.min()); hi=max(y_true.max(), y_pred.max())
    plt.plot([lo,hi],[lo,hi],'k--'); plt.xlabel('target y'); plt.ylabel('model y')
    plt.title(title); plt.show()

def plot_hidden_surfaces_reg(model, X, y, split_name):
    n_hidden=len(model.layer_sizes)-2
    for h in range(1, n_hidden+1):
        n_nodes=model.layer_sizes[h]
        for node in range(n_nodes):
            if X.shape[1]==1:
                plt.figure(figsize=(6,4))
                xs=np.linspace(X[:,0].min()-0.5, X[:,0].max()+0.5, 300)[:,None]
                As=model._forward(xs)
                plt.plot(xs[:,0], As[h][:,node], 'b-', lw=2)
                # scatter actual
                As_data=model._forward(X)
                plt.scatter(X[:,0], As_data[h][:,node], c='k', s=10, alpha=0.5)
                plt.xlabel('x'); plt.ylabel(f'H{h}N{node} output')
                plt.title(f'{split_name} — Hidden {h} node {node}'); plt.show()
            else:
                fig=plt.figure(figsize=(6,5)); ax=fig.add_subplot(111, projection='3d')
                gx,gy=np.meshgrid(np.linspace(X[:,0].min()-0.5,X[:,0].max()+0.5,25), np.linspace(X[:,1].min()-0.5,X[:,1].max()+0.5,25))
                grid=np.c_[gx.ravel(), gy.ravel()]; gz=model._forward(grid)[h][:,node].reshape(gx.shape)
                ax.plot_surface(gx,gy,gz, alpha=0.6, cmap='viridis')
                As_data=model._forward(X)
                ax.scatter(X[:,0], X[:,1], As_data[h][:,node], c='k', s=10, alpha=0.6)
                ax.set_xlabel('x1'); ax.set_ylabel('x2'); ax.set_zlabel(f'H{h}N{node}')
                ax.set_title(f'{split_name} — Hidden {h} node {node}'); plt.show()
    # output node
    if X.shape[1]==1:
        plt.figure(figsize=(6,4))
        xs=np.linspace(X[:,0].min()-0.5, X[:,0].max()+0.5, 300)[:,None]
        plt.plot(xs[:,0], model.predict(xs), 'r-', lw=2, label='output')
        plt.scatter(X[:,0], y, c='k', s=10, alpha=0.4, label='target')
        plt.xlabel('x'); plt.ylabel('y'); plt.title(f'{split_name} — Output node'); plt.legend(); plt.show()
    else:
        fig=plt.figure(figsize=(7,5)); ax=fig.add_subplot(111, projection='3d')
        gx,gy=np.meshgrid(np.linspace(X[:,0].min()-0.5,X[:,0].max()+0.5,25), np.linspace(X[:,1].min()-0.5,X[:,1].max()+0.5,25))
        gz=model.predict(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)
        ax.plot_surface(gx,gy,gz, alpha=0.6, cmap='coolwarm')
        ax.scatter(X[:,0], X[:,1], y, c='r', s=10, alpha=0.6)
        ax.set_xlabel('x1'); ax.set_ylabel('x2'); ax.set_zlabel('y')
        ax.set_title(f'{split_name} — Output node'); plt.show()

base1='data/Regression/UnivariateData/24.csv'
base2='data/Regression/BivariateData/24.csv'
X_uni_raw, y_uni_raw = load_or_synthesize_regression(base1, 'uni')
X_bi_raw,  y_bi_raw  = load_or_synthesize_regression(base2, 'bi')

def standardize_train_val_test(Xtr, Xval, Xte):
    mu=Xtr.mean(axis=0); sd=Xtr.std(axis=0)+1e-8
    return (Xtr-mu)/sd, (Xval-mu)/sd, (Xte-mu)/sd, mu, sd

Xtr_uni, ytr_uni, Xval_uni, yval_uni, Xte_uni, yte_uni, _ = split_60_20_20(X_uni_raw, y_uni_raw, seed=0)
Xtr_bi,  ytr_bi,  Xval_bi,  yval_bi,  Xte_bi,  yte_bi,  _ = split_60_20_20(X_bi_raw,  y_bi_raw,  seed=0)
Xtr_uni_s, Xval_uni_s, Xte_uni_s, mu_uni, sd_uni = standardize_train_val_test(Xtr_uni, Xval_uni, Xte_uni)
Xtr_bi_s,  Xval_bi_s,  Xte_bi_s,  mu_bi,  sd_bi  = standardize_train_val_test(Xtr_bi,  Xval_bi,  Xte_bi)
print(f"Uni — train {Xtr_uni_s.shape} val {Xval_uni_s.shape} test {Xte_uni_s.shape}")
print(f"Bi  — train {Xtr_bi_s.shape} val {Xval_bi_s.shape} test {Xte_bi_s.shape}")

uni_configs_1h  = [[Xtr_uni_s.shape[1], h, 1] for h in [2,4,8,16]]
bi_configs_1h   = [[Xtr_bi_s.shape[1],  h, 1] for h in [2,4,8,16]]
bi_configs_2h   = [[Xtr_bi_s.shape[1], h1, h2, 1] for h1,h2 in [(4,4),(8,4),(8,8),(16,8)]]
LR=0.01; EPOCHS=1200; HACT='tanh'
print("uni 1h:", uni_configs_1h)
print("bi  1h:", bi_configs_1h)
print("bi  2h:", bi_configs_2h)
print(f"lr={LR} epochs={EPOCHS} hidden_act={HACT}")

uni_results=[]
for cfg in uni_configs_1h:
    np.random.seed(0); m=FCNNReg(cfg, hidden_act=HACT, lr=LR, epochs=EPOCHS)
    m.train_sgd(Xtr_uni_s, ytr_uni)
    r_tr,_=rmse_metrics(ytr_uni, m.predict(Xtr_uni_s)); r_val,_=rmse_metrics(yval_uni, m.predict(Xval_uni_s))
    uni_results.append((cfg,m,r_tr,r_val))
    print(f"{cfg} -> train RMSE {r_tr:.4f}  val RMSE {r_val:.4f}  final loss {m.loss_history[-1]:.5f}")
best_uni_cfg, best_uni_model, _, best_uni_val = min(uni_results, key=lambda x: x[3])
print(f"\nBest Uni 1h: {best_uni_cfg}  val RMSE {best_uni_val:.4f}")

bi1_results=[]
for cfg in bi_configs_1h:
    np.random.seed(0); m=FCNNReg(cfg, hidden_act=HACT, lr=LR, epochs=EPOCHS)
    m.train_sgd(Xtr_bi_s, ytr_bi)
    r_tr,_=rmse_metrics(ytr_bi, m.predict(Xtr_bi_s)); r_val,_=rmse_metrics(yval_bi, m.predict(Xval_bi_s))
    bi1_results.append((cfg,m,r_tr,r_val))
    print(f"{cfg} -> train RMSE {r_tr:.4f}  val RMSE {r_val:.4f}  final loss {m.loss_history[-1]:.5f}")
best_bi1_cfg, best_bi1_model, _, best_bi1_val = min(bi1_results, key=lambda x: x[3])
print(f"\nBest Bi 1h: {best_bi1_cfg}  val RMSE {best_bi1_val:.4f}")

bi2_results=[]
for cfg in bi_configs_2h:
    np.random.seed(0); m=FCNNReg(cfg, hidden_act=HACT, lr=LR, epochs=EPOCHS)
    m.train_sgd(Xtr_bi_s, ytr_bi)
    r_tr,_=rmse_metrics(ytr_bi, m.predict(Xtr_bi_s)); r_val,_=rmse_metrics(yval_bi, m.predict(Xval_bi_s))
    bi2_results.append((cfg,m,r_tr,r_val))
    print(f"{cfg} -> train RMSE {r_tr:.4f}  val RMSE {r_val:.4f}  final loss {m.loss_history[-1]:.5f}")
best_bi2_cfg, best_bi2_model, _, best_bi2_val = min(bi2_results, key=lambda x: x[3])
print(f"\nBest Bi 2h: {best_bi2_cfg}  val RMSE {best_bi2_val:.4f}")
# overall best bivariate (spec asks to try both 1h and 2h)
if best_bi1_val < best_bi2_val:
    best_bi_cfg, best_bi_model = best_bi1_cfg, best_bi1_model; best_bi_val=best_bi1_val
else:
    best_bi_cfg, best_bi_model = best_bi2_cfg, best_bi2_model; best_bi_val=best_bi2_val
print(f"Overall Best Bivariate: {best_bi_cfg}  val RMSE {best_bi_val:.4f}")

plot_loss(best_uni_model.loss_history, f'Univariate Best {best_uni_cfg} — Avg Squared Error vs Epochs')
plot_loss(best_bi_model.loss_history,  f'Bivariate  Best {best_bi_cfg} — Avg Squared Error vs Epochs')

print("--- Univariate (1h) ---")
for cfg,m,_,_ in uni_results:
    r_tr,p_tr=rmse_metrics(ytr_uni, m.predict(Xtr_uni_s)); r_val,p_val=rmse_metrics(yval_uni, m.predict(Xval_uni_s))
    print(f"{cfg}: train {r_tr:.4f} ({p_tr:.1f}%)  val {r_val:.4f} ({p_val:.1f}%)")
r,p=rmse_metrics(yte_uni, best_uni_model.predict(Xte_uni_s)); print(f"Best {best_uni_cfg} test: {r:.4f} ({p:.1f}%)")

print("\n--- Bivariate 1h ---")
for cfg,m,_,_ in bi1_results:
    r_tr,p_tr=rmse_metrics(ytr_bi, m.predict(Xtr_bi_s)); r_val,p_val=rmse_metrics(yval_bi, m.predict(Xval_bi_s))
    print(f"{cfg}: train {r_tr:.4f} ({p_tr:.1f}%)  val {r_val:.4f} ({p_val:.1f}%)")
print("--- Bivariate 2h ---")
for cfg,m,_,_ in bi2_results:
    r_tr,p_tr=rmse_metrics(ytr_bi, m.predict(Xtr_bi_s)); r_val,p_val=rmse_metrics(yval_bi, m.predict(Xval_bi_s))
    print(f"{cfg}: train {r_tr:.4f} ({p_tr:.1f}%)  val {r_val:.4f} ({p_val:.1f}%)")
r,p=rmse_metrics(yte_bi, best_bi_model.predict(Xte_bi_s)); print(f"Overall best {best_bi_cfg} test: {r:.4f} ({p:.1f}%)")

plot_regression_fit_uni(Xtr_uni_s, ytr_uni, Xval_uni_s, yval_uni, Xte_uni_s, yte_uni, best_uni_model, 'Univariate best')
plot_regression_fit_bi(Xtr_bi_s, ytr_bi, Xval_bi_s, yval_bi, Xte_bi_s, yte_bi, best_bi_model)

scatter_target_vs_pred(ytr_uni, best_uni_model.predict(Xtr_uni_s), 'Uni best — train')
scatter_target_vs_pred(yval_uni, best_uni_model.predict(Xval_uni_s), 'Uni best — val')
scatter_target_vs_pred(yte_uni, best_uni_model.predict(Xte_uni_s), 'Uni best — test')
scatter_target_vs_pred(ytr_bi, best_bi_model.predict(Xtr_bi_s), 'Bi best — train')
scatter_target_vs_pred(yval_bi, best_bi_model.predict(Xval_bi_s), 'Bi best — val')
scatter_target_vs_pred(yte_bi, best_bi_model.predict(Xte_bi_s), 'Bi best — test')

plot_hidden_surfaces_reg(best_uni_model, Xtr_uni_s, ytr_uni, 'Uni best — train')
plot_hidden_surfaces_reg(best_uni_model, Xval_uni_s, yval_uni, 'Uni best — val')
plot_hidden_surfaces_reg(best_uni_model, Xte_uni_s, yte_uni, 'Uni best — test')
plot_hidden_surfaces_reg(best_bi_model, Xtr_bi_s, ytr_bi, 'Bi best — train')
plot_hidden_surfaces_reg(best_bi_model, Xval_bi_s, yval_bi, 'Bi best — val')
plot_hidden_surfaces_reg(best_bi_model, Xte_bi_s, yte_bi, 'Bi best — test')

class LinPerceptron:
    def __init__(self, d, lr=0.01, epochs=800):
        self.w=np.random.randn(d+1)*0.01; self.lr=lr; self.epochs=epochs
    def train(self,X,y):
        n=len(X); Xa=np.c_[np.ones(n),X]
        for _ in range(self.epochs):
            idx=np.random.permutation(n)
            for j in idx:
                err=y[j] - Xa[j]@self.w
                self.w += self.lr * err * Xa[j]
    def predict(self,X):
        return np.c_[np.ones(len(X)),X]@self.w

np.random.seed(0); m=LinPerceptron(Xtr_uni_s.shape[1], lr=0.01, epochs=1200)
m.train(Xtr_uni_s, ytr_uni)
for name,X,y in [('Uni train',Xtr_uni_s,ytr_uni),('Uni val',Xval_uni_s,yval_uni),('Uni test',Xte_uni_s,yte_uni)]:
    r,p=rmse_metrics(y, m.predict(X)); print(f"Single-neuron {name}: RMSE {r:.4f} ({p:.1f}%)")
r,p=rmse_metrics(ytr_uni, best_uni_model.predict(Xtr_uni_s)); print(f"FCNN best Uni train: {r:.4f} ({p:.1f}%)")
r,p=rmse_metrics(yte_uni, best_uni_model.predict(Xte_uni_s)); print(f"FCNN best Uni test:  {r:.4f} ({p:.1f}%)")

np.random.seed(0); m=LinPerceptron(Xtr_bi_s.shape[1], lr=0.01, epochs=1200)
m.train(Xtr_bi_s, ytr_bi)
for name,X,y in [('Bi train',Xtr_bi_s,ytr_bi),('Bi val',Xval_bi_s,yval_bi),('Bi test',Xte_bi_s,yte_bi)]:
    r,p=rmse_metrics(y, m.predict(X)); print(f"Single-neuron {name}: RMSE {r:.4f} ({p:.1f}%)")
r,p=rmse_metrics(ytr_bi, best_bi_model.predict(Xtr_bi_s)); print(f"FCNN best Bi train: {r:.4f} ({p:.1f}%)")
r,p=rmse_metrics(yte_bi, best_bi_model.predict(Xte_bi_s)); print(f"FCNN best Bi test:  {r:.4f} ({p:.1f}%)")

