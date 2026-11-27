
import argparse, json, csv, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from common import MATERIALS, derived, load_csv, h_churchill_bernstein
from baseline_lm import fit_boundary

p_ = argparse.ArgumentParser()
p_.add_argument("csv"); p_.add_argument("--mat", default="al")
p_.add_argument("--iters", type=int, default=15000); p_.add_argument("--seed", type=int, default=0)
p_.add_argument("--stride", type=int, default=5, help="usar 1 de cada N muestras de tiempo")
p_.add_argument("--k0", type=float, default=100.)
p_.add_argument("--domain", choices=["full", "legacy"], default="full")
p_.add_argument("--out", default="results")
p_.add_argument("--threads", type=int, default=2)
a = p_.parse_args()
if min(a.k0, a.iters, a.stride, a.threads) <= 0: p_.error("positive settings required")
torch.set_num_threads(a.threads)
start = time.perf_counter()
folder = Path(a.out) / f"{Path(a.csv).stem}_k{a.k0:g}_seed{a.seed}_{a.domain}_it{a.iters}_s{a.stride}"
folder.mkdir(parents=True, exist_ok=False)
history = []
torch.manual_seed(a.seed); np.random.seed(a.seed); torch.set_default_dtype(torch.float32)

p = MATERIALS[a.mat]; X = np.array(p["X"]); A, s = derived(p)
rho, c, L = p["rho"], p["c"], p["L"]
t, TT = load_csv(a.csv); alpha, tau = fit_boundary(t, TT, X)
TC, TH = 300.0, 40.0                      # escalas de tiempo [s] y temperatura [°C]

tmax = float(t.max()) / TC if a.domain == "full" else 1.0
# datos (x*, t*) -> theta*
idx = np.arange(0, len(t), a.stride)
xd = np.repeat(X/L, len(idx)); td = np.tile(t[idx]/TC, 6); thd = (TT[idx, :].T.reshape(-1))/TH
xd, td, thd = [torch.tensor(v, dtype=torch.float32).view(-1, 1) for v in (xd, td, thd)]

net = nn.Sequential(nn.Linear(2, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(),
                    nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))
# parámetros físicos (en escala log). Inicio genérico: NO usa el resultado del baseline
log_k = nn.Parameter(torch.log(torch.tensor(a.k0)))
log_h = nn.Parameter(torch.log(torch.tensor(float(h_churchill_bernstein(p["v"], p["D"])))))
log_P = nn.Parameter(torch.log(torch.tensor(1.0)))
opt = torch.optim.Adam([{"params": net.parameters(), "lr": 1e-3},
                        {"params": [log_k, log_h, log_P], "lr": 1e-2}])
sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(a.iters//3, 1), gamma=0.8)

def theta(x, t_): return net(torch.cat([x, t_], 1))
def grads(x, t_):
    x.requires_grad_(True); t_.requires_grad_(True)
    u = theta(x, t_)
    ux = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    ut = torch.autograd.grad(u.sum(), t_, create_graph=True)[0]
    uxx = torch.autograd.grad(ux.sum(), x, create_graph=True)[0]
    return u, ux, ut, uxx

for it in range(1, a.iters+1):
    opt.zero_grad()
    k, h, P = log_k.exp(), log_h.exp(), log_P.exp()
    kappa, nu = k/(rho*c), h*s/(rho*A*c)
    # (1) datos
    L_data = ((theta(xd, td) - thd)**2).mean()
    # (2) residuo de la EDP en puntos de colocación
    xc, tc = torch.rand(2000, 1), torch.rand(2000, 1) * tmax
    u, ux, ut, uxx = grads(xc, tc)
    L_pde = ((ut - (kappa*TC/L**2)*uxx + nu*TC*u)**2).mean()
    # (3) frontera x=0 (flujo) y x=L (adiabática), (4) condición inicial
    tb = torch.rand(300, 1) * tmax; x0 = torch.zeros(300, 1); xL = torch.ones(300, 1)
    _, ux0, _, _ = grads(x0, tb.clone()); _, uxL, _, _ = grads(xL, tb.clone())
    F = P*(1 - torch.exp(-alpha*(tb*TC + tau)))
    L_bc0 = (((k*A*TH/L*ux0 + F))**2).mean()
    L_bcL = (uxL**2).mean()
    xi = torch.rand(300, 1); L_ic = (theta(xi, torch.zeros(300, 1))**2).mean()
    loss = 10*L_data + L_pde + L_bc0 + L_bcL + 10*L_ic
    loss.backward(); opt.step(); sched.step()
    if not torch.isfinite(loss): raise RuntimeError("Nonfinite loss")
    if it % 100 == 0 or it in (1, a.iters):
        history.append(dict(iteration=it, loss=loss.item(), data=L_data.item(), pde=L_pde.item(), bc0=L_bc0.item(), bcL=L_bcL.item(), ic=L_ic.item(), k=k.item(), h=h.item(), Pss=P.item()))
    if it % 1000 == 0 or it == 1:
        print(f"it {it:6d} loss {loss.item():.3e} data {L_data.item():.2e} pde {L_pde.item():.2e} "
              f"| k={k.item():.1f} h={h.item():.1f} P={P.item():.3f}")

with torch.no_grad():
    allx = torch.tensor(np.repeat(X/L, len(t)), dtype=torch.float32).view(-1, 1)
    allt = torch.tensor(np.tile(t/TC, 6), dtype=torch.float32).view(-1, 1)
    pred = theta(allx, allt).numpy().ravel()*TH
rmse = float(np.sqrt(np.mean((pred - TT.T.reshape(-1))**2)))
res = dict(archivo=a.csv, seed=a.seed, iters=a.iters, k=float(log_k.detach().exp()), h=float(log_h.detach().exp()),
           Pss=float(log_P.detach().exp()), rmse_C=rmse)
res.update(k0=a.k0, domain=a.domain, stride=a.stride, gamma=0.8, step_size=max(a.iters//3,1), alpha=float(alpha), tau=float(tau), seconds=time.perf_counter()-start, nrmse_range=rmse/float(np.ptp(TT)), t_max=float(t.max()), torch_version=str(torch.__version__), threads=a.threads)
with (folder / "summary.json").open("w") as f: json.dump(res, f, indent=2, allow_nan=False)
with (folder / "history.csv").open("w", newline="") as f:
    w=csv.DictWriter(f, fieldnames=list(history[0])); w.writeheader(); w.writerows(history)
np.savetxt(folder / "predictions.csv", np.column_stack([np.tile(t,6), np.repeat(X,len(t)), TT.T.reshape(-1), pred]), delimiter=",", header="time_s,x_m,observed_C,predicted_C", comments="")
print(json.dumps(res, indent=2))
