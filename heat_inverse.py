import time
import jax, jax.numpy as jnp
from jax import jit, grad, lax
import numpy as np
import matplotlib.pyplot as plt

def build_laplacian_1d(Nx):
    n = Nx - 2
    main = -2.0 * jnp.ones((n,))
    off  =  1.0 * jnp.ones((n-1,))
    return main, off

def solve_tridiag(main, off, rhs):
    n = rhs.shape[0]
    def fwd(carry, i):
        c_star, d_star = carry
        denom = main[i] - (off[i-1]**2) * c_star[i-1]
        c_star = c_star.at[i].set(off[i-1]/denom if i < n-1 else 0.0)
        d_star = d_star.at[i].set((rhs[i]-off[i-1]*d_star[i-1])/denom)
        return (c_star, d_star), None
    c_star = jnp.zeros((n,)); d_star = jnp.zeros((n,))
    denom0 = main[0]
    c_star = c_star.at[0].set(off[0]/denom0 if n>1 else 0.0)
    d_star = d_star.at[0].set(rhs[0]/denom0)
    (c_star, d_star), _ = lax.scan(fwd, (c_star, d_star), jnp.arange(1, n))
    def bwd(u_next, i): return d_star[i] - c_star[i]*u_next, None
    u_last = d_star[-1]
    u_rev, _ = lax.scan(bwd, u_last, jnp.arange(n-2, -1, -1))
    u = jnp.concatenate([u_rev[1:], jnp.array([u_last])])
    return u

def make_stepper(Nx, dx, dt, kappa):
    main_L, off_L = build_laplacian_1d(Nx)
    alpha = -kappa*dt/(dx*dx)
    main_A = 1.0 - alpha*main_L
    off_A  = -alpha*off_L
    @jit
    def step(u, s):
        rhs = u + dt*s
        return solve_tridiag(main_A, off_A, rhs)
    return step

def rollout(u0, s, step, Nt):
    def body(u, _):
        u_next = step(u, s)
        return u_next, u_next
    uT, traj = lax.scan(body, u0, None, length=Nt)
    return uT, traj

def synth_setup(Nx=129, Nt=200, T=0.1, kappa=1.0, key=0):
    key = jax.random.PRNGKey(key)
    dx = 1.0/(Nx-1); dt = T/Nt
    xs = jnp.linspace(0,1,Nx)[1:-1]
    s_true = jnp.sin(2*jnp.pi*xs)
    u0 = jnp.zeros_like(s_true)
    step = make_stepper(Nx, dx, dt, kappa)
    uT, _ = rollout(u0, s_true, step, Nt)
    noise = 0.01*jax.random.normal(key, uT.shape)
    y_final = uT + noise
    return xs, s_true, u0, y_final, step, (dx,dt,kappa,Nt,Nx)

def loss_finaltime(s, u0, y_final, step, Nt, lam=1e-2):
    uT, _ = rollout(u0, s, step, Nt)
    mis = 0.5*jnp.mean((uT - y_final)**2)
    reg = 0.5*lam*jnp.mean(s**2)
    return mis + reg

loss_finaltime_grad = jit(grad(loss_finaltime))

@jit
def gd_step(s, g, lr=0.5): return s - lr*g

def run_exp_A():
    print("\n[Exp-A] Final-time full-field")
    xs, s_true, u0, y_final, step, params = synth_setup()
    _, _, _, Nt, _ = params
    s = jnp.zeros_like(s_true)
    _ = loss_finaltime(s, u0, y_final, step, Nt)   # JIT warmup
    losses=[]; t0=time.perf_counter()
    for it in range(200):
        g = loss_finaltime_grad(s, u0, y_final, step, Nt)
        s = gd_step(s, g, lr=0.5)
        if it%5==0:
            losses.append(float(loss_finaltime(s, u0, y_final, step, Nt)))
    t1=time.perf_counter()
    rel=float(jnp.linalg.norm(s-s_true)/jnp.linalg.norm(s_true))
    print(f"rel_err(s)={rel:.3e} | time={(t1-t0):.2f}s | iters=200")
    import numpy as np
    plt.figure(); plt.semilogy(np.arange(len(losses))*5, losses)
    plt.xlabel("iter"); plt.ylabel("loss"); plt.title("Exp-A loss"); plt.savefig("expA_loss.png", dpi=160)
    plt.figure(); plt.plot(np.array(xs), np.array(s_true), label="s_true")
    plt.plot(np.array(xs), np.array(s), '--', label="s_est"); plt.legend(); plt.title("Exp-A source")
    plt.savefig("expA_source.png", dpi=160)

def run_exp_B():
    print("\n[Exp-B] Sparse sensors")
    xs, s_true, u0, y_final, step, params = synth_setup()
    _, _, _, Nt, _ = params
    s = jnp.zeros_like(s_true)
    M=12; idx=np.linspace(0, s_true.shape[0]-1, M, dtype=int)
    y_sparse=np.array(y_final)[idx]
    def loss_sparsepoints(s, u0, idx, y, step, Nt, lam=5e-2):
        uT,_=rollout(u0,s,step,Nt)
        return 0.5*jnp.mean((uT[idx]-y)**2)+0.5*lam*jnp.mean(s**2)
    grad_sparse = jit(grad(loss_sparsepoints))
    _ = loss_sparsepoints(s, u0, idx, y_sparse, step, Nt)
    losses=[]; t0=time.perf_counter()
    for it in range(300):
        g = grad_sparse(s, u0, idx, y_sparse, step, Nt)
        s = gd_step(s, g, lr=0.3)
        if it%10==0:
            losses.append(float(loss_sparsepoints(s, u0, idx, y_sparse, step, Nt)))
    t1=time.perf_counter()
    rel=float(jnp.linalg.norm(s-s_true)/jnp.linalg.norm(s_true))
    print(f"rel_err(s)={rel:.3e} | time={(t1-t0):.2f}s | iters=300")
    plt.figure(); plt.semilogy(np.arange(len(losses))*10, losses)
    plt.xlabel("iter"); plt.ylabel("loss"); plt.title("Exp-B loss"); plt.savefig("expB_loss.png", dpi=160)
    plt.figure(); plt.plot(np.array(xs), np.array(s_true), label="s_true")
    plt.plot(np.array(xs), np.array(s), '--', label="s_est")
    plt.scatter(np.array(xs)[idx], y_sparse*0, s=18, marker='x', label='sensors'); plt.legend(); plt.title("Exp-B source")
    plt.savefig("expB_source.png", dpi=160)

def run_exp_C():
    print("\n[Exp-C] Multi-time")
    xs, s_true, u0, y_final, step, params = synth_setup()
    _, _, _, Nt, _ = params
    s = jnp.zeros_like(s_true)
    t_indices = np.array([int(Nt*0.3), int(Nt*0.6), Nt-1], dtype=int)
    _, traj = rollout(u0, s_true, step, Nt)
    y_list = [np.array(traj[i]) + 0.01*np.random.randn(traj[i].shape[0]) for i in t_indices]
    def loss_multitime(s, u0, y_list, step, t_indices, lam=1e-2):
        uT,traj=rollout(u0,s,step,t_indices[-1]+1)
        mis=0.0
        for idx,y in zip(t_indices,y_list): mis = mis + 0.5*jnp.mean((traj[idx]-y)**2)
        return mis + 0.5*lam*jnp.mean(s**2)
    grad_multi = jit(grad(loss_multitime))
    _ = loss_multitime(s, u0, y_list, step, t_indices)
    losses=[]; t0=time.perf_counter()
    for it in range(200):
        g = grad_multi(s, u0, y_list, step, t_indices)
        s = gd_step(s, g, lr=0.4)
        if it%5==0:
            losses.append(float(loss_multitime(s, u0, y_list, step, t_indices)))
    t1=time.perf_counter()
    rel=float(jnp.linalg.norm(s-s_true)/jnp.linalg.norm(s_true))
    print(f"rel_err(s)={rel:.3e} | time={(t1-t0):.2f}s | iters=200")
    plt.figure(); plt.semilogy(np.arange(len(losses))*5, losses)
    plt.xlabel("iter"); plt.ylabel("loss"); plt.title("Exp-C loss"); plt.savefig("expC_loss.png", dpi=160)
    plt.figure(); plt.plot(np.array(xs), np.array(s_true), label="s_true")
    plt.plot(np.array(xs), np.array(s), '--', label="s_est"); plt.legend(); plt.title("Exp-C source")
    plt.savefig("expC_source.png", dpi=160)

def main():
    print("JAX no-NN baseline: 1D heat source inversion")
    run_exp_A(); run_exp_B(); run_exp_C()
    print("Saved plots: expA_loss.png, expA_source.png, expB_loss.png, expB_source.png, expC_loss.png, expC_source.png")

if __name__ == "__main__":
    main()
