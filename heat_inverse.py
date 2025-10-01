import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(__file__), ".matplotlib"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import time
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


def solve_tridiag(main, off, rhs):
    """Thomas algorithm using JAX primitives for tri-diagonal systems."""
    main = jnp.asarray(main)
    rhs = jnp.asarray(rhs)
    off = jnp.asarray(off)
    n = main.shape[0]

    if n == 1:
        return rhs / main

    off_pad = jnp.concatenate([off, jnp.zeros((1,), dtype=off.dtype)])

    denom0 = main[0]
    d0 = rhs[0] / denom0
    c0 = off_pad[0] / denom0

    def forward(carry, i):
        c_prev, d_prev = carry
        idx = i + 1
        denom = main[idx] - off[idx - 1] * c_prev
        d_curr = (rhs[idx] - off[idx - 1] * d_prev) / denom
        c_curr = off_pad[idx] / denom
        return (c_curr, d_curr), (c_curr, d_curr)

    (_, _), (c_tail, d_tail) = jax.lax.scan(forward, (c0, d0), jnp.arange(n - 1))
    c_seq = jnp.concatenate([jnp.array([c0]), c_tail])
    d_seq = jnp.concatenate([jnp.array([d0]), d_tail])

    x_last = d_seq[-1]

    def backward(carry, inputs):
        c_i, d_i = inputs
        x_i = d_i - c_i * carry
        return x_i, x_i

    rev_inputs = (jnp.flip(c_seq[:-1]), jnp.flip(d_seq[:-1]))
    _, rev_sol = jax.lax.scan(backward, x_last, rev_inputs)
    sol = jnp.concatenate([jnp.flip(rev_sol), jnp.array([x_last])])
    return sol


def make_stepper(nx, dx, dt, kappa):
    alpha = kappa * dt / (dx * dx)
    n_inner = nx - 2
    main = jnp.full((n_inner,), 1.0 + 2.0 * alpha)
    off = jnp.full((n_inner - 1,), -alpha)

    @jax.jit
    def step(u, s):
        rhs = u + dt * s
        return solve_tridiag(main, off, rhs)

    return step


def rollout(u0, s, step_fn, nt):
    def body(u, _):
        u_next = step_fn(u, s)
        return u_next, u_next

    final_u, traj = jax.lax.scan(body, u0, None, length=nt)
    return final_u, traj


def loss_finaltime(s, u0, y_final, step_fn, nt, lam):
    uT, _ = rollout(u0, s, step_fn, nt)
    misfit = uT - y_final
    data_term = 0.5 * jnp.sum(misfit ** 2)
    reg = 0.5 * lam * jnp.sum(s ** 2)
    return data_term + reg


def loss_sparsepoints(s, u0, idx, y_sparse, step_fn, nt, lam):
    uT, _ = rollout(u0, s, step_fn, nt)
    pred = uT[idx]
    misfit = pred - y_sparse
    data_term = 0.5 * jnp.sum(misfit ** 2)
    reg = 0.5 * lam * jnp.sum(s ** 2)
    return data_term + reg


def loss_multitime(s, u0, y_all, step_fn, nt, t_indices, lam):
    _, traj = rollout(u0, s, step_fn, nt)
    snapshots = jnp.take(traj, t_indices - 1, axis=0)
    misfit = snapshots - y_all
    data_term = 0.5 * jnp.sum(misfit ** 2)
    reg = 0.5 * lam * jnp.sum(s ** 2)
    return data_term + reg


def generate_true_source(nx):
    x = jnp.linspace(0.0, 1.0, nx)
    interior = x[1:-1]
    return jnp.sin(2.0 * jnp.pi * interior)


def generate_observations(step_fn, u0, s_true, nt, noise_std, key):
    uT, traj = rollout(u0, s_true, step_fn, nt)
    noise_key, key = jax.random.split(key)
    noise_final = noise_std * jax.random.normal(noise_key, uT.shape)
    y_final = uT + noise_final

    sensor_idx = np.linspace(0, uT.shape[0] - 1, num=16, dtype=int)
    noise_key, key = jax.random.split(key)
    noise_sparse = noise_std * jax.random.normal(noise_key, (sensor_idx.size,))
    y_sparse = uT[sensor_idx] + noise_sparse

    t_indices = jnp.array([int(0.3 * nt), int(0.6 * nt), nt])
    t_indices = jnp.clip(t_indices, 1, nt)
    noise_key, _ = jax.random.split(key)
    noise_multi = noise_std * jax.random.normal(noise_key, (t_indices.shape[0], uT.shape[0]))
    snapshots = jnp.take(traj, t_indices - 1, axis=0) + noise_multi

    return {
        "y_final": y_final,
        "sensor_idx": jnp.array(sensor_idx),
        "y_sparse": y_sparse,
        "t_indices": t_indices,
        "y_multi": snapshots,
    }


def run_adam(loss_fn, grad_fn, s_init, steps, lr, beta1=0.9, beta2=0.999, eps=1e-8):
    s = s_init
    m = jnp.zeros_like(s)
    v = jnp.zeros_like(s)
    loss_history = []
    beta1_power = 1.0
    beta2_power = 1.0
    for _ in range(1, steps + 1):
        loss_val = loss_fn(s)
        grad_val = grad_fn(s)
        m = (1.0 - beta1) * grad_val + beta1 * m
        v = (1.0 - beta2) * (grad_val ** 2) + beta2 * v
        beta1_power *= beta1
        beta2_power *= beta2
        m_hat = m / (1.0 - beta1_power)
        v_hat = v / (1.0 - beta2_power)
        s = s - lr * m_hat / (jnp.sqrt(v_hat) + eps)
        loss_history.append(float(loss_val))
    final_loss = float(loss_fn(s))
    return s, np.array(loss_history), final_loss


def plot_loss(path, losses):
    plt.figure()
    plt.semilogy(np.arange(1, losses.size + 1), losses)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_sources(path, x_interior, s_true, s_est, sensors=None):
    x_vals = np.asarray(x_interior)
    s_true_np = np.asarray(s_true)
    s_est_np = np.asarray(s_est)
    plt.figure()
    plt.plot(x_vals, s_true_np, label="true", linewidth=2)
    plt.plot(x_vals, s_est_np, label="estimated", linestyle="--", linewidth=2)
    if sensors is not None:
        sensor_pos = x_vals[np.asarray(sensors)]
        plt.scatter(sensor_pos, s_true_np[np.asarray(sensors)], marker="x", color="black", label="sensors")
    plt.xlabel("x")
    plt.ylabel("s(x)")
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    nx = 129
    nt = 200
    T = 0.1
    kappa = 1.0
    dt = T / nt
    dx = 1.0 / (nx - 1)

    u0 = jnp.zeros((nx - 2,))
    s_true = generate_true_source(nx)

    step_fn = make_stepper(nx, dx, dt, kappa)

    key = jax.random.PRNGKey(0)
    obs = generate_observations(step_fn, u0, s_true, nt, noise_std=0.001, key=key)

    x = jnp.linspace(0.0, 1.0, nx)[1:-1]

    experiments = [
        {
            "name": "expA",
            "loss": partial(loss_finaltime, u0=u0, y_final=obs["y_final"], step_fn=step_fn, nt=nt, lam=1e-4),
            "lr": 1.5,
            "steps": 3000,
        },
        {
            "name": "expB",
            "loss": partial(
                loss_sparsepoints,
                u0=u0,
                idx=obs["sensor_idx"],
                y_sparse=obs["y_sparse"],
                step_fn=step_fn,
                nt=nt,
                lam=1e-5,
            ),
            "lr": 0.4,
            "steps": 4500,
        },
        {
            "name": "expC",
            "loss": partial(
                loss_multitime,
                u0=u0,
                y_all=obs["y_multi"],
                step_fn=step_fn,
                nt=nt,
                t_indices=obs["t_indices"],
                lam=1e-4,
            ),
            "lr": 1.2,
            "steps": 3000,
        },
    ]

    for exp in experiments:
        loss_fn = jax.jit(exp["loss"])
        grad_fn = jax.jit(jax.grad(exp["loss"]))
        s0 = jnp.zeros_like(s_true)
        start = time.time()
        s_est, losses, _ = run_adam(loss_fn, grad_fn, s0, exp["steps"], exp["lr"])
        duration = time.time() - start

        rel_error = float(jnp.linalg.norm(s_est - s_true) / jnp.linalg.norm(s_true))
        print(f"{exp['name']}: relative error={rel_error:.4f}, duration={duration:.2f}s")

        plot_loss(f"{exp['name']}_loss.png", losses)
        sensors = None
        if exp["name"] == "expB":
            sensors = obs["sensor_idx"]
        plot_sources(f"{exp['name']}_source.png", x, s_true, s_est, sensors=sensors)


if __name__ == "__main__":
    main()
