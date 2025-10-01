没问题。我给你两份可直接用的文档：

1. 一个“**一键搞定的详细 Prompt**”（你贴给 Codex / 代码代理就行）。
2. 一个“**agent.md**”项目执行说明（放在仓库根目录，规范任务拆解、目录结构、质量标准、兼容性与常见坑避雷）。

——你把这两份直接交给 Codex，它就能在你的仓库里把无NN的 JAX demo 做完、能跑、能出图。

---

# ✅ 详细 Prompt （直接贴给 Codex）

**角色**：你是资深数值计算与可微编程工程师，熟悉 PDE 约束优化、JAX、离散伴随（通过自动微分实现）。
**目标**：在现有仓库 `PDE-` 中实现一个**不含神经网络**的、可复现的 **1D 热方程源项反演** JAX 基线，并生成可展示图表与 README。要求在 **CPU** 环境下**稳定运行**（需要规避 Apple Metal 后端的崩溃），且 **JIT/grad** 兼容、无 tracer 条件分支错误。
**时限**：一次性提交可运行代码与图像产物。

---

## 功能要求

1. **前向模型（1D 热方程）**

   * PDE：( u_t = \kappa u_{xx} + s(x) ) on (x\in(0,1))，Dirichlet (u(0,t)=u(1,t)=0)，初值 (u(x,0)=0)。
   * 空间离散：均匀网格，内点自由度（去掉两端），二阶中心差分拉普拉斯。
   * 时间离散：**隐式Euler**。每一步解一个三对角线性系统：
     [
     (I - \Delta t, \kappa L),u^{n+1} = u^n + \Delta t, s
     ]
   * 时间推进用 `jax.lax.scan`。
   * 解三对角系统用 **Thomas 算法**，但必须 **JAX/scan 友好**：

     * 不允许在被 trace 的循环索引上使用 Python `if`；
     * 允许使用 `lax.fori_loop` / `lax.scan`；
     * 不要产生 `TracerBoolConversionError`。

2. **反演任务（无 NN）**

   * 待估计量：**空间源项** ( s \in \mathbb{R}^{N_x-2} )。
   * 观测三种设置（对应三组实验）：

     * **Exp-A**：末时刻全场 ( y \approx u(\cdot, T) )；
     * **Exp-B**：末时刻稀疏传感器（选择 8–16 个空间点）；
     * **Exp-C**：多时刻全场（例如 (t=0.3T, 0.6T, T)）。
   * 合成数据：用一个可控真值 ( s_\text{true}(x)=\sin(2\pi x) )（可配置），用前向模型生成观测并加小噪声。
   * 损失：数据拟合 + Tikhonov 正则 ( \frac\lambda2|s|^2 )。
   * 优化：用纯 JAX 的梯度下降或 Adam（如有 `jaxopt` 可选 L-BFGS 版本，但不是必须）。
   * 输出：每个实验的 **loss 曲线（半对数）** 和 **源项重建对比图**；控制台打印相对误差与总耗时。

3. **稳定性与兼容性要求**

   * **强制 CPU 后端运行**（避免 Metal 崩溃）：

     * 在代码顶层加入 `os.environ["JAX_PLATFORMS"]="cpu"`（在 `import jax` 之前）；
     * 或在 README 说明 `export JAX_PLATFORMS=cpu`。
   * **不得**在被 `jit`/`scan` 跟踪的循环索引上使用 Python `if`，例如 `if i < n-1`；必须用 **无分支**公式或 `jnp.where`。
   * 代码需可在 **Python 3.10–3.12**、`jax`/`jaxlib` 的 CPU 版本下运行（无需 GPU）。
   * 运行结束应在仓库根目录生成 6 张图：

     * `expA_loss.png`, `expA_source.png`
     * `expB_loss.png`, `expB_source.png`
     * `expC_loss.png`, `expC_source.png`

---

## 交付物

* `heat_inverse.py`：主脚本，可一键运行并产生 6 张图。
* `requirements.txt`：包含 `jax`, `jaxlib`, `numpy`, `matplotlib`（可加 `jaxopt` 为可选依赖）。
* `README.md`：包含问题描述、方程与离散、运行方法、默认参数、三组实验说明、生成图片示例与指标。
* `.gitignore`：忽略 `.venv/`, `__pycache__/`, `*.ipynb_checkpoints`, `*.png`。
* （可选）`solver.py`, `losses.py`, `experiments.py` 拆分，但主脚本必须可独立运行。

---

## 目录结构（可选）

```
PDE-/
  heat_inverse.py
  requirements.txt
  README.md
  .gitignore
```

---

## 具体实现细节与接口

1. **三对角解法器 `solve_tridiag(main, off, rhs)`**

   * 输入：主对角 `main` (n,), 次对角 `off` (n-1,), 右端 `rhs` (n,)。
   * Thomas 前消元（forward elimination）+ 反代（back substitution）。
   * 用 `lax.scan` / `lax.fori_loop` 实现**无 Python 分支**版本。
   * 对 `n==1` 情况安全（`scan` 0 次迭代、回代跳过即可）。

2. **时间步进器 `make_stepper(Nx, dx, dt, kappa)`**

   * 预装配 `A = I - α L` 的三对角对角线，`α = kappa*dt/dx^2`（注意号）。
   * 返回一个 `@jit` 的 `step(u, s)`，内部仅调用 `solve_tridiag` 解线性系统。

3. **时间滚动 `rollout(u0, s, step, Nt)`**

   * 用 `lax.scan`：携带 `u`，忽略输入序列（长度 `Nt`）。
   * 返回末时刻 `uT` 和整个轨迹 `traj`（shape `(Nt, n)`）。

4. **损失函数**

   * `loss_finaltime(s, u0, y_final, step, Nt, lam=1e-2)`
   * `loss_sparsepoints(s, u0, idx, y_sparse, step, Nt, lam=5e-2)`
   * `loss_multitime(s, u0, y_list, step, t_indices, lam=1e-2)`
   * 为每个损失定义 `jit(grad(loss_*))` 的梯度函数。

5. **实验配置**

   * 默认：`Nx=129, Nt=200, T=0.1, kappa=1.0`，`s_true=sin(2πx)`，噪声标准差 0.01。
   * 学习率：A=0.5, B=0.3, C=0.4；迭代步数：A=200, B=300, C=200（可调整）。
   * 打印：相对误差 `||s_est-s_true||/||s_true||`、总运行时间（不强求精准、只要可比）。
   * 画图：loss（半对数）、`s_true` vs `s_est`；B 实验在源图上标出传感器位置（散点 `x`）。

6. **避坑指令（必须遵守）**

   * **不要**在 `scan` 循环变量上使用 Python `if`；**不要** `if i < n-1`；
   * 如确有需要，使用 `jnp.where(condition, a, b)`，但优先用“无分支公式”。
   * 在文件最顶部、`import jax` 之前添加：

     ```python
     import os
     os.environ.setdefault("JAX_PLATFORMS", "cpu")
     ```

---

## 验收标准（自动与人工）

* 能在 **CPU** 下**无错误**运行：`python heat_inverse.py`。
* 控制台输出每个实验的相对误差与耗时；
* 根目录生成 6 张图；
* 相对误差（A 实验）通常 < 0.15（不同随机噪声会有差异，可通过 `lambda` 调参确保合理）；
* 无 `TracerBoolConversionError` / 无 Metal 相关崩溃；
* 代码结构清晰，函数内联注释简洁准确。

---

## 提交与信息记录

* 创建或直接提交到分支 `demo/jax-heat-inverse`；
* 提交信息：

  ```
  JAX no-NN baseline: 1D heat source inversion (Exp A/B/C) + plots
  - CPU-only via JAX_PLATFORMS to avoid Metal crash
  - scan/fori_loop-friendly tridiagonal Thomas solver (no tracer if)
  - loss/grad JIT-ready; three experiments implemented
  - README + requirements + generated figures
  ```
* 成功后在 README 内嵌 1–2 张小图（或提供相对路径）。

---
