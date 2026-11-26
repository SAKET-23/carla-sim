# Equations

The math behind each module, with the file that implements it. Symbols match the code.

- [1. Vehicle model](#1-vehicle-model) · `src/hybrid_av/vehicle/`
- [2. LTV-MPC](#2-ltv-mpc-tracker-and-safety-shield) · `src/hybrid_av/mpc/`
- [3. Reference speed](#3-reference-speed) · `src/hybrid_av/planning/`
- [4. Friction estimator](#4-friction-estimator) · `src/hybrid_av/features/`
- [5. Policy and PPO](#5-policy-and-ppo) · `src/hybrid_av/policy/`, `src/hybrid_av/rl/`
- [6. Tracking and radar fusion](#6-tracking-and-radar-fusion) · `src/hybrid_av/tracking/`
- [7. Weight-space editing](#7-weight-space-editing) · `src/hybrid_av/weightspace/`
- [8. Driving score](#8-driving-score) · `src/hybrid_av/eval/`

---

## 1. Vehicle model

State and input:

```math
\mathbf{x} = [v_x,\ v_y,\ r,\ X,\ Y,\ \psi]^T,\qquad \mathbf{u} = [\delta,\ a_x]^T
```

$v_x, v_y$ are body-frame velocities at the centre of gravity, $r=\dot\psi$ the yaw rate, $(X,Y,\psi)$ the world pose with $\psi$ unwrapped, $\delta$ the front wheel angle and $a_x$ the commanded acceleration.

### Dynamic bicycle (linear tyres)

Slip angles, with $v_x$ floored at $V_\epsilon = 1$ m/s:

```math
\alpha_f = \delta - \frac{v_y + a r}{\max(v_x, V_\epsilon)},\qquad
\alpha_r = -\frac{v_y - b r}{\max(v_x, V_\epsilon)},\qquad
F_{yf} = C_f\alpha_f,\quad F_{yr} = C_r\alpha_r
```

```math
\begin{aligned}
\dot v_x &= a_x + r\,v_y \\
\dot v_y &= \frac{F_{yf}\cos\delta + F_{yr}}{m} - r\,v_x \\
\dot r   &= \frac{a F_{yf}\cos\delta - b F_{yr}}{I_z} \\
\dot X   &= v_x\cos\psi - v_y\sin\psi \\
\dot Y   &= v_x\sin\psi + v_y\cos\psi \\
\dot\psi &= r
\end{aligned}
```

Default parameters: $m = 1500$ kg, $I_z = 2250$ kg m², $a = 1.2$ m, $b = 1.6$ m, $C_f = 19$ kN/rad, $C_r = 20$ kN/rad, $\delta_{max} = 0.70$ rad.

### Low-speed blend

The $v_y$ dynamics have eigenvalue $\approx -(C_f+C_r)/(m v_x)$, which is stiff near standstill. Below 5 m/s the model blends into a kinematic model, where $v_y$ and $r$ relax to their no-slip values with $\tau_k = 0.1$ s:

```math
r_{kin} = \frac{v_x\tan\delta}{L},\quad v_{y,kin} = b\,r_{kin},\quad
\dot v_y = \frac{v_{y,kin}-v_y}{\tau_k},\quad \dot r = \frac{r_{kin}-r}{\tau_k}
```

```math
w = \mathrm{clip}\!\left(\frac{v_x - 3}{5 - 3},\,0,\,1\right),\qquad
f = w\,f_{dyn} + (1-w)\,f_{kin}
```

Its Jacobian includes the blend term:

```math
\frac{\partial f}{\partial \mathbf x} = w\,J_{dyn} + (1-w)\,J_{kin} + (f_{dyn}-f_{kin})\frac{\partial w}{\partial \mathbf x}
```

Integration uses RK4 with $dt = 0.05$ s. The analytic Jacobian is tested against finite differences to 1e-5 relative error.

---

## 2. LTV-MPC (tracker and safety shield)

### Problem

Horizon $N = 20$, $dt = 0.05$ s (1 s):

```math
\min_{\mathbf x,\mathbf u,s}\ \sum_{k=0}^{N-1}\Big(\ell(\mathbf x_k) + \|\mathbf u_k\|^2_R + \|\mathbf u_k-\mathbf u_{k-1}\|^2_{R_\Delta}\Big)
+ P\,\ell(\mathbf x_N) + w_f\sum_k s^f_k + w_o\sum_{j,k} s^o_{j,k}
```

The stage cost uses path-frame errors. They are linear in $X, Y$ because $\psi^{ref}_k$ is known:

```math
e^{lat}_k = -\sin\psi^{ref}_k(X_k - X^{ref}_k) + \cos\psi^{ref}_k(Y_k - Y^{ref}_k),\qquad
e^{lon}_k = \cos\psi^{ref}_k(X_k - X^{ref}_k) + \sin\psi^{ref}_k(Y_k - Y^{ref}_k)
```

```math
\ell(\mathbf x_k) = Q_{v_x}(v_{x,k}-v^{ref}_k)^2 + Q_{v_y}v_{y,k}^2 + Q_r(r_k - v^{ref}_k\kappa_k)^2
+ w_{lat}(e^{lat}_k)^2 + w_{lon}(e^{lon}_k)^2 + w_\psi(\psi_k-\psi^{ref}_k)^2
```

The reference heading is shifted by a multiple of $2\pi$ so that $|\psi^{ref}_0 - \psi_0| \le \pi$. The heading error then stays linear across the whole horizon, with no wrap inside the QP.

### Linearisation with the affine term

Around the previous plan $(\bar{\mathbf x}_k, \bar{\mathbf u}_k)$:

```math
f(\mathbf x,\mathbf u) \approx A_c\mathbf x + B_c\mathbf u + \mathbf c_c,\qquad
\mathbf c_c = f(\bar{\mathbf x},\bar{\mathbf u}) - A_c\bar{\mathbf x} - B_c\bar{\mathbf u}
```

### Exact zero-order-hold discretisation

```math
\exp\!\left(\begin{bmatrix} A_c & B_c \\ 0 & 0 \end{bmatrix} dt\right) = \begin{bmatrix} A_k & B_k \\ 0 & I \end{bmatrix},\qquad
\mathbf c_k = \mathrm{rk4}(\bar{\mathbf x}_k,\bar{\mathbf u}_k) - A_k\bar{\mathbf x}_k - B_k\bar{\mathbf u}_k
```

```math
\mathbf x_{k+1} = A_k\mathbf x_k + B_k\mathbf u_k + \mathbf c_k
```

This makes the prediction exact (to 1e-9) on the nominal trajectory. Forward Euler would be unstable at low speed.

### Friction constraint (slack-softened)

```math
|v_{x,k}\,r_k| \le \bar a_k = 0.85\,\hat\mu\,g
```

The constraint is linearised as $g \approx \bar r_k v_{x,k} + \bar v_{x,k} r_k - \bar v_{x,k}\bar r_k$ and bounded on both sides with slack $s^f_k \ge 0$.

### Obstacle half-planes

For actor $j$ at step $k$, with the outward normal from the actor to the ego:

```math
\mathbf n_{j,k} = \frac{\bar{\mathbf p}_k - \mathbf p_{j,k}}{\|\bar{\mathbf p}_k - \mathbf p_{j,k}\|},\qquad
\mathbf n_{j,k}^T\mathbf p_k + s^o_{j,k} \ge \mathbf n_{j,k}^T\mathbf p_{j,k} + d_{j,k}
```

```math
d_{j,k} = h_j(\mathbf n) + h_{ego}(\mathbf n) + d_{margin},\qquad
h(\mathbf n) = \tfrac{L}{2}|\mathbf n\cdot\mathbf t| + \tfrac{W}{2}|\mathbf n\cdot\mathbf t^\perp|
```

### QP

The decision vector stacks states, inputs and slacks: 126 + 40 + 20 + 80 = **266 variables, 406 rows**. It is solved by OSQP in the form

```math
\min_{\mathbf z}\ \tfrac12\mathbf z^T P\mathbf z + \mathbf q^T\mathbf z\quad \text{s.t.}\quad \mathbf l \le A\mathbf z \le \mathbf u
```

The sparsity pattern is fixed and each solve is warm-started. Slack keeps the QP feasible in every case. If a solve fails anyway, the controller shifts the previous plan; after 3 failures in a row it brakes.

---

## 3. Reference speed

```math
v_{cap}(s) = \min\!\left(\alpha_v\,v_{limit}(s),\ \sqrt{\frac{\min(a_{lat,max},\ 0.85\,\hat\mu g)}{\max(|\kappa(s)|,\,10^{-4})}}\right)
```

The policy sets $\alpha_v$; the curvature $\kappa$ comes from the route.

---

## 4. Friction estimator

Grip can only be observed when the tyres saturate. A parallel linear-tyre model predicts $a_{y,model}$. Let $a_y = v_x r$ be the measured lateral acceleration and $u = \sqrt{a_x^2+a_y^2}/g$ the utilisation:

```math
\rho = \frac{a_y}{a_{y,model}} < 0.7\ \ (\text{3 steps}) \;\Rightarrow\; \mu_{obs} = \frac{u}{0.9},\qquad
\hat\mu \leftarrow \hat\mu + \frac{dt}{\tau}(\mu_{obs}-\hat\mu)
```

The estimate is held at or above $u/0.95$. When the tyres are not saturated, it relaxes back to 1.0 with a 3 s time constant, and it is clipped to $[0.1, 1.2]$.

---

## 5. Policy and PPO

### CompactPolicy (24 → 64 → 64 → 3, 5,955 parameters)

```math
a = \pi_\theta(z) = \tanh\!\big(W_3\,\mathrm{ReLU}(W_2\,\mathrm{ReLU}(W_1 z + b_1) + b_2) + b_3\big)
```

The input $z\in[-1,1]^{24}$ is a **fixed** min-max normalisation of named features (no running statistics), so every weight multiplies a known physical quantity. The outputs are speed scale $\alpha_v$, lateral offset and a brake flag (> 0.5 requests braking).

### Stochastic policy during training

```math
u = m_\theta(z) + \sigma\odot\epsilon,\quad a = \tanh(u),\qquad
\log\pi(a\mid z) = \log\mathcal N(u;\,m_\theta(z),\,\sigma^2) - \sum_i \log(1-\tanh^2 u_i + 10^{-6})
```

### PPO and GAE

```math
L^{CLIP}(\theta) = \mathbb E_t\Big[\min\big(\rho_t\hat A_t,\ \mathrm{clip}(\rho_t,1-\epsilon,1+\epsilon)\hat A_t\big)\Big],\qquad
\rho_t = \frac{\pi_\theta(u_t\mid z_t)}{\pi_{\theta_{old}}(u_t\mid z_t)}
```

```math
\delta_t = r_t + \gamma(1-d_t)V(z_{t+1}) - V(z_t),\qquad \hat A_t = \sum_{l\ge 0}(\gamma\lambda)^l\delta_{t+l}
```

The total loss is $-L^{CLIP} + c_v L^V - c_e\mathcal H$. The base policy in `results/` is behaviour cloning plus DAgger from the MPC expert.

### Reward (per sim step)

```math
r_t = 0.05\,\mathrm{clip}\!\Big(\tfrac{\Delta s}{v_{limit}dt},-1,1.2\Big)
- \tfrac{\max(0,v_x-v_{limit})}{v_{limit}}
- 0.1\tfrac{(e_{lat}-\delta_{y})^2}{1.75^2}
- \max\!\Big(0,\tfrac{|v_x r|}{\hat\mu g}-0.8\Big)
- 2\,\mathbb 1_{slack} - 5\,\mathbb 1_{fallback} - \dots
```

There are also terminal terms: collision −50, off-road −30, blocked −20, timeout −10, completion +20.

---

## 6. Tracking and radar fusion

Each track is a constant-velocity Kalman filter in the map frame, $\mathbf x = [x, y, v_x, v_y]^T$:

```math
F = \begin{bmatrix}1&0&\Delta t&0\\0&1&0&\Delta t\\0&0&1&0\\0&0&0&1\end{bmatrix},\qquad
Q = q\begin{bmatrix}\Delta t^3/3&0&\Delta t^2/2&0\\0&\Delta t^3/3&0&\Delta t^2/2\\ \Delta t^2/2&0&\Delta t&0\\0&\Delta t^2/2&0&\Delta t\end{bmatrix}
```

The LiDAR box update is linear: $H = [I_2\ 0]$. Association uses the Hungarian algorithm on Mahalanobis distance, gated at $\chi^2_{2,0.99} = 9.21$.

**Radar range rate.** First remove the sensor's own motion:

```math
\mathbf v_s = \mathbf v_{ego} + r_{ego}\,\hat{\mathbf z}\times(\mathbf s - \mathbf p_{ego}),\qquad
z_{rr} = \frac{\dot\rho_{meas}}{\cos(\text{alt})} + \mathbf v_s\cdot\mathbf u
```

Then apply an EKF update with $d = \mathbf p - \mathbf s$ and $\rho = \|d\|$:

```math
h(\mathbf x) = \frac{d_x v_x + d_y v_y}{\rho},\qquad
H_{rr} = \left[\tfrac{v_x}{\rho} - \tfrac{d_x(d\cdot v)}{\rho^3},\ \ \tfrac{v_y}{\rho} - \tfrac{d_y(d\cdot v)}{\rho^3},\ \ \tfrac{d_x}{\rho},\ \ \tfrac{d_y}{\rho}\right]
```

The Jacobian is tested against central differences (max relative error < 1e-5).

---

## 7. Weight-space editing

A **zero-gradient edit** changes the weights, $\theta' = \theta + \Delta$, using an explicit list of `(tensor, index, op, value)` entries. The edit comes from inspecting the network, not from an optimiser step.

Each first-layer unit is a half-space detector over named features. The exact contribution of second-layer unit $j$ to output $o$ is:

```math
h_{1,i} = \max(0,\ \mathbf w_i^T z + b_i),\qquad c_{o,j} = W_3[o,j]\,h_{2,j}
```

Acceptance gates:

```math
\text{sparsity} = \frac{|\{i:\theta'_i\ne\theta_i\}|}{5955} \le 3.5\%,\qquad
\text{retention} = \frac{\overline{DS}(\theta')}{\overline{DS}(\theta)} \ge 0.95,\qquad
\text{OOD success} \ge 0.80
```

**Accepted wet-curve edit:** one unit recruited, 5 non-zero values. It requests braking when road roughness is high and speed is above about 6.5 m/s:

```math
h_{1,0} = \mathrm{ReLU}(3 z_{speed} + 3 z_{rough} + 0.9),\qquad h_{2,0} = \mathrm{ReLU}(h_{1,0}),\qquad \text{brake}_{pre} \mathrel{+}= 3\,h_{2,0}
```

The unit's other incoming and outgoing weights are set to zero; together with the 5 values above, the edit touches 93 entries (1.6 % of the network).

---

## 8. Driving score

This is a local implementation of the Leaderboard 2.1 formula ("leaderboard-style"):

```math
DS = 100\cdot\frac{\max\!\big(0,\ RC - d_{offroad}/L_{route}\big)}{1 + \sum_k c_k\, n_k}
```

$RC$ is route completion, and $n_k$ counts infractions of type $k$ with penalty coefficient $c_k$.
