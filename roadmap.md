# Inverted Pendulum Control Roadmap: Theory to Hardware

This document outlines the complete step-by-step pipeline for balancing an inverted pendulum on a cart, moving from theoretical Lagrangian mechanics to real-time hardware execution.

---

## Phase 1: The Physics Derivation (The SymPy Phase)
*This phase extracts the fundamental equations of motion. It is executed purely in symbolic math.*

1. **The Lagrangian:** 
   - Define Kinetic Energy ($T$) and Potential Energy ($V$) using the physical lengths and masses of the 3D-printed parts. 
   - Calculate $L = T - V$.
2. **Euler-Lagrange Equations:** 
   - Take the derivatives of $L$ to generate the two highly non-linear, second-order differential equations for cart acceleration ($\ddot{x}$) and pendulum angular acceleration ($\ddot{\theta}$).
3. **The Target Vector ($f$):** 
   - Define the state vector $\mathbf{x} = [x, \dot{x}, \theta, \dot{\theta}]^T$. 
   - Stack the equations to create the symbolic vector $f(\mathbf{x}, \mathbf{u})$ representing the first derivatives: $[\dot{x}, \ddot{x}, \dot{\theta}, \ddot{\theta}]^T$.

---

## Phase 2: The Linearization (The Math Phase)
*This phase translates the complex, non-linear physics into clean linear algebra suitable for control theory.*

1. **The Jacobians:** 
   - Use SymPy to take the Jacobian of $f$ with respect to the state vector $\mathbf{x}$. This generates the symbolic System Matrix ($A$).
   - Take the Jacobian of $f$ with respect to the input force $u$. This generates the symbolic Input Matrix ($B$).
2. **The Equilibrium Point:** 
   - Substitute the target balance state ($\theta = 0, \dot{\theta} = 0, \dot{x} = 0$) into the symbolic $A$ and $B$ matrices to lock them into the linear region.
3. **The Handoff:** 
   - Use `sp.lambdify` to compile these linearized, symbolic matrices into highly optimized, lightning-fast NumPy functions.

---

## Phase 3: The Control Design (The SciPy Phase)
*This phase calculates how hard the stepper motors need to pull to maintain balance.*

1. **Hardware Substitution:** 
   - Feed the exact physical masses ($M$, $m$) and lengths ($l$) of the physical rig into the compiled NumPy functions to generate pure numerical $A$ and $B$ matrices.
2. **LQR (Linear Quadratic Regulator):** 
   - Utilize a control library (e.g., `control.lqr(A, B, Q, R)`). 
   - Pass the numerical matrices into the solver to automatically calculate the optimal Control Matrix ($K$). 
   - *Note: This $K$ matrix contains the four gain numbers that shift the eigenvalues of $(A - BK)$ into the stable, negative territory on the complex plane.*

---

## Phase 4: The Hardware Execution (The Real-Time Phase)
*This is the real-time loop running on the physical hardware, utilizing raw sensor data.*

1. **The Swing-Up (Energy Control):** 
   - Because the linear $K$ matrix only works near the top upright position ($\theta \approx 0$), the system starts in an energy-pumping mode. 
   - The cart rocks back and forth, reading the real angle data until the pendulum swings up to within the linear threshold (approx. $\pm 15^\circ$ from vertical).
2. **The Catch (State-Feedback Control):** 
   - The exact millisecond the encoder reads an angle within the threshold, the software switches control modes.
3. **The Real-Time Loop:**
   - Read the actual, physical encoder data to populate the current state vector $\mathbf{x}$.
   - Calculate the required motor force using a fast NumPy dot product: $u = -K\mathbf{x}$.
   - Send the command $u$ to the stepper motor drivers.
   - Repeat continuously.