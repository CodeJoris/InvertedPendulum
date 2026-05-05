# PID Optimization Implementation

## Overview

The inverted pendulum controller now includes an automated PID parameter optimization system using `scipy.optimize.minimize()`. This allows you to discover optimal P, I, D gains without manual slider adjustment.

## How It Works

### Optimization Algorithm
- **Method**: L-BFGS-B (Limited-memory Broyden-Fletcher-Goldfarb-Shanno)
- **Search Space**: 
  - Kp (Proportional): [-200, 200]
  - Ki (Integral): [-1.0, 1.0]  
  - Kd (Derivative): [-50, 50]
- **Objective Function**: Minimizes `integral(theta² + 0.001 * control_effort²) dt`
  - Balances pendulum stabilization (minimize theta deviation)
  - Penalizes excessive control effort (smooth, efficient control)

### Key Functions

#### `simulate_pid(kp, ki, kd, duration=10.0, dt=5/60)`
Runs a single simulation with given PID parameters and returns the cumulative loss.
- Creates a fresh, isolated physics simulation
- Runs for specified duration at fixed time step
- Returns scalar loss value for the optimizer to minimize
- Called 10-20+ times per optimization run

#### `optimize_pid(initial_kp=1.0, initial_ki=0.0, initial_kd=0.0, duration=10.0)`
Orchestrates the optimization search.
- Takes initial parameter guesses
- Runs scipy.optimize.minimize with L-BFGS-B algorithm
- Returns optimal (Kp, Ki, Kd) tuple
- Prints detailed progress and final results to console

## Usage

### Option 1: Quick Integration (Recommended First)
Edit `main.py` and set `ENABLE_OPTIMIZATION = True`:

```python
ENABLE_OPTIMIZATION = True  # Set to True to run PID optimization before interactive mode
```

Run the script:
```bash
python main.py
```

The optimization will run first (1-2 minutes), then automatically load the optimized parameters into the slider UI. You can then interactively adjust from there.

### Option 2: Standalone Test
Run the test script to see optimization in action:
```bash
python test_optimization.py
```

This runs a quick ~1 second per evaluation test and shows you the optimization process.

### Option 3: Custom Parameters in Code
Call directly in your scripts:
```python
from main import optimize_pid, simulate_pid

# Find optimal parameters (takes 1-2 minutes)
kp, ki, kd = optimize_pid(initial_kp=50.0, initial_ki=0.0, initial_kd=10.0, duration=10.0)

# Or evaluate specific parameters
loss = simulate_pid(kp=100, ki=0.5, kd=20, duration=5.0)
```

## Performance Characteristics

| Setting | Time | Quality | Use Case |
|---------|------|---------|----------|
| `duration=1.0` | ~30 sec | Fair | Quick tuning, initial exploration |
| `duration=5.0` | ~2-3 min | Good | Production tuning (recommended) |
| `duration=10.0` | ~5-10 min | Excellent | Final validation |

**Timing Note**: Optimization is CPU-bound. Total time = (evaluation_duration × number_of_evaluations).
L-BFGS-B typically needs 15-25 evaluations to converge.

## Example Output

```
Starting PID optimization...
Initial guess: Kp=50.000, Ki=0.000, Kd=10.000
  Evaluating: Kp=50.000, Ki=0.000, Kd=10.000  Loss=234.5621
  Evaluating: Kp=52.341, Ki=0.015, Kd=11.234  Loss=198.3421
  Evaluating: Kp=48.921, Ki=-0.008, Kd=9.876  Loss=201.2340
  ...
Optimization complete!
Optimal parameters: Kp=105.234, Ki=0.342, Kd=28.123
Final loss: 42.1234

Sliders updated with optimized parameters. Ready to run simulation.
```

## Architecture

### Integration with Existing Code
- **Non-invasive**: Original interactive slider mode unchanged
- **Feature flagged**: Toggle with `ENABLE_OPTIMIZATION` boolean
- **Modular**: `simulate_pid()` and `optimize_pid()` are standalone functions
- **Direct slider loading**: Optimized values automatically loaded into UI sliders post-optimization

### Physics Simulation Isolation
Each optimization evaluation:
1. Creates fresh pymunk.Space with independent physics state
2. Initializes bodies and constraints
3. Runs simulation with given PID parameters
4. Computes cumulative loss
5. Cleans up (garbage collected)

This ensures no state leakage between evaluations.

## Advanced Customization

### Change Optimization Bounds
Edit the bounds in `optimize_pid()`:
```python
bounds = [(-200.0, 200.0), (-1.0, 1.0), (-50.0, 50.0)]  # Kp, Ki, Kd
```

### Adjust Loss Function
Modify the weight of control effort in `simulate_pid()`:
```python
cumulative_loss += (theta**2 + 0.001 * (force / 6000.0)**2) * dt
                   # Increase 0.001 to penalize control effort more heavily
```

### Change Optimizer Algorithm
Replace `method='L-BFGS-B'` in `optimize_pid()` with:
- `'Nelder-Mead'`: Non-gradient-based (slower but robust)
- `'COBYLA'`: Constrained optimization by linear approximation
- `'trust-constr'`: Trust-region with general constraints

## Validation

After optimization completes:
1. Note the optimal parameter values displayed
2. Run the interactive mode with `ENABLE_OPTIMIZATION = False`
3. Manually set sliders to the reported optimal values
4. Observe system behavior and fine-tune if needed

## Troubleshooting

**"scipy not found"**:
```bash
pip install scipy
```

**Optimization is slow**:
- Reduce `duration` parameter (faster evaluation, less accurate)
- Check CPU load; optimize is single-threaded and CPU-intensive

**Results seem poor**:
- Verify initial guess is reasonable: `initial_kp=50` for example
- Increase `duration` for more accurate loss calculations
- Check loss function weights in `simulate_pid()`

## Future Enhancements

- Parallel evaluation using `scipy.optimize.differential_evolution()` 
- Multi-objective optimization (Pareto frontier of stabilization vs. energy)
- Adaptive duration: start fast, increase during convergence
- Machine learning integration for warm-start initial guesses
