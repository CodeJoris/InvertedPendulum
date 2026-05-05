import pymunk
import pymunk.pygame_util
import pygame
from collections import deque
import numpy as np

import matplotlib.pyplot as plt
from scipy.optimize import minimize

GRAY = (220, 220, 220)
CONTROL_BG = (35, 35, 35)
TRACK_BG = (70, 70, 70)
TRACK_FG = (120, 180, 255)
KNOB = (240, 240, 240)
WHITE = (245, 245, 245)
ENABLE_LIVE_PLOTS = False
ENABLE_OPTIMIZATION = True  # Set to True to run PID optimization before interactive mode
space = pymunk.Space()
space.gravity = 0, 9
b0 = space.static_body
WIDTH, HEIGHT = 1000, 600
CONTROL_HEIGHT = 180
l = 100  # length of the pendulum
m = 1  # mass of the mass at the end of pendulum
M = 10  # mass of the pivot (cart) - reduced from 1000 for better responsiveness
FORCE_LIMIT = 10000  # increased from 6000 for stronger control
WALL_MARGIN = 50  # distance from boundary to trigger wall correction (pixels)


class Slider:
    def __init__(self, x, y, width, label, min_value, max_value, value):
        self.rect = pygame.Rect(x, y, width, 18)
        self.label = label
        self.min_value = float(min_value)
        self.max_value = float(max_value)
        self.value = float(value)
        self.dragging = False

    def _clamp(self, value):
        return max(self.min_value, min(self.max_value, value))

    def _value_to_x(self):
        span = self.max_value - self.min_value
        if span == 0:
            return self.rect.left
        return self.rect.left + (self.value - self.min_value) / span * self.rect.width

    def _x_to_value(self, x):
        ratio = (x - self.rect.left) / self.rect.width
        return self._clamp(self.min_value + ratio * (self.max_value - self.min_value))

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.dragging = True
                self.value = self._x_to_value(event.pos[0])
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            self.value = self._x_to_value(event.pos[0])

    def draw(self, surface, font):
        pygame.draw.rect(surface, TRACK_BG, self.rect, border_radius=8)
        knob_x = int(self._value_to_x())
        fill_rect = pygame.Rect(self.rect.left, self.rect.top, max(0, knob_x - self.rect.left), self.rect.height)
        pygame.draw.rect(surface, TRACK_FG, fill_rect, border_radius=8)
        pygame.draw.circle(surface, KNOB, (knob_x, self.rect.centery), 10)
        pygame.draw.circle(surface, (25, 25, 25), (knob_x, self.rect.centery), 10, 2)
        label_surface = font.render(f"{self.label}: {self.value:.2f}", True, WHITE)
        surface.blit(label_surface, (self.rect.left, self.rect.top - 24))


class LivePlot:
    def __init__(self, window_seconds=20.0):
        self.window_seconds = window_seconds
        self.t_values = deque()
        self.v_values = deque()
        self.vtheory_values = deque()
        self.a_values = deque()

        plt.ion()
        self.fig, (self.ax_v, self.ax_a, self.ax_vtheory) = plt.subplots(3, 1, sharex=True, figsize=(7, 5))
        self.fig.canvas.manager.set_window_title("Pivot Motion")

        self.v_line, = self.ax_v.plot([], [], color="tab:blue", label="v_x")
        self.a_line, = self.ax_a.plot([], [], color="tab:red", label="a_x")
        self.vtheory_line, = self.ax_vtheory.plot([], [], color="tab:green", label="v_xtheory")

        self.ax_v.set_ylabel("Velocity (px/s)")
        self.ax_vtheory.set_ylabel("Velocity (px/s)")
        self.ax_a.set_ylabel("Acceleration (px/s^2)")
        self.ax_a.set_xlabel("Time (s)")
        self.ax_v.grid(True, alpha=0.3)
        self.ax_vtheory.grid(True, alpha=0.3)
        self.ax_a.grid(True, alpha=0.3)
        self.ax_v.legend(loc="upper right")
        self.ax_vtheory.legend(loc="upper right")
        self.ax_a.legend(loc="upper right")
        self.fig.tight_layout()

    def update(self, t, v_x, a_x, th, th_dot):
        if not plt.fignum_exists(self.fig.number):
            return

        v_theory = xdot(th, th_dot)

        self.t_values.append(t)
        self.v_values.append(v_x)
        self.vtheory_values.append(v_theory)
        self.a_values.append(a_x)

        while self.t_values and (t - self.t_values[0] > self.window_seconds):
            self.t_values.popleft()
            self.v_values.popleft()
            self.vtheory_values.popleft()
            self.a_values.popleft()

        self.v_line.set_data(self.t_values, self.v_values)
        self.vtheory_line.set_data(self.t_values, self.vtheory_values)
        self.a_line.set_data(self.t_values, self.a_values)

        x_min = max(0.0, t - self.window_seconds)
        x_max = x_min + self.window_seconds
        self.ax_v.set_xlim(x_min, x_max)

        self.ax_v.relim()
        self.ax_v.autoscale_view(scalex=False, scaley=True)
        self.ax_vtheory.relim()
        self.ax_vtheory.autoscale_view(scalex=False, scaley=True)
        self.ax_a.relim()
        self.ax_a.autoscale_view(scalex=False, scaley=True)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def close(self):
        if plt.fignum_exists(self.fig.number):
            plt.close(self.fig)


def xdot(theta, thetadot):
    return -((2 * l) / (M + m)) * thetadot * np.cos(theta)


def simulate_pid(kp, ki, kd, duration=20.0, dt=5/60):
    """
    Simulate the system with given PID parameters and return cumulative loss.
    Loss = integral of (theta^2 + 0.001 * control_effort^2) over simulation time.
    """
    # Create fresh simulation state
    test_space = pymunk.Space()
    test_space.gravity = 0, 9
    test_b0 = test_space.static_body
    
    test_body = pymunk.Body(mass=m, moment=10)
    test_body.position = (WIDTH / 2 , HEIGHT / 2)
    test_circle = pymunk.Circle(test_body, radius=20)
    
    # Make pivot kinematic so it's unaffected by the swinging mass
    test_pivot = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
    test_pivot.position = WIDTH / 2, HEIGHT / 2
    
    test_joint = pymunk.PinJoint(test_pivot, test_body, (0, 0), (0, 0))
    test_space.add(test_pivot, test_body, test_circle, test_joint)
    
    # Run simulation
    t = 0.0
    integral_error = 0.0
    cumulative_loss = 0.0
    prev_th = 0.0
    target_theta = 0.0
    
    steps = int(duration / dt)
    for step in range(steps):
        # Compute theta as angle from upward vertical
        dx = test_body.position.x - test_pivot.position.x
        dy = test_body.position.y - test_pivot.position.y
        theta = np.arctan2(dx, -dy)  # angle from upward vertical, counterclockwise positive
        theta_dot = (theta - prev_th) / dt
        
        # PID control
        error = target_theta - theta
        integral_error = float(np.clip(integral_error + error * dt, -4.0, 4.0))
        acceleration = kp * error + ki * integral_error - kd * theta_dot
        
        # Wall collision prevention
        cart_x = test_pivot.position.x
        if cart_x < WALL_MARGIN:
            acceleration = max(acceleration, 20)
        elif cart_x > WIDTH - WALL_MARGIN:
            acceleration = min(acceleration, -20)
        
        acceleration = float(np.clip(acceleration, -FORCE_LIMIT, FORCE_LIMIT))
        
        # Set kinematic body velocity based on acceleration
        test_pivot.velocity = (acceleration * dt, 0.0)
        
        # Accumulate loss: penalize deviation from upright and control effort
        cumulative_loss += (theta**2 + 0.001 * (acceleration / FORCE_LIMIT)**2) * dt
        
        # Step physics
        test_space.step(dt)
        prev_th = theta
        t += dt
    
    return cumulative_loss


def optimize_pid(initial_kp=1.0, initial_ki=0.0, initial_kd=0.0, duration=50.0):
    """
    Find optimal PID parameters using scipy.optimize.minimize.
    
    Args:
        initial_kp, initial_ki, initial_kd: Initial guesses for gains
        duration: Simulation duration per evaluation (seconds)
    
    Returns:
        (kp, ki, kd): Optimal parameter values
    """
    print("Starting PID optimization...")
    print(f"Initial guess: Kp={initial_kp:.3f}, Ki={initial_ki:.3f}, Kd={initial_kd:.3f}")
    
    def objective(gains):
        kp, ki, kd = gains
        loss = simulate_pid(kp, ki, kd, duration=duration)
        print(f"  Evaluating: Kp={kp:.3f}, Ki={ki:.3f}, Kd={kd:.3f}  Loss={loss:.4f}")
        return loss
    
    # Define bounds for gains
    bounds = [(-1000.0, 1000.0), (-10.0, 10.0), (-500.0, 500.0)]  # Kp, Ki, Kd
    
    # Run optimization
    result = minimize(
        objective,
        x0=[initial_kp, initial_ki, initial_kd],
        method='L-BFGS-B',
        bounds=bounds,
        options={'ftol': 1e-5, 'maxiter': 50}
    )
    
    kp_opt, ki_opt, kd_opt = result.x
    print(f"\nOptimization complete!")
    print(f"Optimal parameters: Kp={kp_opt:.3f}, Ki={ki_opt:.3f}, Kd={kd_opt:.3f}")
    print(f"Final loss: {result.fun:.4f}\n")
    
    return kp_opt, ki_opt, kd_opt



class App:
    size = WIDTH, HEIGHT + CONTROL_HEIGHT

    def __init__(self, pivot_body, mass_body):
        pygame.init()
        self.screen = pygame.display.set_mode(self.size)
        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
        self.running = True
        self.clock = pygame.time.Clock()
        self.dt = 5 / 60
        self.t = 0.0
        self.pivot_body = pivot_body
        self.mass_body = mass_body
        self.prev_vx = self.pivot_body.velocity.x
        ratio = (self.mass_body.position.x - self.pivot_body.position.x) / l
        self.prev_th = np.arcsin(np.clip(ratio, -1.0, 1.0))
        self.integral_error = 0.0
        self.target_theta = 0.0
        self.font = pygame.font.SysFont("arial", 16)
        self.live_plot = LivePlot(window_seconds=40.0) if ENABLE_LIVE_PLOTS else None
        self.sliders = [
            Slider(20, HEIGHT + 45, WIDTH - 20, "P", -200.0, 200.0, 50.0),
            Slider(20, HEIGHT + 100, WIDTH - 20, "I", -1.0, 1.0, 0.1),
            Slider(20, HEIGHT + 155, WIDTH - 20, "D", -50.0, 50.0, 20.0),
        ]

    def force_from_error(self, theta, theta_dot):
        error = self.target_theta - theta
        self.integral_error = float(np.clip(self.integral_error + error * self.dt, -4.0, 4.0))
        kp = self.sliders[0].value
        ki = self.sliders[1].value
        kd = self.sliders[2].value
        acceleration = kp * error + ki * self.integral_error - kd * theta_dot
        
        # Wall collision prevention: apply corrective force if cart is near boundaries
        cart_x = self.pivot_body.position.x
        if cart_x < WALL_MARGIN:
            # Too close to left wall, push right
            acceleration = max(acceleration, 20)
        elif cart_x > WIDTH - WALL_MARGIN:
            # Too close to right wall, push left
            acceleration = min(acceleration, -20)
        
        return float(np.clip(acceleration, -FORCE_LIMIT, FORCE_LIMIT))

    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                for slider in self.sliders:
                    slider.handle_event(event)

            self.screen.fill(GRAY)
            space.debug_draw(self.draw_options)
            pygame.draw.rect(self.screen, CONTROL_BG, (0, HEIGHT, WIDTH, CONTROL_HEIGHT))

            # Compute theta as angle from upward vertical
            dx = self.mass_body.position.x - self.pivot_body.position.x
            dy = self.mass_body.position.y - self.pivot_body.position.y
            theta = np.arctan2(dx, -dy)  # angle from upward vertical, counterclockwise positive
            theta_dot = (theta - self.prev_th) / self.dt
            accel_x = self.force_from_error(theta, theta_dot)
            # Set kinematic pivot velocity based on acceleration
            self.pivot_body.velocity = (accel_x * self.dt, 0.0)

            pygame.draw.rect(self.screen, (15, 15, 15), (0, HEIGHT, WIDTH, 24))
            status_surface = self.font.render(f"theta={theta:.3f}  theta_dot={theta_dot:.3f}  accel={accel_x:.1f}", True, WHITE)
            self.screen.blit(status_surface, (12, HEIGHT + 2))

            for slider in self.sliders:
                slider.draw(self.screen, self.font)

            pygame.display.update()
            space.step(self.dt)
            self.clock.tick(60)

            self.t += self.dt
            v_x = self.pivot_body.velocity.x
            a_x = (v_x - self.prev_vx) / self.dt
            self.prev_vx = v_x
            self.prev_th = theta
            if self.live_plot is not None:
                self.live_plot.update(self.t, v_x, a_x, theta, theta_dot)

        if self.live_plot is not None:
            self.live_plot.close()
        pygame.quit()


body = pymunk.Body(mass=m, moment=10)
body.position = (WIDTH / 2 - 100, HEIGHT / 2)
circle = pymunk.Circle(body, radius=20)

# Make pivot kinematic so it's unaffected by the swinging mass
pivot = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
pivot.position = WIDTH / 2, HEIGHT / 2

joint = pymunk.PinJoint(pivot, body, (0, 0), (0, 0))
space.add(pivot, body, circle, joint)

# Run optimization if enabled
app = App(pivot, body)
if ENABLE_OPTIMIZATION:
    kp_opt, ki_opt, kd_opt = optimize_pid(initial_kp=1.0, initial_ki=0.0, initial_kd=0.0, duration=500.0)
    # Load optimized values into sliders
    app.sliders[0].value = kp_opt
    app.sliders[1].value = ki_opt
    app.sliders[2].value = kd_opt
    print("Sliders updated with optimized parameters. Ready to run simulation.\n")

app.run()
