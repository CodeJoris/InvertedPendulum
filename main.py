import pymunk
import pymunk.pygame_util
import pygame
from collections import deque
import numpy as np

import matplotlib.pyplot as plt

GRAY = (220, 220, 220)
CONTROL_BG = (35, 35, 35)
TRACK_BG = (70, 70, 70)
TRACK_FG = (120, 180, 255)
KNOB = (240, 240, 240)
WHITE = (245, 245, 245)
ENABLE_LIVE_PLOTS = False
space = pymunk.Space()
space.gravity = 0, 9
b0 = space.static_body
WIDTH, HEIGHT = 1000, 600
CONTROL_HEIGHT = 180
l = 100  # length of the pendulum
m = 1  # mass of the mass at the end of pendulum
M = 1000  # mass of the pivot (cart)


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
            Slider(20, HEIGHT + 45, WIDTH - 20, "P", -200.0, 200.0, 1.0),
            Slider(20, HEIGHT + 100, WIDTH - 20, "I", -1.0, 1.0, 0.0),
            Slider(20, HEIGHT + 155, WIDTH - 20, "D", -50.0, 50.0, 0.0),
        ]

    def force_from_error(self, theta, theta_dot):
        error = self.target_theta - theta
        self.integral_error = float(np.clip(self.integral_error + error * self.dt, -4.0, 4.0))
        kp = self.sliders[0].value
        ki = self.sliders[1].value
        kd = self.sliders[2].value
        force = kp * error + ki * self.integral_error - kd * theta_dot
        return float(np.clip(force, -6000.0, 6000.0))

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

            ratio = (self.mass_body.position.x - self.pivot_body.position.x) / l
            theta = np.arcsin(np.clip(ratio, -1.0, 1.0))
            theta_dot = (theta - self.prev_th) / self.dt
            force_x = self.force_from_error(theta, theta_dot)
            self.pivot_body.apply_force_at_world_point((M * force_x, 0.0), self.pivot_body.position)

            pygame.draw.rect(self.screen, (15, 15, 15), (0, HEIGHT, WIDTH, 24))
            status_surface = self.font.render(f"theta={theta:.3f}  theta_dot={theta_dot:.3f}  force={force_x:.1f}", True, WHITE)
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

pivot = pymunk.Body(mass=M, moment=float('inf'))
pivot.position = WIDTH / 2, HEIGHT / 2

groove = pymunk.GrooveJoint(b0, pivot, (0, HEIGHT / 2), (WIDTH, HEIGHT / 2), (0, 0))
groove.collide_bodies = False

joint = pymunk.PinJoint(pivot, body, (0, 0), (0, 0))
space.add(pivot, body, circle, groove, joint)

App(pivot, body).run()
