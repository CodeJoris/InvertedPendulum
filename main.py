import pymunk
import pymunk.pygame_util
import pygame
from collections import deque
import numpy as np

import matplotlib.pyplot as plt

GRAY = (220, 220, 220)
space = pymunk.Space()
space.gravity = 0, 9
b0 = space.static_body
WIDTH, HEIGHT = 400, 400
l=100 # length of the pendulum
m=1 # mass of the mass at the end of pendulum
M=1 # mass of the pivot (cart)


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


def xdot(theta,thetadot):
    return -((2*l)/(M+m)) * thetadot * np.cos(theta)

class App:
    size = WIDTH, HEIGHT
    def __init__(self, pivot_body, mass_body):
        pygame.init()
        self.screen = pygame.display.set_mode(self.size, pygame.HIDDEN)
        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
        self.running = True
        self.dt = 0.2
        self.t = 0.0
        self.pivot_body = pivot_body
        self.mass_body = mass_body
        self.prev_vx = self.pivot_body.velocity.x
        self.prev_th = np.arcsin((self.mass_body.position.x - self.pivot_body.position.x)/l)
        self.live_plot = LivePlot(window_seconds=40.0)

    def run(self):
        while self.running:
            pygame.time.delay(1)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
            self.screen.fill(GRAY)
            space.debug_draw(self.draw_options)
            pygame.display.update()
            space.step(self.dt)

            self.t += self.dt
            th = np.arcsin((self.mass_body.position.x - self.pivot_body.position.x)/l)
            th_dot = (th - self.prev_th) / self.dt
            v_x = self.pivot_body.velocity.x
            a_x = (v_x - self.prev_vx) / self.dt
            self.prev_vx = v_x
            self.prev_th = th
            self.live_plot.update(self.t, v_x, a_x, th, th_dot)

        self.live_plot.close()
        pygame.quit()

if __name__ == '__main__':
    body = pymunk.Body(mass=m, moment=10)
    body.position = (WIDTH / 2 -100, HEIGHT / 2 )
    circle = pymunk.Circle(body, radius=20)


    pivot = pymunk.Body(mass=M, moment=float('inf'))
    pivot.position = WIDTH / 2, HEIGHT / 2

    groove = pymunk.GrooveJoint(b0, pivot, (0, HEIGHT / 2), (WIDTH, HEIGHT / 2), (0, 0))
    groove.collide_bodies = False

    joint = pymunk.PinJoint(pivot, body, (0, 0), (0, 0))
    space.add(pivot, body, circle, groove, joint)


    App(pivot, body).run()
