import pygame
import sys
import pymunk
import pymunk.pygame_util
import numpy as np
from control import get_K, get_energy, get_target_energy, calc_F_LQR, calc_F_swing
from filters import EWMA, MedianFitler, MovingAverage

# initialize the game window
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
FPS = 150

# Pymunk setup
space = pymunk.Space()
space.gravity = (0.0, 10.0) # px/s^2
draw_options = pymunk.pygame_util.DrawOptions(screen)

# Game state
running = True

# Physical Parameters
M = 1.0     # kg
m = 0.15    # kg
l = 100     # pixels
g = 100      # pixels/s^2

# initial conditions (radians, rad/s)
x0 = last_x = WIDTH // 2
vx0 = 0
theta0 = np.pi-0.15
last_theta = ( theta0 + 2 * np.pi ) % ( 2 * np.pi )
omega0 = 0

# compute bob offset so theta=0 means bob is directly above cart
rx = l * np.sin(theta0)
ry = - l * np.cos(theta0)          # negative = above cart

# Define the cart
cart_w, cart_h = 120, 30
cart_y = HEIGHT // 2
cart = pymunk.Body(M, float('inf'))
cart.position = (x0, cart_y)
cart.velocity = (vx0,0)
cart_shape = pymunk.Poly.create_box(cart, (cart_w, cart_h))

# Constraint the cart to move horizontally
groove = pymunk.GrooveJoint(space.static_body, cart, (50, cart_y), (WIDTH - 50, cart_y), (0,0))
groove.collide_bodies = False

# Define the pendulum mass
radius = 15
moment = pymunk.moment_for_circle(m,0,radius)
mass = pymunk.Body(m, moment)
mass.position = cart.position + pymunk.Vec2d(-rx, ry)
mass_shape = pymunk.Circle(mass, radius)

# choose joint anchors so the pin connects cart center to bob center:
# (this is the simplest: both anchors at (0,0) local coordinates)
pin = pymunk.PinJoint(cart, mass, (0, 0), (0, 0))

# set bob linear velocity corresponding to angular velocity about the cart:
# v = omega x r  =>  v = (-omega * ry, omega * rx)
mass.velocity = cart.velocity + pymunk.Vec2d(-omega0 * ry, omega0 * rx)
space.add(cart, cart_shape, groove, mass, mass_shape, pin)
font = pygame.font.SysFont("Arial", 18)

# Get the K matrix from the control module
K = get_K(M, m, l, g)
target_energy = get_target_energy(m, l, g)

# Define hardware limits
max_motor_force = 50.0  # The max force your steppers can handle without skipping
energy_threshold = 500  # How close to E_target before we ease off the throttle

# Define Deadzone +-5degs from the linear regime limit to avoid fast switching
linear_regime = np.deg2rad(15)
linear = False
energy = True
if abs(theta0) < linear_regime:
    linear = True
    energy = False

# Blending mechanic for the handoff from swing up to LQR controll
time = 1e-3 # 100ms
frames = time * FPS # s * frames / s --> frames
w = 0.5 # blending variable [0,1]

def get_blended_force(theta, u_swing, u_lqr):
    # Define your window (in radians)
    # e.g., 20 degrees for the outer edge, 5 degrees for full LQR
    outer_limit = np.deg2rad(20)  # ~20 degrees
    inner_limit = np.deg2rad(5)  # ~5 degrees
    
    abs_theta = abs(theta)

    if abs_theta > outer_limit:
        # Too far away, just swing up
        w = 0.0
    elif abs_theta < inner_limit:
        # Close enough, full LQR
        w = 1.0
    else:
        # Linear interpolation between the two
        # (outer - current) / (outer - inner)
        w = (outer_limit - abs_theta) / (outer_limit - inner_limit)

    # The final control law
    return (1 - w) * u_swing + w * u_lqr


# Instanciate the filters
spike_killer = MedianFitler(window_size=5)
smoother = EWMA(alpha=0.05) # closer to 0 gives more weight to old values
force_arr = list()

# Controller UI state
controller_enabled = True
button_rect = pygame.Rect(WIDTH - 170, 10, 160, 34)
BUTTON_ON_COLOR = (50, 200, 50)
BUTTON_OFF_COLOR = (200, 50, 50)
BUTTON_TEXT_COLOR = (255, 255, 255)
CLIP_MAX = 5000.0


while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        # mouse click toggles controller when clicking the button
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if button_rect.collidepoint(event.pos):
                controller_enabled = not controller_enabled
        # keyboard toggle
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_c:
                controller_enabled = not controller_enabled
    
    # Physics
    dt = 1 / FPS
    space.step(dt)

    # compute pendulum angle theta (0 = upright)
    dx = cart.position.x - mass.position.x 
    dy = cart.position.y - mass.position.y
    theta = ( np.arctan2(dx, dy) )# angle measured from upward vertical (radians)
    # theta_continuous = ( theta + 2 * np.pi ) % ( 2 * np.pi )# angle measured from upward vertical (radians)

    # Compute linear and angular velocities
    omega = (theta - last_theta) / dt
    v = (cart.position.x - last_x) / dt

    last_x = cart.position.x
    last_theta = theta

    # Define the state vector with numerical values
    x = np.array([cart.position.x - (WIDTH//2), v, theta, omega])

    # print(x)
    current_energy = get_energy(cart.position.x, v, theta, omega, M, m, l, g)


    # Compute the force F = - K . x (K matrix and state vector evaluated) and apply it on the cart
    if abs(theta) < linear_regime:
        F = calc_F_LQR(K, x)

    else:
        F = calc_F_swing(target_energy, current_energy, x, v, theta, omega, max_motor_force, energy_threshold)

    # Ensure F is a scalar and clip to a safe maximum
    try:
        F = float(F)
    except Exception:
        F = float(np.array(F).flatten()[0])

    F = np.clip(F, -max_motor_force, max_motor_force)
    F = get_blended_force(theta, calc_F_swing(target_energy, current_energy, x, v, theta, omega, max_motor_force, energy_threshold), calc_F_LQR(K, x))
    F = smoother.apply(spike_killer.apply(F))
    force_arr.append(F)

    # Apply force only when controller is enabled
    if controller_enabled:
        cart.apply_force_at_world_point((F, 0), cart.position)



    # --- draw ---
    screen.fill((255, 255, 255))
    space.debug_draw(draw_options)

    # Draw controller toggle button
    btn_color = BUTTON_ON_COLOR if controller_enabled else BUTTON_OFF_COLOR
    pygame.draw.rect(screen, btn_color, button_rect, border_radius=6)
    label = "Controller: ON" if controller_enabled else "Controller: OFF"
    label_surf = font.render(label, True, BUTTON_TEXT_COLOR)
    screen.blit(label_surf, (button_rect.x + 10, button_rect.y + 6))

    info = f"cart_x={cart.position.x:.1f} cart_v={v:.1f} theta={theta:.3f} rad omega={omega:.3f} force={F:.3f}"
    screen.blit(font.render(info, True, (220,220,220)), (8,8))

    pygame.display.flip()
    # clock.tick(60)

pygame.quit()

import matplotlib.pyplot as plt
frames = np.arange(0,len(force_arr))
force_arr = np.array(force_arr)
plt.plot(frames, force_arr)
plt.xlabel("Frame number")
plt.ylabel("Force (N)")
plt.grid(True)
plt.show()
    
sys.exit()