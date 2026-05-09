import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Grid size
size = 250

# Diffusion rates per color
Du, Dv, Dw = 0.12, 0.10, 0.08

# Reaction parameters
F, k = 0.03, 0.06
dt = 1.0

# Ripple parameters
damping = 0.995
wave = np.zeros((size, size))
wave_prev = np.zeros((size, size))

# Color channels (start smooth)
R = 0.5*np.ones((size, size))
G = 0.5*np.ones((size, size))
B = 0.5*np.ones((size, size))

# Velocity field for swirling
vx = np.zeros((size, size))
vy = np.zeros((size, size))

def add_droplet():
    x, y = np.random.randint(20, size-20, 2)
    r = np.random.randint(10, 20)
    # Smooth radial gradient droplet
    Y, X = np.ogrid[-r:r, -r:r]
    mask = X**2 + Y**2 <= r**2
    val = np.linspace(0.2,0.7,mask.sum())
    color_choice = np.random.choice(['R','G','B'])
    if color_choice == 'R':
        R[x-r:x+r, y-r:y+r][mask] = val
    elif color_choice == 'G':
        G[x-r:x+r, y-r:y+r][mask] = val
    else:
        B[x-r:x+r, y-r:y+r][mask] = val
    # Ripple effect
    wave[x-r:x+r, y-r:y+r][mask] += np.linspace(0.2,0.5,mask.sum())
    # Add small swirling velocities
    vx[x-r:x+r, y-r:y+r][mask] += np.random.uniform(-0.5,0.5,mask.sum())
    vy[x-r:x+r, y-r:y+r][mask] += np.random.uniform(-0.5,0.5,mask.sum())

def laplacian(Z):
    return (
        -4*Z
        + np.roll(Z, 1, axis=0)
        + np.roll(Z, -1, axis=0)
        + np.roll(Z, 1, axis=1)
        + np.roll(Z, -1, axis=1)
    )

def advect(Z, vx, vy):
    # Backward particle tracing (semi-Lagrangian)
    coords_x, coords_y = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')
    x_new = np.clip(coords_x - vx, 0, size-1)
    y_new = np.clip(coords_y - vy, 0, size-1)
    x0, y0 = x_new.astype(int), y_new.astype(int)
    return Z[x0, y0]

def update(frame):
    global R, G, B, wave, wave_prev, vx, vy

    # Add droplet every few frames
    if frame % 40 == 0:
        add_droplet()

    # Ripple propagation
    wave_new = (2*wave - wave_prev + laplacian(wave)*0.5)*damping
    wave_prev = wave.copy()
    wave = wave_new

    # Reaction-diffusion for colors
    LR, LG, LB = laplacian(R), laplacian(G), laplacian(B)
    dR = Du*LR - R*G*B + F*(1 - R)
    dG = Dv*LG - G*B*R + F*(1 - G)
    dB = Dw*LB - B*R*G + F*(1 - B)
    R += dR*dt
    G += dG*dt
    B += dB*dt

    # Advect colors with swirling velocities
    R = advect(R, vx, vy)
    G = advect(G, vx, vy)
    B = advect(B, vx, vy)

    # Slowly decay velocities for smooth flow
    vx *= 0.95
    vy *= 0.95

    # Smooth psychedelic modulation
    R_vis = np.clip(R + 0.25*np.sin(wave*6 + frame*0.02), 0, 1)
    G_vis = np.clip(G + 0.25*np.sin(wave*7 + frame*0.025 + 1.5), 0, 1)
    B_vis = np.clip(B + 0.25*np.sin(wave*8 + frame*0.03 + 3.0), 0, 1)

    rgb = np.dstack((R_vis, G_vis, B_vis))
    mat.set_data(rgb)
    return [mat]

# Plot setup
fig, ax = plt.subplots(figsize=(6,6))
mat = ax.imshow(np.dstack((R,G,B)), interpolation='bilinear')
ax.axis('off')

ani = FuncAnimation(fig, update, frames=3000, interval=30, blit=True)
plt.show()
