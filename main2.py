import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Grid size
size = 200

# Diffusion rates per color
Du, Dv, Dw = 0.18, 0.12, 0.08

# Reaction parameters
F, k = 0.035, 0.065
dt = 1.0

# Ripple parameters
damping = 0.97
wave = np.zeros((size, size))
wave_prev = np.zeros((size, size))

# Color channels
R = np.ones((size, size))
G = np.ones((size, size))
B = np.ones((size, size))

def add_droplet():
    x, y = np.random.randint(10, size-10, 2)
    r = np.random.randint(2, 6)
    color_choice = np.random.choice(['R','G','B'])
    if color_choice == 'R':
        R[x-r:x+r, y-r:y+r] = np.random.uniform(0.2,0.5)
    elif color_choice == 'G':
        G[x-r:x+r, y-r:y+r] = np.random.uniform(0.2,0.5)
    else:
        B[x-r:x+r, y-r:y+r] = np.random.uniform(0.2,0.5)
    # Add wave perturbation
    wave[x-r:x+r, y-r:y+r] += np.random.uniform(0.3, 0.6)

def laplacian(Z):
    return (
        -4*Z
        + np.roll(Z, 1, axis=0)
        + np.roll(Z, -1, axis=0)
        + np.roll(Z, 1, axis=1)
        + np.roll(Z, -1, axis=1)
    )

def update(frame):
    global R, G, B, wave, wave_prev

    # Random droplets
    if np.random.rand() < 0.08:
        add_droplet()

    # Ripple propagation
    wave_new = (2*wave - wave_prev + laplacian(wave)*0.6)*damping
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

    # Psychedelic color modulation using waves + sine functions
    R_vis = np.clip(R + 0.3*np.sin(wave*10 + frame*0.1), 0, 1)
    G_vis = np.clip(G + 0.3*np.sin(wave*12 + frame*0.15 + 2), 0, 1)
    B_vis = np.clip(B + 0.3*np.sin(wave*14 + frame*0.2 + 4), 0, 1)

    rgb = np.dstack((R_vis, G_vis, B_vis))
    mat.set_data(rgb)
    return [mat]

# Plot setup
fig, ax = plt.subplots(figsize=(6,6))
mat = ax.imshow(np.dstack((R,G,B)), interpolation='bilinear')
ax.axis('off')

ani = FuncAnimation(fig, update, frames=2000, interval=30, blit=True)
plt.show()
