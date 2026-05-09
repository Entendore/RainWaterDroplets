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

def add_droplet():
    x, y = np.random.randint(20, size-20, 2)
    r = np.random.randint(5, 15)
    # Smooth gradient droplet
    Y, X = np.ogrid[-r:r, -r:r]
    mask = X**2 + Y**2 <= r**2
    color_choice = np.random.choice(['R','G','B'])
    val = np.linspace(0.3,0.7,mask.sum())
    if color_choice == 'R':
        R[x-r:x+r, y-r:y+r][mask] = val
    elif color_choice == 'G':
        G[x-r:x+r, y-r:y+r][mask] = val
    else:
        B[x-r:x+r, y-r:y+r][mask] = val
    # Smooth ripple
    wave[x-r:x+r, y-r:y+r][mask] += np.linspace(0.2,0.5,mask.sum())

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

    # Occasionally add smooth droplet
    if frame % 50 == 0:
        add_droplet()

    # Smooth ripple propagation
    wave_new = (2*wave - wave_prev + laplacian(wave)*0.5)*damping
    wave_prev = wave.copy()
    wave = wave_new

    # Reaction-diffusion for colors (smooth interactions)
    LR, LG, LB = laplacian(R), laplacian(G), laplacian(B)
    dR = Du*LR - R*G*B + F*(1 - R)
    dG = Dv*LG - G*B*R + F*(1 - G)
    dB = Dw*LB - B*R*G + F*(1 - B)
    R += dR*dt
    G += dG*dt
    B += dB*dt

    # Smooth psychedelic modulation
    R_vis = np.clip(R + 0.2*np.sin(wave*5 + frame*0.02), 0, 1)
    G_vis = np.clip(G + 0.2*np.sin(wave*6 + frame*0.025 + 1.5), 0, 1)
    B_vis = np.clip(B + 0.2*np.sin(wave*7 + frame*0.03 + 3.0), 0, 1)

    rgb = np.dstack((R_vis, G_vis, B_vis))
    mat.set_data(rgb)
    return [mat]

# Plot setup
fig, ax = plt.subplots(figsize=(6,6))
mat = ax.imshow(np.dstack((R,G,B)), interpolation='bilinear')
ax.axis('off')

ani = FuncAnimation(fig, update, frames=2000, interval=30, blit=True)
plt.show()
