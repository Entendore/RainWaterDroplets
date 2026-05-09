import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Grid size
size = 250
layers = 3  # number of overlapping droplet layers

# Diffusion rates per layer
diff_rates = [0.12, 0.10, 0.08]

# Reaction parameters
F, k = 0.03, 0.06
dt = 1.0

# Ripple parameters
damping = 0.995

# Initialize layers
R_layers = [0.5*np.ones((size,size)) for _ in range(layers)]
G_layers = [0.5*np.ones((size,size)) for _ in range(layers)]
B_layers = [0.5*np.ones((size,size)) for _ in range(layers)]
wave_layers = [np.zeros((size,size)) for _ in range(layers)]
wave_prev_layers = [np.zeros((size,size)) for _ in range(layers)]
vx_layers = [np.zeros((size,size)) for _ in range(layers)]
vy_layers = [np.zeros((size,size)) for _ in range(layers)]

def add_droplet(layer):
    x, y = np.random.randint(20, size-20, 2)
    r = np.random.randint(10, 20)
    Y, X = np.ogrid[-r:r, -r:r]
    mask = X**2 + Y**2 <= r**2
    val = np.linspace(0.2,0.7,mask.sum())
    color_choice = np.random.choice(['R','G','B'])
    if color_choice == 'R':
        R_layers[layer][x-r:x+r, y-r:y+r][mask] = val
    elif color_choice == 'G':
        G_layers[layer][x-r:x+r, y-r:y+r][mask] = val
    else:
        B_layers[layer][x-r:x+r, y-r:y+r][mask] = val
    wave_layers[layer][x-r:x+r, y-r:y+r][mask] += np.linspace(0.2,0.5,mask.sum())
    vx_layers[layer][x-r:x+r, y-r:y+r][mask] += np.random.uniform(-0.5,0.5,mask.sum())
    vy_layers[layer][x-r:x+r, y-r:y+r][mask] += np.random.uniform(-0.5,0.5,mask.sum())

def laplacian(Z):
    return (
        -4*Z
        + np.roll(Z,1,axis=0)
        + np.roll(Z,-1,axis=0)
        + np.roll(Z,1,axis=1)
        + np.roll(Z,-1,axis=1)
    )

def advect(Z, vx, vy):
    coords_x, coords_y = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')
    x_new = np.clip(coords_x - vx, 0, size-1)
    y_new = np.clip(coords_y - vy, 0, size-1)
    x0, y0 = x_new.astype(int), y_new.astype(int)
    return Z[x0, y0]

def update(frame):
    for layer in range(layers):
        # Add droplet occasionally per layer
        if frame % (40 + layer*10) == 0:
            add_droplet(layer)
        
        # Ripple propagation
        wave_new = (2*wave_layers[layer] - wave_prev_layers[layer] + laplacian(wave_layers[layer])*0.5)*damping
        wave_prev_layers[layer] = wave_layers[layer].copy()
        wave_layers[layer] = wave_new

        # Reaction-diffusion
        LR, LG, LB = laplacian(R_layers[layer]), laplacian(G_layers[layer]), laplacian(B_layers[layer])
        dR = diff_rates[layer]*LR - R_layers[layer]*G_layers[layer]*B_layers[layer] + F*(1 - R_layers[layer])
        dG = diff_rates[layer]*LG - G_layers[layer]*B_layers[layer]*R_layers[layer] + F*(1 - G_layers[layer])
        dB = diff_rates[layer]*LB - B_layers[layer]*R_layers[layer]*G_layers[layer] + F*(1 - B_layers[layer])
        R_layers[layer] += dR*dt
        G_layers[layer] += dG*dt
        B_layers[layer] += dB*dt

        # Advect with swirling velocities
        R_layers[layer] = advect(R_layers[layer], vx_layers[layer], vy_layers[layer])
        G_layers[layer] = advect(G_layers[layer], vx_layers[layer], vy_layers[layer])
        B_layers[layer] = advect(B_layers[layer], vx_layers[layer], vy_layers[layer])

        # Decay velocities
        vx_layers[layer] *= 0.95
        vy_layers[layer] *= 0.95

    # Combine layers with wave-based modulation
    R_vis = np.zeros((size,size))
    G_vis = np.zeros((size,size))
    B_vis = np.zeros((size,size))
    for layer in range(layers):
        R_vis += R_layers[layer] + 0.2*np.sin(wave_layers[layer]*6 + frame*0.02 + layer)
        G_vis += G_layers[layer] + 0.2*np.sin(wave_layers[layer]*7 + frame*0.025 + layer*1.5)
        B_vis += B_layers[layer] + 0.2*np.sin(wave_layers[layer]*8 + frame*0.03 + layer*3.0)

    # Normalize to 0-1
    R_vis = np.clip(R_vis/layers, 0, 1)
    G_vis = np.clip(G_vis/layers, 0, 1)
    B_vis = np.clip(B_vis/layers, 0, 1)

    rgb = np.dstack((R_vis, G_vis, B_vis))
    mat.set_data(rgb)
    return [mat]

# Plot setup
fig, ax = plt.subplots(figsize=(6,6))
mat = ax.imshow(np.dstack((R_vis:=np.zeros((size,size)),G_vis:=np.zeros((size,size)),B_vis:=np.zeros((size,size)))), interpolation='bilinear')
ax.axis('off')

ani = FuncAnimation(fig, update, frames=4000, interval=30, blit=True)
plt.show()
