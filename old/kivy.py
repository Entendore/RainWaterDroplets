import numpy as np
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.widget import Widget
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.dropdown import DropDown
from kivy.properties import NumericProperty, ObjectProperty, StringProperty, ListProperty
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.core.window import Window
from scipy.ndimage import map_coordinates, laplace
from matplotlib import cm

# --- KV Language String for UI Layout ---
# This defines the look and layout of our application's widgets.
KV_STRING = '''
<RainRootWidget>:
    orientation: 'vertical'
    
    # Top bar for style selection
    BoxLayout:
        size_hint_y: None
        height: 50
        padding: 10
        spacing: 10
        
        Label:
            text: 'Style:'
            size_hint_x: None
            width: 50
            color: 1, 1, 1, 1
        
        Button:
            id: style_button
            text: root.current_style
            on_release: root.show_style_dropdown()
    
    # Main display area for the rain animation
    Image:
        id: rain_display
        allow_stretch: True
        on_touch_down: root.on_image_touch_down(args[1])
    
    # Bottom area for sliders
    BoxLayout:
        size_hint_y: None
        height: 150
        orientation: 'vertical'
        padding: 10
        spacing: 5
        
        Label:
            text: f'Rain Intensity: {int(rain_intensity_slider.value)}'
            color: 1, 1, 1, 1
        Slider:
            id: rain_intensity_slider
            min: 10
            max: 200
            value: 50
            step: 5
            
        Label:
            text: f'Ripple Damping: {damping_slider.value:.3f}'
            color: 1, 1, 1, 1
        Slider:
            id: damping_slider
            min: 0.99
            max: 0.999
            value: 0.998
            step: 0.001

        Label:
            text: f'Droplet Size: {int(droplet_size_slider.value)}'
            color: 1, 1, 1, 1
        Slider:
            id: droplet_size_slider
            min: 3
            max: 20
            value: 8
            step: 1
'''

# --- Core Simulation Logic ---
# This class handles all the mathematical calculations, separate from the UI.
class RainSimulation:
    def __init__(self, size=400, layers=2, style='realism'):
        self.size = size
        self.layers = layers
        self.frame_count = 0
        self.load_style(style)

    def load_style(self, style_name):
        self.style = style_name
        style_params = {
            'realism': {'diff_rates': np.array([0.14, 0.14]), 'damping': 0.998, 'velocity_decay': 0.98, 'drop_velocity_range': (1.0, 1.5), 'base_color': 0.05, 'visual_type': 'water_specular'},
            'psychedelic': {'diff_rates': np.array([0.12, 0.12]), 'damping': 0.995, 'velocity_decay': 0.95, 'drop_velocity_range': (2.0, 3.5), 'base_color': 0.1, 'visual_type': 'psychedelic_sine'},
            'inferno': {'diff_rates': np.array([0.08, 0.08]), 'damping': 0.999, 'velocity_decay': 1.0, 'drop_velocity_range': (0, 0), 'base_color': 0.95, 'visual_type': 'colormap'}
        }
        self.params = style_params[style_name]
        self.reset_state()

    def reset_state(self):
        base = self.params['base_color']
        self.state = {
            'R': np.full((self.layers, self.size, self.size), base),
            'G': np.full((self.layers, self.size, self.size), base),
            'B': np.full((self.layers, self.size, self.size), base),
            'wave': np.zeros((self.layers, self.size, self.size)),
            'wave_prev': np.zeros((self.layers, self.size, self.size)),
            'vx': np.zeros((self.layers, self.size, self.size)),
            'vy': np.zeros((self.layers, self.size, self.size)),
        }

    def step(self, rain_intensity, droplet_size, damping, feed_rate):
        self.frame_count += 1
        
        # Add raindrops
        interval = max(1, int(15 * (100 / rain_intensity)))
        if self.frame_count % interval == 0:
            self.add_raindrop(droplet_size)

        # Wave Propagation
        wave_new = (2*self.state['wave'] - self.state['wave_prev'] + laplace(self.state['wave'], mode='constant', cval=0.0)*0.5) * damping
        self.state['wave_prev'] = self.state['wave'].copy()
        self.state['wave'] = wave_new

        # Reaction-Diffusion
        LR, LG, LB = laplace(self.state['R'], mode='constant', cval=0.0), laplace(self.state['G'], mode='constant', cval=0.0), laplace(self.state['B'], mode='constant', cval=0.0)
        diff_rates_reshaped = self.params['diff_rates'][:, np.newaxis, np.newaxis]
        dR = diff_rates_reshaped * LR - self.state['R']*self.state['G']*self.state['B'] + feed_rate*(1 - self.state['R'])
        dG = diff_rates_reshaped * LG - self.state['G']*self.state['B']*self.state['R'] + feed_rate*(1 - self.state['G'])
        dB = diff_rates_reshaped * LB - self.state['B']*self.state['R']*self.state['G'] + feed_rate*(1 - self.state['B'])
        self.state['R'] += dR
        self.state['G'] += dG
        self.state['B'] += dB

        # Advection
        if self.params['velocity_decay'] < 1.0:
            self.state['R'] = self.advect(self.state['R'], self.state['vx'], self.state['vy'])
            self.state['G'] = self.advect(self.state['G'], self.state['vx'], self.state['vy'])
            self.state['B'] = self.advect(self.state['B'], self.state['vx'], self.state['vy'])
            self.state['vx'] *= self.params['velocity_decay']
            self.state['vy'] *= self.params['velocity_decay']

        np.clip(self.state['R'], 0, 1, out=self.state['R'])
        np.clip(self.state['G'], 0, 1, out=self.state['G'])
        np.clip(self.state['B'], 0, 1, out=self.state['B'])

    def add_raindrop(self, droplet_size):
        layer_idx = np.random.randint(0, self.layers)
        r = droplet_size
        x = np.random.randint(r, self.size - r)
        y = r
        Y, X = np.ogrid[-r:r, -r:r]
        mask = X**2 + Y**2 <= r**2
        sl = (slice(layer_idx, layer_idx + 1), slice(x-r, x+r), slice(y-r, y+r))
        
        if self.style == 'inferno':
            val = 0.05
            self.state['R'][sl][:, mask] = val; self.state['G'][sl][:, mask] = val; self.state['B'][sl][:, mask] = val
        elif self.style == 'realism':
            self.state['R'][sl][:, mask] = 0.1; self.state['G'][sl][:, mask] = 0.5; self.state['B'][sl][:, mask] = 0.9
        else: # psychedelic
            val = np.linspace(0.2, 0.7, mask.sum())
            color_choice = np.random.choice(['R', 'G', 'B'])
            self.state[color_choice][sl][:, mask] = val
            
        self.state['wave'][sl][:, mask] += 0.8
        v_min, v_max = self.params['drop_velocity_range']
        self.state['vy'][sl][:, mask] += np.random.uniform(v_min, v_max, mask.sum())

    def add_splash(self, x, y, droplet_size):
        layer_idx = np.random.randint(0, self.layers)
        r = droplet_size * 2
        Y, X = np.ogrid[-r:r, -r:r]
        mask = X**2 + Y**2 <= r**2
        sl = (slice(layer_idx, layer_idx + 1), slice(x-r, x+r), slice(y-r, y+r))
        self.state['wave'][sl][:, mask] += 1.0
        self.state['vx'][sl][:, mask] += np.random.uniform(-1.0, 1.0, mask.sum())
        self.state['vy'][sl][:, mask] += np.random.uniform(-1.0, 1.0, mask.sum())

    def advect(self, field, vx, vy):
        y_coords, x_coords = np.mgrid[0:self.size, 0:self.size]
        x_prev = x_coords - vx
        y_prev = y_coords - vy
        output = np.empty_like(field)
        for i in range(self.layers):
            coords = np.array([y_prev.ravel(), x_prev.ravel()])
            advected_layer = map_coordinates(field[i], coords, order=1, mode='nearest', prefilter=False)
            output[i] = advected_layer.reshape(self.size, self.size)
        return output

    def composite(self):
        R_avg = np.mean(self.state['R'], axis=0)
        G_avg = np.mean(self.state['G'], axis=0)
        B_avg = np.mean(self.state['B'], axis=0)
        wave_avg = np.mean(self.state['wave'], axis=0)

        if self.params['visual_type'] == 'water_specular':
            specular = np.clip(wave_avg * 1.5, 0, 1)
            R_vis = np.clip(R_avg + specular * 0.5, 0, 1)
            G_vis = np.clip(G_avg + specular * 0.7, 0, 1)
            B_vis = np.clip(B_avg + specular * 1.0, 0, 1)
        elif self.params['visual_type'] == 'psychedelic_sine':
            phase_r = wave_avg * 6 + self.frame_count * 0.05
            phase_g = wave_avg * 7 + self.frame_count * 0.06 + 2.0
            phase_b = wave_avg * 8 + self.frame_count * 0.07 + 4.0
            R_vis = np.clip(R_avg + 0.5 * np.sin(phase_r), 0, 1)
            G_vis = np.clip(G_avg + 0.5 * np.sin(phase_g), 0, 1)
            B_vis = np.clip(B_avg + 0.5 * np.sin(phase_b), 0, 1)
        elif self.params['visual_type'] == 'colormap':
            intensity_map = np.clip(1.0 - (R_avg + G_avg + B_avg) / 3.0 + wave_avg, 0, 1)
            colored_map = cm.get_cmap('inferno')(intensity_map)
            R_vis, G_vis, B_vis = colored_map[:,:,0], colored_map[:,:,1], colored_map[:,:,2]
        else:
            R_vis, G_vis, B_vis = R_avg, G_avg, B_avg
        
        # Convert to 0-255 range and flatten for Kivy texture
        rgb_image = np.dstack((R_vis, G_vis, B_vis))
        return (rgb_image * 255).astype(np.ubyte).flatten()

# --- Main Kivy App and Widget ---
class RainRootWidget(BoxLayout):
    # These properties will be automatically updated by the sliders
    rain_intensity = NumericProperty(50)
    damping = NumericProperty(0.998)
    droplet_size = NumericProperty(8)
    feed_rate = NumericProperty(0.035)
    
    # Property to hold the current style name
    current_style = StringProperty('realism')
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.simulation = RainSimulation(style=self.current_style)
        Clock.schedule_interval(self.update_animation, 1.0 / 30.0) # 30 FPS

    def show_style_dropdown(self):
        dropdown = DropDown()
        styles = ['realism', 'psychedelic', 'inferno']
        for style_name in styles:
            btn = Button(text=style_name, size_hint_y=None, height=40)
            btn.bind(on_release=lambda b: self.select_style(style_name))
            dropdown.add_widget(btn)
        
        dropdown.open(self.ids['style_button'])

    def select_style(self, style_name):
        self.current_style = style_name
        self.ids['style_button'].text = style_name
        self.simulation.load_style(style_name)
        # Reset sliders to new style's defaults
        self.damping = self.simulation.params['damping']

    def on_image_touch_down(self, touch):
        if self.ids['rain_display'].collide_point(*touch.pos):
            # Convert touch position to simulation grid coordinates
            img_widget = self.ids['rain_display']
            x = int(touch.x / img_widget.width * self.simulation.size)
            y = int(touch.y / img_widget.height * self.simulation.size)
            self.simulation.add_splash(x, y, self.droplet_size)

    def update_animation(self, dt):
        # Get slider values from properties
        intensity = self.rain_intensity
        damping = self.damping
        size = self.droplet_size
        feed = self.feed_rate

        # Step the simulation
        self.simulation.step(intensity, size, damping, feed)
        
        # Composite the final image and update the Kivy texture
        flat_image_data = self.simulation.composite()
        
        # Create and update the texture
        texture = Texture.create(size=(self.simulation.size, self.simulation.size), colorfmt='rgb')
        texture.blit_buffer(flat_image_data, colorfmt='rgb', bufferfmt='ubyte')
        
        # Apply the texture to the Image widget
        self.ids['rain_display'].texture = texture
        self.ids['rain_display'].canvas.ask_update()


class RainApp(App):
    def build(self):
        # Set a dark background for the window
        Window.clearcolor = (0.1, 0.1, 0.1, 1)
        return RainRootWidget()

if __name__ == '__main__':
    RainApp().run()