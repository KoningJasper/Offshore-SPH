from matplotlib.animation import FuncAnimation
from sys import stdout
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

# TODO: Move this constant to a parameter.
plt.rcParams['animation.ffmpeg_path'] = 'C:\\ffmpeg\\bin\\ffmpeg.exe'


class Animation:
    x: np.array
    y: np.array
    c: np.array
    fps: float
    t_max: float
    fig: Figure = None
    sctr = None
    ax: Axes = None
    xlim: list
    ylim: list
    xsolid: np.array
    ysolid: np.array

    def __init__(self, x: np.array, y: np.array, c: np.array, fps: float = 30, xlim: list = [-10, 10], ylim: list = [-10, 10], xsolid: np.array = None, ysolid: np.array = None, r: float = 1):
        """
        Initialize a new animation

        # Parameters #
        x: numpy array of the x positions
        """
        self.x = x
        self.y = y
        self.c = c
        self.fps = 30
        self.t_max = x.shape[1]
        self.xlim = xlim
        self.ylim = ylim
        self.xsolid = xsolid
        self.ysolid = ysolid
        self.r = r

    def export(self, file: str, codec: str = 'libx264'):
        """
        Export as animation

        Parameters:
        file: The file to export to, should end with .mp4
        codec: The codec ffmpeg should use. Can be 'libx264' or 'libx265'.
        """

        # Create size array
        s: np.array = np.ones(len(self.x[:, 1])) * (self.r ** 2 / 4)

        # Create the figure with the scatter plot
        self.fig = plt.figure(figsize=(8, 8), dpi=200)
        self.ax = self.fig.add_axes([0, 0, 1, 1], frameon=True)
        self.sctr = self.ax.scatter(
            self.x[:, 0], self.y[:, 0], c=self.c[:, 0], s=s)

        # Add the solid if any
        if (self.xsolid is not None) and (self.ysolid is not None):
            self.ax.scatter(self.xsolid, self.ysolid)

        # Add the (pressure) colorbar
        self.fig.colorbar(self.sctr, label='Pressure [Pa]')

        # Decorate the plot
        self.ax.grid()
        self.ax.set_xlim(self.xlim[0], self.xlim[1])
        self.ax.set_ylim(self.ylim[0], self.ylim[1])
        self.ax.set_xlabel('X [m]')
        self.ax.set_ylabel('Y [m]')
        self.ax.set_title('Dam Break 2D')

        # Create and save the animation
        animation = FuncAnimation(
            self.fig, self.update, frames=range(self.t_max))
        animation.save(file, extra_args=[
                       '-vcodec', codec, '-framerate', f'{self.fps}'])

    def update(self, frame_number: int):
        if not frame_number:
            return

        stdout.flush()
        percent = round(frame_number / self.t_max * 100, 2)
        print(f'{percent}%', end='\r')

        # The colors (Pressure)
        self.sctr.set_array(self.c[:, frame_number])

        xx = self.x[:, frame_number]
        yy = self.y[:, frame_number]
        self.sctr.set_offsets(np.transpose(np.array([xx, yy])))
