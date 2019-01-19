import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sys import stdout
from tqdm.autonotebook import tqdm
from matplotlib.figure import Figure
from matplotlib.axes import Axes

# Own files
from src.FasterFFMpegWriter import FasterFFMpegWriter

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

    """ Progress-bar handle, initialized with none. """
    pbar: tqdm = None

    def __init__(self, x: np.array, y: np.array, c: np.array, fps: float = 30, xlim: list = [-10, 10], ylim: list = [-10, 10], xsolid: np.array = None, ysolid: np.array = None, r: float = 1):
        """
        Initialize a new animation

        # Parameters #
        x: numpy array of the x positions
        """
        self.x = x
        self.y = y
        self.c = c
        self.fps = fps
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
        self.fig = plt.figure()
        self.ax = self.fig.add_axes([0, 0, 1, 1], frameon=True)
        self.sctr = self.ax.scatter(
            self.x[:, 0], self.y[:, 0], c=self.c[:, 0], s=s)

        # Add the solid if any
        if (self.xsolid is not None) and (self.ysolid is not None):
            self.ax.scatter(self.xsolid, self.ysolid, s=(np.ones(len(self.xsolid)) * (self.r ** 2 / 4)))

        # Add the (pressure) colorbar
        #self.fig.colorbar(self.sctr, label='Pressure [Pa]')

        # Decorate the plot
        self.ax.grid()
        self.ax.set_xlim(self.xlim[0], self.xlim[1])
        self.ax.set_ylim(self.ylim[0], self.ylim[1])
        self.ax.set_xlabel('X [m]')
        self.ax.set_ylabel('Y [m]')
        self.ax.set_title('Dam Break 2D')

        # Create progress-bar
        self.pbar = tqdm(total=self.t_max, unit='fps', desc='Exporting animation')
        self.pbar.update() # Increment with one

        # Create a writer
        #writer = FasterFFMpegWriter('-vcodec', codec, '-framerate', f'{self.fps}')
        writer = FasterFFMpegWriter()

        # Create and save the animation
        animation = FuncAnimation(
            self.fig, self.update, frames=range(self.t_max))
        animation.save(file, writer=writer)

    def update(self, frame_number: int):
        # Sometimes called without a framenumber for some weird reason.
        if not frame_number:
            # Return the artist, requires a trailing comma.
            return self.sctr,

        # Update the progress-bar, increment by one (1).
        self.pbar.update()

        # The colors (Pressure)
        self.sctr.set_array(self.c[:, frame_number])

        # Update the x-y
        xx = self.x[:, frame_number]
        yy = self.y[:, frame_number]
        self.sctr.set_offsets(np.transpose(np.array([xx, yy])))

        # Return the artist, requires a trailing comma.
        return self.sctr,
