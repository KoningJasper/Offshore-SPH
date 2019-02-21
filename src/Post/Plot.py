import sys
import tempfile
import subprocess
from time import perf_counter

# External
import numpy as np
from tqdm import tqdm
from colorama import Fore, Style
import pyqtgraph as pg
import pyqtgraph.exporters
from pyqtgraph.Qt import QtCore
from pyqtgraph.graphicsItems import TextItem
from pyqtgraph.graphicsItems.GradientEditorItem import Gradients

# Own
from src.ColorBar import ColorBar

#TODO: Test for ffmpeg
class Plot():
    def __init__(self, file: str, slowdown: float = 8.0, fps: int = 10, title: str ='SPH Plot', xmin: float = None, xmax: float = None, ymin: float = None, ymax: float = None, height: int = 700, width: int = 500, exportAllFrames: bool = False):
        """
            Initializes a new plot export. This requires ffmpeg to be present in system path.

            Parameters
            ----------

            file: str
                path to the export (.npz) file

            slowdown: float
                The amount of slow down applied to the animation. A slowdown of 1.0 is equal to actual speed.

            fps: int
                The number of frames per second rendered as output, the larger the number the smoother the video but longer export times.

            title: str
                The title shown on the plot

            xmin: float
                Minimum x-range to show, if none determined automaticall from first frame. Both xmin and xmax have to be set to be effective.

            xmax: float
                Maximum x-range to show, if none determined automaticall from first frame. Both xmin and xmax have to be set to be effective.

            ymin: float
                Minimum y-range to show, if none determined automaticall from first frame. Both ymin and ymax have to be set to be effective.

            ymax: float
                Maximum y-range to show, if none determined automaticall from first frame. Both ymin and ymax have to be set to be effective.

            height: int
                The height of the exported frames

            width: int
                The width of the exported frames

            exportAllFrames: bool
                Should all frames be exported or only at framerate intervals. Takes significantly longer.
        """

        self.input     = file
        self.slowdown  = slowdown
        self.video_fps = fps
        self.title     = title
        self.height    = height
        self.width     = width

        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

        self.exportAllFrames = exportAllFrames

    def save(self, file: str):
        """
            Renders the plot to an .mp4 file.
        
            Parameters
            ----------

            file: str
                path to export the file to, including .mp4
        """
        print('Starting plot export.')

        start = perf_counter()

        self.output = file

        self._load()
        print(f'{Fore.GREEN}File loaded successfully.{Style.RESET_ALL}')
        self._init_plot()

        # Create a temp-dir for storage of files
        self.tempdir = tempfile.TemporaryDirectory()
        print(f'Exporting frames to directory: {self.tempdir.name}')

        t: float = 0
        dt_max: float = 1 / self.fps # Maximum delta between frames
        dt_c: float = 0.0 # Current delta
        frames: int = 0 # Total number of exported frames.
        for frame in tqdm(range(self.steps - 1), desc='Exporting frames'):
            dt_c += self.dts[frame]

            # Perform an export if 
            if dt_c > dt_max or frame == 0 or self.exportAllFrames == True:
                # Draw
                self._update_frame(frame, t)

                # Export to temp
                self._export(frames)

                dt_c = 0.0
                frames += 1

            # Increment time
            t += self.dts[frame]

        print(f'{Fore.GREEN}Frame export complete.{Style.RESET_ALL}')
        print(f'Exported a total of {Fore.YELLOW}{frames}{Style.RESET_ALL} frames.')
        print(f'Converting to animation')
        self._export_mp4()

        end = perf_counter() - start
        print(f'Rendered to file in {Fore.GREEN}{end:f}{Style.RESET_ALL} [s]')

        print(f'{Fore.YELLOW}Plot will now throw an error that can be ignored.{Style.RESET_ALL}')

    # Private methods down from here.

    def _load(self):
        """ Loads the required parameters from the file. """
        
        f = np.load(self.input)
        self.data = f['data']
        self.dts = f['dts']


        # Compute average frames per second.
        self.steps    = len(self.data)
        self.duration = np.sum(self.dts)
        self.fps      = np.round(self.steps / self.duration / self.slowdown) # Frames per second

    def _export(self, frame: int):
        self.exporter = pg.exporters.ImageExporter(self.pw.plotItem)
        self.exporter.params.param('width').setValue(self.height, blockSignal=self.exporter.widthChanged)
        self.exporter.params.param('height').setValue(self.width, blockSignal=self.exporter.heightChanged)

        frame_str = "{:06}".format(frame)
        self.exporter.export(f'{self.tempdir.name}/export_{frame_str}.png')

    def _update_frame(self, frame: int, time: float) -> None:
        self.pl.setData(x=self.data[frame]['x'], y=self.data[frame]['y'], symbolBrush=self.cm.map(self.data[frame]['p'] / 1000, 'qcolor'), symbolSize=self.sZ)
        self.txtItem.setText(f't = {time:f} [s]')

    def _init_plot(self):
        """ Initialize the plot. """
        start: float = perf_counter()

        # use less ink
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        pg.setConfigOptions(antialias=True)

        # Plot
        self.pw = pg.plot(title=self.title, labels={'left': 'Y [m]', 'bottom': 'X [m]'})

        if (self.xmin and self.xmax):
            self.pw.setXRange(self.xmin, self.xmax)
        else:
            # Determine automatically
            xmin = self.data[0]['x'].min() - 1
            xmax = self.data[0]['x'].max() + 1
            self.pw.setXRange(xmin, xmax)

        if (self.ymin and self.ymax):
            self.pw.setYRange(self.ymin, self.ymax)
        else:
            # Determine automatically
            ymin = self.data[0]['y'].min() - 1
            ymax = self.data[0]['y'].max() + 1
            self.pw.setYRange(ymin, ymax)

        # Text
        # TODO: Place the text, better automatically instead of hard-coded.
        self.txtItem = pg.TextItem(text = f't = 0.00000 [s]')
        self.pw.scene().addItem(self.txtItem)
        [[xmin, xmax], [_, _]] = self.txtItem.getViewBox().viewRange()
        xrange = xmax - xmin
        self.txtItem.translate(350.0 - xrange, 90.0)

        # make colormap
        c_range = np.linspace(0, 1, num=10)
        colors = []
        for cc in c_range:
            colors.append([cc, 0.0, 1 - cc, 1.0]) # Blue to red color spectrum
        stops = np.round(c_range * (np.max(self.data[0]['p'])) / 1000, 0)
        self.cm = pg.ColorMap(stops, np.array(colors))
        
        # make colorbar, placing by hand
        # TODO: Place color bar automatically instead of manually.
        cb = ColorBar(cmap=self.cm, width=10, height=200, label='Pressure [kPa]')
        self.pw.scene().addItem(cb)
        cb.translate(610.0, 90.0)

        # Initial points
        num = len(self.data[0]['x'])
        self.sZ = (40 * (2 / (num ** 0.5)))
        self.pl = self.pw.plot(self.data[0]['x'], self.data[0]['y'], pen=None, symbol='o', symbolBrush=self.cm.map(self.data[0]['p'] / 1000, 'qcolor'), symbolPen=None, symbolSize=self.sZ)

    def _export_mp4(self):
        """ Exports gathered frames to mp4 file using ffmpeg. """
        subprocess.run(f'ffmpeg -hide_banner -loglevel panic -y -framerate {self.video_fps} -i "{self.tempdir.name}\\export_%06d.png" -s:v {self.height}x{self.width} -c:v libx264 \
-profile:v high -crf 20 -pix_fmt yuv420p {self.output}"')

        # Cleanup export dir
        self.tempdir.cleanup()
