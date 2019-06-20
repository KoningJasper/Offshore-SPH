import sys, tempfile, subprocess, math, shutil
from typing import List, Tuple
from time import perf_counter

# External
import numpy as np
import h5py
from tqdm import tqdm
from colorama import Fore, Style
import pyqtgraph as pg
import pyqtgraph.exporters
from pyqtgraph.Qt import QtCore
from pyqtgraph.graphicsItems import TextItem
from pyqtgraph.graphicsItems.GradientEditorItem import Gradients

# Own
from src.ColorBar import ColorBar
from src.Common import ParticleType

class Plot():
    def __init__(self, file: str, slowdown: float = 8.0, fps: int = 10, title: str ='SPH Plot', xmin: float = None, xmax: float = None, ymin: float = None, ymax: float = None, height: int = 700, width: int = 500, exportAllFrames: bool = False):
        """
            Initializes a new plot export. This requires ffmpeg to be present in system path.

            Parameters
            ----------

            file: str
                path to the export (.hdf5) file

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

        self.check()

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

    def check(self):
        """
            Verifies that ffmpeg is in the path.
        """
        if shutil.which('ffmpeg') is None:
            raise Exception("ffmpeg is not defined, check you path. Without ffmpeg, plots cannot be created.")

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
        print('{0}File loaded successfully.{1}'.format(Fore.GREEN, Style.RESET_ALL))
        self._init_plot()

        # Create a temp-dir for storage of files
        self.tempdir = tempfile.mkdtemp()
        print('Exporting frames to directory: {0}'.format(self.tempdir))

        frames, t_series = self._calcFrames()
        for i, frame in tqdm(enumerate(frames), desc='Exporting frames', total=(len(frames) - 1), unit='frame'):
            # Draw
            self._update_frame(frame, t_series[i])

            # Export to temp
            self._export(i)

        print('{0}Frame export complete.{1}'.format(Fore.GREEN, Style.RESET_ALL))
        print('Exported a total of {0}{2}{1} frames.'.format(Fore.YELLOW, Style.RESET_ALL, len(frames)))
        print('Converting to animation')
        self._export_mp4()

        end = perf_counter() - start
        print('Rendered to file in {0}{2:f}{1} [s]'.format(Fore.GREEN, Style.RESET_ALL, end))

        print('{0}Plot will now throw an error that can be ignored.{1}'.format(Fore.YELLOW, Style.RESET_ALL, end))

    # Private methods down from here.

    def _load(self):
        """ Loads the required parameters from the file. """
        
        with h5py.File(self.input, 'r') as h5f:
            pA              = h5f['particleArray'][:]
            self.x          = h5f['x'][:]
            self.y          = h5f['y'][:]
            self.p          = h5f['p'][:]
            self.dts        = h5f['dt_a'][:]        # The actual time
            self.settleTime = h5f['settleTime'][()]

        # Compute duration
        self.fluid_ind  = pA['label'] == ParticleType.Fluid
        self.border_ind = pA['label'] == ParticleType.Boundary
        self.temp_ind   = pA['label'] == ParticleType.TempBoundary
        self.coupl_ind  = pA['label'] == ParticleType.Coupled
        self.duration   = np.sum(self.dts)

    def _export(self, frame: int):
        self.exporter = pg.exporters.ImageExporter(self.pw.plotItem)
        self.exporter.params.param('width').setValue(self.height, blockSignal=self.exporter.widthChanged)
        self.exporter.params.param('height').setValue(self.width, blockSignal=self.exporter.heightChanged)

        frame_str = "{:06}".format(frame)
        self.exporter.export('{0}/export_{1}.png'.format(self.tempdir, frame_str))

    def _calcFrames(self) -> Tuple[List[int], List[int]]:
        # Compute the target times, with avg. frame-rate.
        total_frames = math.ceil(self.duration * self.slowdown * self.video_fps)
        targets = np.linspace(0, 1, num=total_frames) * self.duration

        # Compute the frames to export
        t = 0; exp_frames = 0
        frames = []; t_series = []
        for frame in range(len(self.x)):
            if t >= targets[exp_frames]:
                frames.append(frame)
                t_series.append(t - self.settleTime)
                exp_frames += 1
            t += self.dts[frame]

        # Add the final frame
        frames.append(len(self.x) - 1)
        t_series.append(self.duration - self.settleTime)
        
        return frames, t_series

    def _update_frame(self, frame: int, time: float) -> None:
        self.pl_f.setData(x=self.x[frame, self.fluid_ind], y=self.y[frame, self.fluid_ind], symbolBrush=self.cm.map(self.p[frame, self.fluid_ind] / 1000, 'qcolor'), symbolSize=self.sZ)
        self.pl_b.setData(x=self.x[frame, self.border_ind], y=self.y[frame, self.border_ind], symbolBrush=pg.mkBrush('k'), symbolPen=None, symbolSize=self.sZ)
        self.pl_c.setData(x=self.x[frame, self.coupl_ind], y=self.y[frame, self.coupl_ind], symbolBrush=pg.mkBrush('m'), symbolPen=None, symbolSize=self.sZ)

        if time >= self.settleTime:
            self.pl_t.clear()
        else:
            self.pl_t.setData(x=self.x[frame, self.temp_ind], y=self.y[frame, self.temp_ind], symbolBrush=pg.mkBrush('y'), symbolPen=None, symbolSize=self.sZ)
        self.txtItem.setText('t = {0:f} [s]'.format(time))

    def _init_plot(self):
        """ Initialize the plot. """
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
            xmin = self.x[0].min() - 1
            xmax = self.x[0].max() + 1
            self.pw.setXRange(xmin, xmax)

        if (self.ymin and self.ymax):
            self.pw.setYRange(self.ymin, self.ymax)
        else:
            # Determine automatically
            ymin = self.y[0].min() - 1
            ymax = self.y[0].max() + 1
            self.pw.setYRange(ymin, ymax)

        # Text
        # TODO: Place the text, better automatically instead of hard-coded.
        self.txtItem = pg.TextItem(text = 't = 0.00000 [s]')
        self.pw.scene().addItem(self.txtItem)
        [[xmin, xmax], [_, _]] = self.txtItem.getViewBox().viewRange()
        xrange = xmax - xmin
        self.txtItem.translate(350.0 - xrange, 90.0)

        # make colormap
        c_range = np.linspace(0, 1, num=10); colors = []
        for cc in c_range:
            colors.append([cc, 0.0, 1 - cc, 1.0]) # Blue to red color spectrum
        
        stops = np.round(c_range * np.max(self.p[0,self.fluid_ind]) / 1000, 0)
        self.cm = pg.ColorMap(stops, np.array(colors))

        # make colorbar, placing by hand
        # TODO: Place color bar automatically instead of manually.
        cb = ColorBar(cmap=self.cm, width=10, height=200, label='Pressure [kPa]')
        self.pw.scene().addItem(cb)
        cb.translate(610.0, 90.0)

        # Initial points
        num = len(self.x[0])
        self.sZ = (40 * (2 / (num ** 0.5)))
        self.pl_f = self.pw.plot(self.x[0, self.fluid_ind], self.y[0, self.fluid_ind], pen=None, symbol='o', symbolBrush=self.cm.map(self.p[0, self.fluid_ind] / 1000, 'qcolor'), symbolPen=None, symbolSize=self.sZ)
        self.pl_b = self.pw.plot(self.x[0, self.border_ind], self.y[0, self.border_ind], pen=None, symbol='o', symbolBrush=pg.mkBrush('k'), symbolPen=None, symbolSize=self.sZ)
        self.pl_t = self.pw.plot(self.x[0, self.temp_ind], self.y[0, self.temp_ind], pen=None, symbol='o', symbolBrush=pg.mkBrush('y'), symbolPen=None, symbolSize=self.sZ)
        self.pl_c = self.pw.plot(self.x[0, self.coupl_ind], self.y[0, self.coupl_ind], pen=None, symbol='o', symbolBrush=pg.mkBrush('m'), symbolPen=None, symbolSize=self.sZ)

    def _export_mp4(self):
        """ Exports gathered frames to mp4 file using ffmpeg. """
        subprocess.run('ffmpeg -hide_banner -loglevel panic -y -framerate {0} -i "{1}/export_%06d.png" -s:v {2}x{3} -c:v libx264 \
-profile:v high -crf 20 -pix_fmt yuv420p "{4}"'.format(self.video_fps, self.tempdir, self.height, self.width, self.output), shell=True)

        # Cleanup export dir
        shutil.rmtree(self.tempdir)
