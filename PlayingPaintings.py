############################################################################
# Project: PlayingPaintings
# Author: Paola Gervasio
# https://github.com/pgerva/playing-paintings
#
# To cite this project: P. Gervasio, A. Quarteroni, D. Cassani.
#            Let the paintings play. (2022)
#            https://arxiv.org/abs/2206.14142
#
#  Copyright (C) 2022 by Paola Gervasio.
#
#   PlayingPaintings is free software; you can redistribute it and/or modify
#   it under the terms of the GNU Lesser General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   PlayingPaintings is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#   Lesser General Public License for more details.
#
#   You should have received a copy of the GNU Lesser General Public License
#   along with PlayingPaintings.  If not, see <http://www.gnu.org/licenses/>.
#
# ####################################################################

import os
from pathlib import Path
from os.path import exists
import sys
import warnings
import csv
import numpy as np # pip install numpy

from scipy import fft
import pywt  # pip install PyWavelets
import pywt.data
import librosa # pip install librosa
import soundfile # pip install soundfile
import PIL  # pip install Pillow
from PIL import Image

from PySide2 import QtGui, QtWidgets, QtCharts, QtCore, QtMultimedia # pip install PySide2

# import PySide2 before matplotlib
import matplotlib # pip install matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

matplotlib.use("Qt5Agg")

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


class PaintingListComboBox(QtWidgets.QComboBox):
    # A ComboBox to read the names of the paintings from the file paintings_list_filename
    def __init__(self, paintings_list_filename):
        super().__init__()
        # self.paintings_list_filename = paintings_list_filename
        with open(paintings_list_filename, "r") as input_file:
            menu_csv = list(csv.reader(input_file, delimiter="\n"))
        menu_csv1 = []
        for sublist in menu_csv:
            for item in sublist:
                menu_csv1.append(item)
        self.addItems(menu_csv1)


class ScaledPixmap(QtGui.QPixmap):
    # define a pixmap and scale it,
    # save its scaled version in self.scaled_pixmap
    def __init__(self, filename, directory):
        super().__init__()
        paintings_dir_small = "./_small/"
        self.load(paintings_dir_small + filename + "_small.png")
        self.scaled_pixmap = self.scale()
        # print("filename", filename)
        # print("directory",paintings_dir_small)
        # print("directory+filename",paintings_dir_small+filename)

    def scale(self):
        # method to scale the pixmap
        pw = self.width()  # minimum width of the image
        ph = self.height()  # minimum height of the image
        # the scaled image has to be saved in a new variable
        # print("pw=",pw, " ph=",ph)
        scaled_pixmap = " "
        if pw > 0 and ph > 0:
            scaled_pixmap = self.scaled(pw, ph, QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation)
        return scaled_pixmap


class MusicListWidget(QtWidgets.QListWidget):
    # A Widget to read the list of music tracks
    def __init__(self, music_list_filename):
        super().__init__()
        self.setSelectionMode(QtWidgets.QListWidget.ExtendedSelection)
        self.resize(min(self.width(), 160), min(self.width(), 160))
        with open(music_list_filename, "r") as input_file:
            menu_csv = list(csv.reader(input_file, delimiter="\n"))
        menu_csv1 = []
        for sublist in menu_csv:
            for item in sublist:
                menu_csv1.append(item)
        self.addItems(menu_csv1)


class TransformButtonGroup(QtWidgets.QButtonGroup):
    # ButtonGroup to select the discrete transform
    def __init__(self, mainwindow):
        super().__init__()
        self.w = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()
        self.w.setLayout(layout)
        rdb1 = QtWidgets.QRadioButton('DWT - 1D unrolling')
        rdb1.setCheckable(True)
        rdb2 = QtWidgets.QRadioButton('DWT - full 2D')
        rdb2.setCheckable(True)
        rdb3 = QtWidgets.QRadioButton('DFT - 1D unrolling')
        rdb3.setCheckable(True)
        rdb4 = QtWidgets.QRadioButton('DFT - full 2D')
        rdb4.setCheckable(True)

        self.addButton(rdb1, 0)
        self.addButton(rdb2, 1)
        self.addButton(rdb3, 2)
        self.addButton(rdb4, 3)

        layout.addWidget(rdb1)
        layout.addWidget(rdb2)
        layout.addWidget(rdb3)
        layout.addWidget(rdb4)

        self.buttonClicked.connect(mainwindow.set_selected_transform)


class WaveletComboBox(QtWidgets.QComboBox):
    # build a combobox for the wavelets
    def __init__(self):
        super().__init__()
        self.addItems(['Haar', 'db3', 'db5', 'sym8', 'bior5.5'])


class WaveletLabel(QtWidgets.QWidget):
    # build a widget with a double label,
    # the second label has a smaller fontsize.
    def __init__(self, string1, string2):
        super().__init__()
        label1 = QtWidgets.QLabel(string1)
        label2 = QtWidgets.QLabel(string2)
        label2.setFont(QtGui.QFont('Arial', 10))
        ll = QtWidgets.QVBoxLayout()
        ll.addWidget(label1)
        ll.addWidget(label2)
        self.setLayout(ll)


class SignalMplCanvas(FigureCanvasQTAgg):
    #  Matplot object to plot signals
    def __init__(self, width=4, height=1):
        # width and height in inches
        fig = Figure(figsize=(width, height))
        self.axes = fig.add_subplot(111)
        super().__init__(fig)
        self.axes.grid(True)
        self.axes.tick_params(axis='both', which='major', labelsize=10)
        # Add Axis Labels
        self.axes.set_xlabel("t (sec)", fontsize=10, loc='right')
        #self.axes.set_ylabel("Music Track   ", fontsize=10)
        fig.tight_layout(pad=0.02)

    def my_plot(self, data, samplerate, plotname, color):
        data = data.real / np.linalg.norm(data.real, np.inf)
        # samplerate, data = wavfile.read(filename)
        length = data.shape[0] / samplerate
        time = np.linspace(0., length, data.shape[0])
        self.axes.plot(time, data, color=color, linewidth=1)
        self.axes.set_xlim(time[0], time[-1])
        self.axes.set_ylim(-1, 1)
        self.axes.annotate(plotname, xy=(10, 5), xycoords='figure pixels',
                           color=color, fontsize=11, weight="bold")

    def clear_plot(self):
        self.axes.clear()
        self.axes.grid(True)
        self.axes.tick_params(axis='both', which='major', labelsize=10)
        # Add Axis Labels
        self.axes.set_xlabel("t (sec)", fontsize=10, loc='right')

        self.draw()


class TransformMplCanvas(FigureCanvasQTAgg):
    # build a PlotWidget to plot the signal read from file
    def __init__(self, width=6, height=1, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi, layout='constrained')
        self.axes = fig.add_subplot(111)
        super().__init__(fig)
        self.axes.tick_params(axis='both', which='major', labelsize=10)
        # Add Axis Label
        self.axes.set_xlabel("k (Hz)    ", fontsize=10, loc='right')
        self.axes.semilogx([1], [1])
        self.axes.grid(True)
        self.axes.figure.set_size_inches(380, 60)

    def my_plot_dwt(self, y, plotname, color):
        x = np.linspace(0, y.size, y.size)
        self.axes.semilogx(x, y, color=color, linewidth=1.0)
        # Set Range
        self.axes.set_xlim(1, x.size)
        self.axes.set_ylim(-np.amax(y), np.amax(y))
        self.axes.set_xlabel("k (Hz)   ", fontsize=10, loc='right')
        self.axes.annotate(plotname, xy=(10, 5), xycoords='figure pixels',
                           color=color, fontsize=11, weight="bold")

    def my_plot_dft(self, y, plotname, color, sample_rate):
        n = y.size
        x = fft.fftfreq(n, 1./sample_rate)[0:n//2]
        yy = np.abs(y[1:n//2])
        self.axes.semilogx(x[1:], yy, color=color, linewidth=1.0)
        # Set Range
        self.axes.set_xlim(1, x[-1])
        self.axes.set_ylim(0, np.amax(yy))
        self.axes.annotate(plotname, xy=(10, 5), xycoords='figure pixels',
                           color=color, fontsize=11, weight="bold")

    def clear_plot_dt(self):
        self.axes.clear()
        self.axes.tick_params(axis='both', which='major', labelsize=10)
        # Add Axis Labels
        self.axes.set_xlabel("k (Hz)  ", fontsize=10, loc='right')
        #self.axes.set_ylabel("Spectrum  ", fontsize=10)
        self.axes.semilogx([1], [1])
        self.axes.grid(True)
        self.draw()


class PieChart(QtWidgets.QLabel):
    # PieChart is a derived class of QLabel, in which I add a pixmap.
    def __init__(self):
        super().__init__()
        # creat a pixmap with white background
        canvas = QtGui.QPixmap()
        canvas.fill(QtCore.Qt.white)
        # save the pixmap in the Pixmap attribute of the Widget QLabel.
        self.setPixmap(canvas)
        # define an object of class QPieSeries with the data
        self.series = QtCharts.QtCharts.QPieSeries()
        # put the labels outside the pie
        self.series.setLabelsPosition(QtCharts.QtCharts.QPieSlice.LabelOutside)
        # define the Graphic Widget QChart.
        self.chart = QtCharts.QtCharts.QChart()
        # add the data for the pie
        self.chart.addSeries(self.series)
        # generate the pie by the animation
        self.chart.setAnimationOptions(QtCharts.QtCharts.QChart.SeriesAnimations)
        # align the legend on the right
        self.chart.legend().setAlignment(QtCore.Qt.AlignRight)
        # save the graphic widget in chartview
        self.chartview = QtCharts.QtCharts.QChartView(self.chart)
        self.chartview.setAlignment(QtCore.Qt.AlignTop)
        self.chart.layout().setContentsMargins(2, 2, 2, 2)
        
    def fill_pie(self, data, labels, colors):
        # append data and labels
        for s in range(data.shape[0]):
            _slice = QtCharts.QtCharts.QPieSlice(labels[s], data[s])
            _slice.setBrush(QtGui.QBrush(QtGui.QColor(colors[s])))
            self.series.append(_slice)
        self.series.setPieSize(14)
        # self.series.append(labels[slice],data[slice])
        # show the labels around the pie
        # self.series.setLabelsVisible(True)
        # modify the labels with the percentage
        for slice in self.series.slices():
            slice.setLabel("{:.1f}%".format(100 * slice.percentage()))
        # define the legend, without this instruction,
        # the legend would show the labels (that now are the percentages)
        # for slice in range(data.shape[0]):
        #     self.chart.legend().markers(self.series)[slice].setLabel(labels[slice])
        # apply antialiasing
        self.chartview.setRenderHint(QtGui.QPainter.Antialiasing)

    def clear_pie(self):
        self.series.clear()


class Legend(QtWidgets.QLabel):
    def __init__(self):
        super().__init__()
        canvas = QtGui.QPixmap(500, 100)
        canvas.fill(QtCore.Qt.white)
        self.setPixmap(canvas)

    def fill_legend(self, tracks, colors):
        painter = QtGui.QPainter(self.pixmap())
        pen = QtGui.QPen()
        pen.setWidth(8)
        for k in range(len(tracks)):
            pen.setColor(QtGui.QColor(colors[k]))
            painter.setPen(pen)
            painter.drawPoint(30, 20+k*20)
            pen.setColor(QtGui.QColor('black'))
            painter.setPen(pen)
            legend_text = tracks[k]
            music_name = os.path.splitext(legend_text)
            music_name=music_name[0]
            music_name1=music_name.replace("-"," ")
            painter.drawText(50, 25+k*20, music_name1)
        painter.end()
        self.update()

    def clear_legend(self):
        canvas = QtGui.QPixmap(500, 100)
        canvas.fill(QtCore.Qt.white)
        self.setPixmap(canvas)


class Distance(QtWidgets.QLabel):
    def __init__(self):
        super().__init__()
        canvas = QtGui.QPixmap(350, 180)
        canvas.fill(QtCore.Qt.white)
        self.setPixmap(canvas)

    def fill_distances(self, distance):
        painter = QtGui.QPainter(self.pixmap())
        pen = QtGui.QPen()
        pen.setColor(QtGui.QColor('black'))
        painter.setPen(pen)

        painter.drawText(50, 60,
                         ("The normalized distance between"))
        painter.drawText(50, 80,
                         ("the spectrum of the painting and"))
        painter.drawText(50, 100,
                         ("the spectrum of the new piece of music is"))
        painter.drawText(140, 140,
                         (str("{:.6f}".format(distance))))
        painter.end()
        self.update()

    def clear_distances(self):
        canvas = QtGui.QPixmap(350, 180)
        canvas.fill(QtCore.Qt.white)
        self.setPixmap(canvas)


class Player(QtMultimedia.QMediaPlayer):
    def __init__(self, directory, music):
        super().__init__()
        filename = os.path.join(directory, music)
        url = QtCore.QUrl.fromLocalFile(filename)
        self.setMedia(QtMultimedia.QMediaContent(url))


class PlayButton(QtWidgets.QPushButton):
    def __init__(self, icon_file):
        super().__init__()
        self.setMaximumWidth(60)
        self.setMinimumWidth(60)
        self.setMaximumHeight(30)
        self.setMinimumHeight(30)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(icon_file))
        self.setIcon(icon)
        self.setIconSize(QtCore.QSize(26, 26))



class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, music_dir, music_list_filename, paintings_dir, paintings_list_filename):
        super().__init__()
        self.setWindowTitle("Playing paintings")
        self.setGeometry(100, 100, 1300, 800)

        self.paintings_dir = paintings_dir
        self.music_dir = music_dir
        self.paintings_list_filename = paintings_list_filename
        self.music_list_filename = music_list_filename
        self.selected_tracks = list()
        self.n_selected_tracks = 0
        self.sample_rate = list()
        self.my_sample_rate = 44100
        self.painting_name = " "
        self.mother_wavelet = "db5"
        self.wave_nlevels = 8
        self.transform = 0

        self.color_tracks = ["#66b2ff","#3399ff", "#0066cc", "#003366"]
        self.color_painting = ["#ff0000", "#ec7d0d"]

        self.counter_go = 0
        self.alpha = np.zeros([4])
        self.alpha_percento = np.zeros([4])
        # local variables
        painting_name = ""

        ############################################################
        # the global container of the main window.
        global_container = QtWidgets.QWidget()
        # global layout
        global_layout = QtWidgets.QHBoxLayout()
        global_container.setLayout(global_layout)
        self.setCentralWidget(global_container)

        # left widget
        left_widget = QtWidgets.QGroupBox()
        left_layout = QtWidgets.QVBoxLayout()
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_widget.setLayout(left_layout)
        left_widget.setMaximumWidth(250)

        # right widget
        right_widget = QtWidgets.QGroupBox()
        right_layout = QtWidgets.QVBoxLayout()
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_widget.setLayout(right_layout)
        right_widget.setMinimumWidth(1050)

        # put the left and right widgets inside the global layout
        global_layout.addWidget(left_widget)
        global_layout.addWidget(right_widget)


# ##########################################################
#       # left widget: input
# ##########################################################
        left_widget_list = []
#
        # 0. the label Step 1.
        painting_label = QtWidgets.QLabel("Step 1. Choose the painting")
        left_widget_list.append(painting_label)

        # 1. the paintings menu
# read the list of the paintings from a file and select the painting.
        painting_menu = PaintingListComboBox(paintings_list_filename)
        left_widget_list.append(painting_menu)
# create the object for the scaled image (empty image)
        self.pixmap = ScaledPixmap(painting_name, self.music_dir)
        # send the signal to the slot self.load_image_gui to show the image in the gui
        painting_menu.currentTextChanged.connect(self.load_image_gui)
        self.selected_painting = painting_name

        # 2. The image
# define a widget Qlabel to load the scaled image.
        self.image = QtWidgets.QLabel(self)
        self.image.setAlignment(QtCore.Qt.AlignHCenter)
#  define the layout and store the widget
        image_layout = QtWidgets.QHBoxLayout()
        image_layout.setContentsMargins(10, 10, 10, 10)
        self.image.setLayout(image_layout)
        left_widget_list.append(self.image)

        # 3. The label Step 2
# Choice of the music tracks.
        music_label = QtWidgets.QLabel("Step 2. Choose up to 4 tracks")
        left_widget_list.append(music_label)

        # 4. The menu of the music tracks
        self.music_list = MusicListWidget(music_list_filename)
        self.music_list.itemSelectionChanged.connect(self.set_selected_music)
# define the layout and store the widget
        music_layout = QtWidgets.QVBoxLayout()
        self.music_list.setLayout(music_layout)
        left_widget_list.append(self.music_list)

        # 5. QvBox with the list of discrete transforms
# Choice of the discrete transform.
        transforms_widget = QtWidgets.QGroupBox()
        transforms_layout = QtWidgets.QVBoxLayout()
        transforms_widget.setLayout(transforms_layout)
        transforms_widget.setMaximumHeight(200)

        label = QtWidgets.QLabel("Step 3. Select the transform for the painting")
        #label.setFont(QtGui.QFont('Arial', 10))
        label.setFixedWidth(230)
        label.setWordWrap(True)
        transforms_layout.addWidget(label)
        label_description = QtWidgets.QLabel(
            "(the 1D-transform for the music will be of the same type)")
        label_description.setFont(QtGui.QFont('Arial', 8))
        label_description.setFixedWidth(230)
        label_description.setWordWrap(True)
        transforms_layout.addWidget(label_description)

# define the layout and store the widget
        self.transform_widget = TransformButtonGroup(self)
        transforms_layout.addWidget(self.transform_widget.w)

        left_widget_list.append(transforms_widget)

        # 6. mother wavelet
# Choice of the mother wavelet.
        mw_widget = QtWidgets.QGroupBox()
        mw_layout = QtWidgets.QHBoxLayout()
        mw_widget.setLayout(mw_layout)
        mw_widget.setMaximumHeight(80)

        mw_layout.setContentsMargins(10, 10, 10, 10)
        self.wavelet = WaveletComboBox()
        label = WaveletLabel("Mother wavelet", "(only for DWT)")
        mw_layout.addWidget(label)
        label.setMaximumWidth(140)
        default = self.mother_wavelet
        self.wavelet.setCurrentText(default)
        self.wavelet.currentTextChanged.connect(self.select_mother_wavelet)
        mw_layout.addWidget(self.wavelet)

        left_widget_list.append(mw_widget)

        # 7. The number of levels
#   Set the number of levels for DWT.
        nlevels_widget = QtWidgets.QGroupBox()
        nlevels_layout = QtWidgets.QHBoxLayout()
        nlevels_widget.setLayout(nlevels_layout)
        nlevels_widget.setMaximumHeight(70)
        nlevels_layout.setContentsMargins(1, 1, 1, 1)

        label = WaveletLabel("Number of levels", "(only for DWT)")
        nlevels_layout.addWidget(label)

        self.wavelet_levels = QtWidgets.QSpinBox()
        self.wavelet_levels.setMinimum(2)
        self.wavelet_levels.setMaximum(15)
        self.wavelet_levels.setValue(self.wave_nlevels)
        self.wavelet_levels.valueChanged.connect(self.select_levels_wavelet)

        nlevels_layout.addWidget(self.wavelet_levels)

        left_widget_list.append(nlevels_widget)

        # move all the items of left_widget_list into left_layout
        for item in left_widget_list:
            left_layout.addWidget(item)

# ############################################################
# right layout (actions and output)
# ############################################################
#
        # 0. grid widget for the plot of signals and transforms
        plot_widget = QtWidgets.QGroupBox()
        plot_layout = QtWidgets.QGridLayout()
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_widget.setLayout(plot_layout)
        right_layout.addWidget(plot_widget)

        # 1. output widget
        output_widget = QtWidgets.QGroupBox()
        output_widget.setMinimumHeight(180)
        output_widget.setMaximumHeight(180)
        output_layout = QtWidgets.QHBoxLayout()
        output_layout.setContentsMargins(0, 0, 0, 0)
        output_widget.setLayout(output_layout)
        right_layout.addWidget(output_widget)

#       fill the plot_layout 0
        # define the widgets (3 columns) for the plot_layout
        self.signal_widget_list = [] # second column
        self.dt_widget_list = [] # third column
        
        # Column 0: playbuttons for playing sounds
        # 0
        self.music0_button_widget = QtWidgets.QWidget()
        self.music0_button_layout=QtWidgets.QVBoxLayout()
        self.music0_button_widget.setLayout(self.music0_button_layout)
        self.playbutton_music0 = PlayButton('play.png')
        self.playbutton_music0.setEnabled(False)
        self.music0_button_layout.addWidget(self.playbutton_music0,
                                              alignment=QtCore.Qt.AlignHCenter)
        self.pausebutton_music0 = PlayButton('pause.png')
        self.pausebutton_music0.setEnabled(False)
        self.music0_button_layout.addWidget(self.pausebutton_music0,
                                              alignment=QtCore.Qt.AlignHCenter)
        self.music0_button_widget.setMaximumWidth(100)
        plot_layout.addWidget(self.music0_button_widget, 1, 0,
                              alignment=QtCore.Qt.AlignHCenter)

        # 1
        self.music1_button_widget = QtWidgets.QWidget()
        self.music1_button_layout=QtWidgets.QVBoxLayout()
        self.music1_button_widget.setLayout(self.music1_button_layout)
        self.playbutton_music1 = PlayButton('play.png')
        self.playbutton_music1.setEnabled(False)
        self.music1_button_layout.addWidget(self.playbutton_music1,
                                              alignment=QtCore.Qt.AlignHCenter)
        self.pausebutton_music1 = PlayButton('pause.png')
        self.pausebutton_music1.setEnabled(False)
        self.music1_button_widget.setMaximumWidth(100)
        self.music1_button_layout.addWidget(self.pausebutton_music1,
                                              alignment=QtCore.Qt.AlignHCenter)
        plot_layout.addWidget(self.music1_button_widget, 2, 0,
                              alignment=QtCore.Qt.AlignHCenter)
        # 2
        self.music2_button_widget = QtWidgets.QWidget()
        self.music2_button_layout=QtWidgets.QVBoxLayout()
        self.music2_button_widget.setLayout(self.music2_button_layout)
        self.playbutton_music2 = PlayButton('play.png')
        self.playbutton_music2.setEnabled(False)
        self.music2_button_layout.addWidget(self.playbutton_music2,
                                              alignment=QtCore.Qt.AlignHCenter)
        self.pausebutton_music2 = PlayButton('pause.png')
        self.pausebutton_music2.setEnabled(False)
        self.music2_button_widget.setMaximumWidth(100)
        self.music2_button_layout.addWidget(self.pausebutton_music2,
                                              alignment=QtCore.Qt.AlignHCenter)
        plot_layout.addWidget(self.music2_button_widget, 3, 0,
                              alignment=QtCore.Qt.AlignHCenter)
        # 3
        self.music3_button_widget = QtWidgets.QWidget()
        self.music3_button_layout=QtWidgets.QVBoxLayout()
        self.music3_button_widget.setLayout(self.music3_button_layout)
        self.playbutton_music3 = PlayButton('play.png')
        self.playbutton_music3.setEnabled(False)
        self.music3_button_layout.addWidget(self.playbutton_music3,
                                              alignment=QtCore.Qt.AlignHCenter)
        self.pausebutton_music3 = PlayButton('pause.png')
        self.pausebutton_music3.setEnabled(False)
        self.music3_button_widget.setMaximumWidth(100)
        self.music3_button_layout.addWidget(self.pausebutton_music3,
                                              alignment=QtCore.Qt.AlignHCenter)
        plot_layout.addWidget(self.music3_button_widget, 4, 0,
                              alignment=QtCore.Qt.AlignHCenter)

        # player and buttons for the painting
        k=6
        self.painting_button_widget = QtWidgets.QWidget()
        self.painting_button_layout=QtWidgets.QVBoxLayout()
        self.painting_button_widget.setLayout(self.painting_button_layout)
        self.playbutton_painting = PlayButton('play.png')
        self.playbutton_painting.setEnabled(False)
        self.painting_button_layout.addWidget(self.playbutton_painting,
                                              alignment=QtCore.Qt.AlignHCenter)
        self.pausebutton_painting = PlayButton('pause.png')
        self.pausebutton_painting.setEnabled(False)
        self.painting_button_layout.addWidget(self.pausebutton_painting,
                                              alignment=QtCore.Qt.AlignHCenter)
        self.painting_button_widget.setMaximumWidth(100)
        plot_layout.addWidget(self.painting_button_widget, k, 0,
                              alignment=QtCore.Qt.AlignHCenter)

        # Column 1: signals
        signals_title = QtWidgets.QLabel("Waveforms of the music tracks")
        plot_layout.addWidget(signals_title, 0, 1,
                                              alignment=QtCore.Qt.AlignHCenter)
        for k in range(4):
            music_plot = SignalMplCanvas(width=4, height=1)
            music_plot.setMaximumHeight(100)
            music_plot.setMinimumHeight(100)
            music_plot.setMaximumWidth(470)
            music_plot.setMinimumWidth(470)
            self.signal_widget_list.append(music_plot)
            plot_layout.addWidget(self.signal_widget_list[-1], k+1, 1)

        self.legend_widget = Legend()
        self.legend_widget.setMaximumHeight(100)
        self.legend_widget.setMinimumHeight(100)
        self.legend_widget.setMaximumWidth(470)
        self.legend_widget.setMinimumWidth(470)
        self.signal_widget_list.append(self.legend_widget)
        plot_layout.addWidget(self.signal_widget_list[-1], 5, 1)

        paintmusic_plot = SignalMplCanvas(width=4, height=1)
        paintmusic_plot.setMaximumHeight(100)
        paintmusic_plot.setMinimumHeight(100)
        paintmusic_plot.setMaximumWidth(470)
        paintmusic_plot.setMinimumWidth(470)
        self.signal_widget_list.append(paintmusic_plot)
        plot_layout.addWidget(self.signal_widget_list[-1], 6, 1)

        # Column 2: transforms
        transforms_title = QtWidgets.QLabel("Spectra")
        plot_layout.addWidget(transforms_title, 0, 2,
                                              alignment=QtCore.Qt.AlignHCenter)
        for k in range(6):
            transform_plot = TransformMplCanvas(width=6, height=1)
            transform_plot.setMaximumHeight(100)
            transform_plot.setMinimumHeight(100)
            transform_plot.setMaximumWidth(470)
            transform_plot.setMinimumWidth(470)
            self.dt_widget_list.append(transform_plot)
            plot_layout.addWidget(self.dt_widget_list[-1], k+1, 2)

        # fill the output_layout (Horizontal Box)
        # Column 0: the go/clear buttons widget
        goclear_widget = QtWidgets.QGroupBox()
        goclear_widget.setMaximumWidth(200)
        goclear_widget.setMinimumWidth(200)
        goclear_widget.setMinimumHeight(180)
        goclear_layout = QtWidgets.QVBoxLayout()
        goclear_widget.setLayout(goclear_layout)
        # help go button
        self.helpgobutton = QtWidgets.QLabel("Select the inputs on the left. After pressing the Go button, wait for the elaboration.")
        self.helpgobutton.setFont(QtGui.QFont('Arial', 10))
        self.helpgobutton.setFixedWidth(180)
        self.helpgobutton.setFixedHeight(40)
        self.helpgobutton.setWordWrap(True)
        goclear_layout.addWidget(self.helpgobutton, alignment=QtCore.Qt.AlignHCenter)


        # Go
        self.gobutton = QtWidgets.QPushButton("Go")
        self.gobutton.setFixedSize(QtCore.QSize(100, 30))
        self.gobutton.setEnabled(False)
        goclear_layout.addWidget(self.gobutton, alignment=QtCore.Qt.AlignHCenter)
        self.gobutton.clicked.connect(self.numeric_elaboration)

        # Clear
        self.clearbutton = QtWidgets.QPushButton("clear")
        self.clearbutton.setFixedSize(QtCore.QSize(100, 30))
        self.clearbutton.setEnabled(False)
        self.clearbutton.setStyleSheet('QPushButton')
        goclear_layout.addWidget(self.clearbutton, alignment=QtCore.Qt.AlignHCenter)
        self.clearbutton.clicked.connect(self.clear_all)

        # Help for Clear button
        self.helpclearbutton = QtWidgets.QLabel("To run again, press the clear button and change the inputs")
        self.helpclearbutton.setFont(QtGui.QFont('Arial', 10))
        self.helpclearbutton.setFixedWidth(180)
        self.helpclearbutton.setFixedHeight(30)
        self.helpclearbutton.setWordWrap(True)
        goclear_layout.addWidget(self.helpclearbutton, alignment=QtCore.Qt.AlignHCenter)
        self.helpclearbutton.hide()

        output_layout.addWidget(goclear_widget, alignment=QtCore.Qt.AlignHCenter)
        output_layout.addWidget(goclear_widget, alignment=QtCore.Qt.AlignVCenter)

        # Column 1: pie Chart
        self.pie_widget = PieChart()
        output_layout.addWidget(self.pie_widget.chartview)

        # Column 2: distances
        self.distance_widget = Distance()
        self.distance_widget.setMinimumWidth(350)
        self.distance_widget.setMaximumWidth(350)
        output_layout.addWidget(self.distance_widget)

# end constructor of the main class
#     ############################################################

    def load_image_gui(self, text):
        # update the image name from the ComboBox
        # load the image and save it in self.image.
        self.pixmap = ScaledPixmap(text, self.paintings_dir)
        self.image.setPixmap(self.pixmap.scaled_pixmap)
        self.set_selected_painting(text)
        self.counter_go += 10
        if self.counter_go >= 21 and not self.clearbutton.isEnabled():
            self.gobutton.setEnabled(True)
            self.gobutton.setStyleSheet('QPushButton {background-color: #0066CC; color: white;}'
                                        'QPushButton::pressed {background-color: #FF8800; color: white;}')


    def set_selected_painting(self, selected_painting):
        self.painting_name = selected_painting
        # print("selected painting", self.painting_name)

    def set_selected_music(self):
        # save the list of selected tracks in selected_tracks
        # print("selected tracks")
        # print([item.text() for item in self.music_list.selectedItems()])
        selected_tracks_list = list()
        for item in self.music_list.selectedItems():
            selected_tracks_list.append(item.text())
        self.save_selected_tracks(selected_tracks_list)
        self.counter_go += 1
        if self.counter_go >= 21 and not self.clearbutton.isEnabled():
            self.gobutton.setEnabled(True)
            self.gobutton.setStyleSheet('QPushButton {background-color: #0066CC; color: white;}'
                                        'QPushButton::pressed {background-color: #FF8800; color: white;}')


    def save_selected_tracks(self, selected_tracks):
        n_tracks = len(selected_tracks)
        if n_tracks > 4:
            selected_tracks = selected_tracks[0:4]
            n_tracks = 4
        self.selected_tracks = selected_tracks
        self.n_selected_tracks = n_tracks  # number of selected tracks
        # print("number of selected tracks", self.n_selected_tracks)
        # print("save_selected_tracks: selected tracks", self.selected_tracks)

    def set_selected_transform(self):
        self.set_transform(self.transform_widget.checkedId())
        self.counter_go += 10
        if self.counter_go >= 21 and not self.clearbutton.isEnabled():
            self.gobutton.setEnabled(True)
            self.gobutton.setStyleSheet('QPushButton {background-color: #0066CC; color: white;}'
                                        'QPushButton::pressed {background-color: #FF8800; color: white;}')

        # print("selected transform ", self.transform)

    def set_transform(self, value):
        self.transform = value

    def select_mother_wavelet(self, text):
        self.mother_wavelet = text
        if self.counter_go >= 21 and not self.clearbutton.isEnabled():
            self.gobutton.setEnabled(True)
            self.gobutton.setStyleSheet('QPushButton {background-color: #0066CC; color: white;}'
                                        'QPushButton::clicked {background-color: #FF0000; color: white;}')

        # print("mother wavelet", self.mother_wavelet)

    def select_levels_wavelet(self, levels):
        self.wave_nlevels = levels
        if self.counter_go >= 21 and not self.clearbutton.isEnabled():
            self.gobutton.setEnabled(True)
            self.gobutton.setStyleSheet('QPushButton {background-color: #0066CC; color: white;}'
                                        'QPushButton::pressed {background-color: #FF8800; color: white;}')

       # print("number of levels for wavelet=", self.wave_nlevels)


    def clear_all(self):
        self.clearbutton.setEnabled(False)
        self.clearbutton.setDefault(False)
        self.clearbutton.setStyleSheet('QPushButton')
        self.pie_widget.clear_pie()
        self.distance_widget.clear_distances()
        self.legend_widget.clear_legend()
        for k in range(6):
            if k != 4:
                self.signal_widget_list[k].clear_plot()

        for k in range(6):
            self.dt_widget_list[k].clear_plot_dt()

        self.playbutton_music0.setEnabled(False)
        self.playbutton_music1.setEnabled(False)
        self.playbutton_music2.setEnabled(False)
        self.playbutton_music3.setEnabled(False)
        self.playbutton_painting.setEnabled(False)

        self.pausebutton_music0.setEnabled(False)
        self.pausebutton_music1.setEnabled(False)
        self.pausebutton_music2.setEnabled(False)
        self.pausebutton_music3.setEnabled(False)
        self.pausebutton_painting.setEnabled(False)

        self.helpclearbutton.hide()
        self.helpgobutton.setText("Select the inputs on the left. After pressing the Go button, wait for the elaboration.")

    def clean_gobutton(self):
        # deactivate the go button
        self.gobutton.setEnabled(False)
        self.gobutton.setStyleSheet('QPushButton')
        # self.helpgobutton.show()

# the kernel of the app
    def numeric_elaboration(self):

        # fill the legend
        self.legend_widget.fill_legend(self.selected_tracks, self.color_tracks)

        # read and transform the image
        self.image_elaboration()

        # read the audio signals, plot and transform
        ma = []
        for item_index, item in enumerate(self.selected_tracks):
            # extract the name of the music
            music_name = os.path.splitext(item)
            music_name=music_name[0]
            music_name1=music_name.replace("-"," ")
            # read
            audio_signal, sample_rate = librosa.load(self.music_dir + item,
                                                    sr=None, mono=False)
            self.sample_rate.append(sample_rate)

            # if the music track has more than one trace, select the first trace
            # cut the audio signal to the pixels number
            if audio_signal.shape.__len__() > 1:
                audio_signal = (audio_signal[0] + audio_signal[1]) / 2
            audio_length = len(audio_signal)

            if audio_length < self.n_pixels:
                # print a message and stop the execution if the music track is too short.
                print("WARNING: The music track " + item)
                print("is too short compared with the dimension of the image")
                print("number of image pixels: ", self.n_pixels)
                print("number of music-track samples: ", audio_length)
                print("The music-track will be replicated for the computation")
                newaudio_length = audio_length
                newaudio_signal = audio_signal
                while newaudio_length < self.n_pixels:
                    newaudio_signal = np.concatenate((newaudio_signal, audio_signal),
                                                  axis=None)
                    newaudio_length += audio_length
                audio_signal = newaudio_signal
                del newaudio_signal, newaudio_length
            if audio_length> self.n_pixels:
                audio_signal = audio_signal[0:self.n_pixels]
            # normalize
            audio_signal = audio_signal / np.linalg.norm(audio_signal, np.inf)
            # plot the signal
            self.signal_widget_list[item_index].my_plot(audio_signal,
                                                    self.sample_rate[item_index],
                                                    music_name1,
                                                    self.color_tracks[item_index])
            self.signal_widget_list[item_index].draw()

            # print("audio_signal.shape", audio_signal.shape, "len_data", len_data)
            # compute (and normalize) the transform of the audio signal
            coeffs_audio, len_coeffs_audio = self.transform_audio(audio_signal, self.n_pixels)
            # save coeffs_audio into the matrix
            ma.append(coeffs_audio)
            # align 2d-dwt coefficients of the image
            self.dt_widget_list[item_index].axes.grid(True)

            # plot the spectrum

            if self.transform <= 1:
                self.dt_widget_list[item_index].my_plot_dwt(coeffs_audio, music_name1,
                                                        self.color_tracks[item_index])
            else:
                self.dt_widget_list[item_index].my_plot_dft(coeffs_audio, music_name1,
                                                        self.color_tracks[item_index],
                                                        self.sample_rate[item_index])
            self.dt_widget_list[item_index].draw()


        # align 2d-dwt coefficients of the image
        if self.transform == 1:
            self.coeffs_image = self.align_dwt2_to_dwt1(coeffs_audio,
                                                   len_coeffs_audio,
                                                   self.coeffs_image,
                                                   self.len_coeffs_image)
        elif self.transform == 3:
            self.coeffs_image = self.coeffs_image.flatten()

        # plot the transform of the original image
        if self.transform <= 1:
            self.dt_widget_list[4].my_plot_dwt(self.coeffs_image,
                                               "painting",
                                               self.color_painting[0])
        else:
            self.dt_widget_list[4].my_plot_dft(self.coeffs_image,
                                               "painting",
                                               self.color_painting[0],
                                               self.my_sample_rate)
        self.dt_widget_list[4].draw()

        # solve the least square problem
        nmatrix = np.sum(len_coeffs_audio)

        matrix = np.zeros([nmatrix, self.n_selected_tracks], dtype=complex)
        for matrix_column, item in enumerate(ma):
            matrix[:, matrix_column] = np.array(item, dtype=complex)
            # matrix_column += 1
        del ma
        matrix = np.matrix(matrix)
        a = np.matmul(matrix.H, matrix)
        # align the size of coeffs_audio and coeffs_image (to the first one)
        cis = self.coeffs_image.size
        cas = coeffs_audio.size
        if cis < cas:
            self.coeffs_image = np.r_[self.coeffs_image, np.zeros([1, cas-cis])]
        elif cis > cas:
            self.coeffs_image = self.coeffs_image[0:cas]
        c = np.matrix(self.coeffs_image)

        b = np.matmul(matrix.H, c.T)
        # alpha = a \ b
        self.alpha = np.linalg.solve(a, b)
        # print("alpha",self.alpha[0:4])

        # reconstruct the painting signal
        coeffs_projection = np.matmul(matrix, self.alpha)
        del matrix

        coeffs_projection = np.array(coeffs_projection).flatten()
        coeffs_projection_real = coeffs_projection.real
        if self.transform <= 1:
            self.dt_widget_list[5].my_plot_dwt(coeffs_projection_real,
                                               "new piece of music",
                                               self.color_painting[1])
        else:
            self.dt_widget_list[5].my_plot_dft(coeffs_projection_real,
                                               "new piece of music",
                                               self.color_painting[1],
                                               self.my_sample_rate)
        self.dt_widget_list[5].draw()

        # build the music track of the painting
        painting_signal = self.reconstruct_audio_signal(coeffs_projection,
                                                        len_coeffs_audio)
        # plot the signal of the image
        self.signal_widget_list[5].my_plot(painting_signal, self.my_sample_rate,
                                           "new piece of music",
                                           self.color_painting[1])
        self.signal_widget_list[5].draw()

        # save the trace of the new piece of music
        soundfile.write("sound1.wav", painting_signal, self.my_sample_rate, format='WAV')

        # connect the new piece of music to its play button
        self.player_painting = Player(CURRENT_DIR, "sound1.wav")
        self.playbutton_painting.setEnabled(True)
        self.playbutton_painting.clicked.connect(lambda: self.click_playbutton(
            self.player_painting))
        self.pausebutton_painting.setEnabled(True)
        self.pausebutton_painting.clicked.connect(lambda: self.click_pausebutton(
            self.player_painting))

        # connect the music track to its play button
        if self.n_selected_tracks >0:
            self.player_music0 = Player(self.music_dir, self.selected_tracks[0])
            self.playbutton_music0.setEnabled(True)
            self.playbutton_music0.clicked.connect(
                lambda: self.click_playbutton(self.player_music0))
            self.pausebutton_music0.setEnabled(True)
            self.pausebutton_music0.clicked.connect(lambda: self.click_pausebutton(
                self.player_music0))
        if self.n_selected_tracks > 1:
            self.player_music1 = Player(self.music_dir, self.selected_tracks[1])
            self.playbutton_music1.setEnabled(True)
            self.playbutton_music1.clicked.connect(
                lambda: self.click_playbutton(self.player_music1))
            self.pausebutton_music1.setEnabled(True)
            self.pausebutton_music1.clicked.connect(lambda: self.click_pausebutton(
                self.player_music1))
        if self.n_selected_tracks > 2:
            self.player_music2 = Player(self.music_dir, self.selected_tracks[2])
            self.playbutton_music2.setEnabled(True)
            self.playbutton_music2.clicked.connect(
                lambda: self.click_playbutton(self.player_music2))
            self.pausebutton_music2.setEnabled(True)
            self.pausebutton_music2.clicked.connect(lambda: self.click_pausebutton(
                self.player_music2))
        if self.n_selected_tracks > 3:
            self.player_music3 = Player(self.music_dir, self.selected_tracks[3])
            self.playbutton_music3.setEnabled(True)
            self.playbutton_music3.clicked.connect(
                lambda: self.click_playbutton(self.player_music3))
            self.pausebutton_music3.setEnabled(True)
            self.pausebutton_music3.clicked.connect(lambda: self.click_pausebutton(
                self.player_music3))


        # plot the piechart
        self.alpha = abs(self.alpha)
        self.alpha_percento =self.alpha/np.sum(self.alpha)
        self.pie_widget.fill_pie(self.alpha_percento, self.selected_tracks, self.color_tracks)

        # compute the distance between the normalized spectrum of the image and
        #  normalized spectrum of the projection
        painting_spectrum_norm = np.linalg.norm(self.coeffs_image)
        projection_spectrum_norm = np.linalg.norm(coeffs_projection)
        normalized_distance =  np.linalg.norm(self.coeffs_image/painting_spectrum_norm-
                                             coeffs_projection/projection_spectrum_norm)
        self.distance_widget.fill_distances(normalized_distance)

        # deactivate the go button
        self.gobutton.setEnabled(False)
        self.gobutton.setStyleSheet('QPushButton')

        # activate the clear button
        self.clearbutton.setEnabled(True)
        self.clearbutton.setStyleSheet('QPushButton {background-color: #0066CC; color: white;}'
                                        'QPushButton::pressed {background-color: #FF8800; color: white;}')

        self.helpgobutton.setText("Click on the play buttons to listen to the sounds.")
        self.helpclearbutton.show()


    def image_elaboration(self):
        # read yhe image
        data = PIL.Image.open(self.paintings_dir + self.painting_name + ".png")
        # save the image intensity
        image = np.array(data, dtype=np.double)
        if image.shape[2] == 2:
            self.image_intensity = image[:, :, 0]
        elif image.shape[2] == 4:
            self.image_intensity = (image[:, :, 0] + image[:, :, 1] + image[:, :, 2]) / 3

        self.n_pixels = self.image_intensity.size

        # compute the discrete transform of the image
        self.coeffs_image, self.len_coeffs_image = self.transform_image(self.image_intensity)
        #print("number of pixels", self.n_pixels)

    def transform_image(self, image_intensity):
        # transform the image

        if self.transform == 0:
            #  2d --> 1d --> DWT
            x = image_intensity.flatten()
            c = pywt.wavedec(x, wavelet=self.mother_wavelet,
                             level=self.wave_nlevels)
            # coeffs is a list
            len_coeffs = []
            coeffs = []
            for item in c:
                v = np.array(item)  # row vector
                len_coeffs.append(v.size)
                coeffs = np.r_[coeffs, v]  # concatenatepre-norm
        elif self.transform == 1:
            # 2d --> DWT --> 1d
            c = pywt.wavedec2(image_intensity,
                              wavelet=self.mother_wavelet,
                              level=self.wave_nlevels)
            len_coeffs = []
            coeffs = []
            i = 1
            for item in c:
                if i == 1:
                    v = np.array(item)  # matrix
                    v = v.flatten()
                    len_coeffs.append(v.size)
                    coeffs = np.r_[coeffs, v]  # concatenate
                else:
                    for subitem in item:
                        v = np.array(subitem)  # matrix
                        v = v.flatten()
                        len_coeffs.append(v.size)
                        coeffs = np.r_[coeffs, v]  # concatenate
                i += 1
        elif self.transform == 2:
            #   2d --> 1d -->DFT
            x = image_intensity.T.flatten()
            coeffs = fft.fft(x)
            len_coeffs = coeffs.size
        elif self.transform == 3:
            #   2d --> DFT --> 1d
            coeffs = fft.fft2(image_intensity)
            coeffs.flatten()
            len_coeffs = coeffs.size

        return coeffs, len_coeffs

    def transform_audio(self, data, n_pixels):
        # print('dtw_coeff.shape ', dtw_coeff.shape)
        # pad the array with zero values
        len_data = data.size
        if n_pixels < len_data:
            data = data[0:n_pixels]
        elif n_pixels > len_data:
            data = np.r_[data, np.zeros(n_pixels-len_data)]

        if self.transform <= 1:
            c = pywt.wavedec(data, wavelet=self.mother_wavelet,
                             level=self.wave_nlevels)
            # coeffs is a list
            len_coeffs = []
            coeffs = []
            for item in c:
                v = np.array(item)  # row vector
                len_coeffs.append(v.size)
                coeffs = np.r_[coeffs, v]  # concatenate
        else:
            coeffs = fft.fft(data)
            len_coeffs = coeffs.size
        return coeffs, len_coeffs

    def align_dwt2_to_dwt1(self, coeffs_audio, len_coeffs_audio,
                           coeffs_image, len_coeffs_image):
        n = coeffs_audio.size
        coeffs1 = np.zeros(n)
        nci = len_coeffs_image[0]
        nca = len_coeffs_audio[0]
        nrep = nca // nci
        krep = 0
        coeffs1[krep*nci:krep*nci+nci] = coeffs_image[0:nci]
        for krep in range(1,nrep):
            # coeffs1[krep*nci:krep*nci+nci] = coeffs_image[0:nci]
            coeffs1[krep*nci:krep*nci+nci] = np.zeros([nci])
        na = nca
        ni = nci
        for l1 in range(self.wave_nlevels):
            nci = len_coeffs_image[1+l1*3]
            nca = len_coeffs_audio[1+l1]
            nrep = nca // (nci*3)
            hvd = coeffs_image[ni:ni+nci*3]
            if nrep > 0:
                krep = 0
                coeffs1[na+krep*nci*3:na+krep*nci*3+nci*3] = hvd
                for krep in range(1, nrep):
                    # coeffs1[na+krep*nci*3:na+krep*nci*3+nci*3] = hvd
                    coeffs1[na+krep*nci*3:na+krep*nci*3+nci*3] = np.zeros([len(hvd)])
            else:
                coeffs1[na:na+nca] = hvd[0:nca]
            na = na + nca
            ni = ni + nci*3
        return coeffs1

    def reconstruct_audio_signal(self, coeffs_projection, len_coeffs_audio):
        if self.transform <= 1:
            coeffs = []
            index = 0
            for item in len_coeffs_audio:
                v = np.array(coeffs_projection[index:index+item])
                coeffs.append(v)
                index = index+item
            x = pywt.waverec(coeffs, self.mother_wavelet)
        else:
            x = fft.ifft(coeffs_projection)
        signal = x.real / np.linalg.norm(x.real, np.inf)
        signal = signal.flatten()
        return signal


    def click_playbutton(self, player):
            player.play()

    def click_pausebutton(self,player):
            player.pause()

    def change_icon_playbutton(self, playbutton, icon_name):
            icon = QtGui.QIcon()
            icon.addPixmap(QtGui.QPixmap(icon_name))
            playbutton.setIcon(icon)
            playbutton.setIconSize(QtCore.QSize(30, 30))


def generate_small_images(paintings_dir, paintings_list_filename):

# read the list of filenames
    with open(paintings_list_filename, "r") as input_file:
        menu_csv = list(csv.reader(input_file, delimiter="\n"))
    menu_csv1 = []
    for sublist in menu_csv:
        for item in sublist:
            menu_csv1.append(item)

# create the directory "./_small"
    paintings_dir_small = "./_small/"
    Path(paintings_dir_small).mkdir(parents=True, exist_ok=True)

#
    for img in menu_csv1[1:]:
        if not exists(img+"_small.png"):
            im1 = Image.open(paintings_dir + img + ".png")
            if im1.height >= im1.width:
                new_h = 200
                new_w = int(im1.width *200/im1.height)
            else:
                new_w = 200
                new_h = int(im1.height * 200 / im1.width)
            im1 = im1.resize((new_w, new_h), Image.ANTIALIAS)
            im1.save(paintings_dir_small + img + "_small.png", optimize = True, quality = 95)


app = QtWidgets.QApplication(sys.argv)
# directory where the audio-files are stored (absolute path)
music_dir = "/home/gerva/Music/"
# csv file with the list of mp3 files of the music tracks
music_list_filename = "musictracks.csv"
# directory where the images are stored  (absolute or relative path)
paintings_dir = "../Paintings/"
# csv file with the list of images (without extension).
# The first line of the file must be blank or must contain any other string, like e.g. '----'
paintings_list_filename = "paintings.csv"

# generate small_size images
generate_small_images(paintings_dir, paintings_list_filename)

warnings.filterwarnings('ignore')
window = MainWindow(music_dir, music_list_filename,
                    paintings_dir, paintings_list_filename)
window.show()
app.exec_()
