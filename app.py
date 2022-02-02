import os
import sys
import warnings
import csv
import numpy as np

from scipy import fft
import pywt  # pip install PyWavelets
import pywt.data
import librosa
import soundfile
import PIL  # pip install Pillow
from PIL import Image

from PySide2 import QtGui, QtWidgets, QtCharts, QtCore, QtMultimedia
from qtwidgets import Toggle

# import PySide2 before matplotlib
import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
matplotlib.use("Qt5Agg")

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


class PaintingListComboBox(QtWidgets.QComboBox):
    # A ComboBox to read the names of the paintings from the file paintings_list_filename
    def __init__(self, paintings_list_filename):
        super().__init__()
        # self.paintings_list_filename = paintings_list_filename
        with open(paintings_list_filename,  "r") as input_file:
            menu_csv = list(csv.reader(input_file, delimiter="\n"))
        menu_csv1 = []
        for sublist in menu_csv:
            for item in sublist:
                menu_csv1.append(item)
        self.addItems(menu_csv1)


class ScaledPixmap(QtGui.QPixmap):
    # definisco una pixmap e la scalo,
    # salvando la sua scalatura in self.scaled_pixmap
    def __init__(self, filename, directory):
        super().__init__()
        self.load(directory + filename)
        self.scaled_pixmap = self.scale()
        # print("filename", filename)
        # print("directory",directory)
        # print("directory+filename",directory+filename)

    def scale(self):
        # method to scale the pixmap
        pw = min(self.width(), 200)  # minima dimensione dell'immagine
        ph = min(self.height(), 200)  # minima dimensione dell'immagine
        # l'immagine scalata deve essere salvata in una nuova variabile
        # print("pw=",pw, " ph=",ph)
        scaled_pixmap = " "
        if pw > 0 and ph > 0:
            scaled_pixmap = self.scaled(pw, ph, QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation)
        return scaled_pixmap


class MusicListWidget(QtWidgets.QListWidget):
    # A Widget to read the list of musics
    def __init__(self, musics_list_filename):
        super().__init__()
        self.setSelectionMode(QtWidgets.QListWidget.ExtendedSelection)
        self.resize(min(self.width(), 160), min(self.width(), 160))
        with open(musics_list_filename,  "r") as input_file:
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
        self.w = QtWidgets.QWidget()  # definisco il widget che andra' nel self.layout_left4
        layout = QtWidgets.QVBoxLayout()  # definisco il layout del widget w
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
    # the second one is smaller
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
    def __init__(self, width=5, height=1):
        # width and height in inches
        fig = Figure(figsize=(width, height))
        self.axes = fig.add_subplot(111)
        super().__init__(fig)
        self.axes.grid(True)
        self.axes.tick_params(axis='both', which='major', labelsize=8)
        # Add Axis Labels
        self.axes.set_xlabel("t (sec)", fontsize=8, loc='right')
        self.axes.set_ylabel("Signal", fontsize=8)
        fig.tight_layout(pad=0.02)

    def my_plot(self, data, samplerate, plotname, color):
        data = data.real / np.linalg.norm(data.real, np.inf)
        # samplerate, data = wavfile.read(filename)
        length = data.shape[0] / samplerate
        time = np.linspace(0., length, data.shape[0])
        self.axes.plot(time, data, color=color, linewidth=1)
        self.axes.set_xlim(time[0], time[-1])
        self.axes.set_ylim(-1, 1)
        self.axes.annotate(plotname, xy=(10, 5), xycoords='figure pixels', fontsize=8)

    def clear_plot(self):
        self.axes.clear()
        self.axes.grid(True)
        self.axes.tick_params(axis='both', which='major', labelsize=8)
        # Add Axis Labels
        self.axes.set_xlabel("t (sec)", fontsize=8, loc='right')
        self.axes.set_ylabel("Signal", fontsize=8)
        self.draw()


class TransformMplCanvas(FigureCanvasQTAgg):
    # build a PlotWidget to plot the signal read from file
    def __init__(self, width=5, height=1, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)
        self.axes.tick_params(axis='both', which='major', labelsize=8)
        # Add Axis Label
        self.axes.set_xlabel("k (Hz)", fontsize=8, loc='right')
        self.axes.set_ylabel("Frequencies", fontsize=8)
        self.axes.semilogx([1], [1])
        self.axes.grid(True)
        fig.tight_layout(pad=0.1)

    def my_plot_dwt(self, y, plotname, color):
        x = np.linspace(0, y.size, y.size)
        self.axes.semilogx(x, y, color=color, linewidth=1.0)
        # Set Range
        self.axes.set_xlim(1, x.size)
        self.axes.set_ylim(-1, 1)
        self.axes.set_xlabel("k", fontsize=8, loc='right')
        self.axes.annotate(plotname, xy=(10, 5), xycoords='figure pixels', fontsize=8)

    def my_plot_dft(self, y, plotname, color, sample_rate):
        n = y.size
        x = fft.fftfreq(n, 1./sample_rate)[0:n//2]
        yy = np.abs(y[1:n//2])
        self.axes.semilogx(x[1:], yy, color=color, linewidth=1.0)
        # Set Range
        self.axes.set_xlim(1, x[-1])
        self.axes.set_ylim(0, np.amax(yy))
        self.axes.annotate(plotname, xy=(10, 5), xycoords='figure pixels', fontsize=8)

    def clear_plot_dt(self):
        self.axes.clear()
        self.axes.tick_params(axis='both', which='major', labelsize=8)
        # Add Axis Labels
        self.axes.set_xlabel("k", fontsize=8, loc='right')
        self.axes.set_ylabel("Frequencies", fontsize=8)
        self.axes.semilogx([1], [1])
        self.axes.grid(True)
        self.draw()


class PieChart(QtWidgets.QLabel):
    # PieChart is a derived class of QLabel, in which I add a pixmap
    def __init__(self):
        super().__init__()
        # cretaa pixmap with white background
        canvas = QtGui.QPixmap()
        canvas.fill(QtCore.Qt.white)
        # save the pixmap in the Pixmap attribute of the Widget QLabel
        self.setPixmap(canvas)
        # define an object of class QPieSeries with the data
        self.series = QtCharts.QtCharts.QPieSeries()
        # put the labels outside the pie
        self.series.setLabelsPosition(QtCharts.QtCharts.QPieSlice.LabelOutside)
        # defined the  Graphic Widget  QChart
        self.chart = QtCharts.QtCharts.QChart()
        # add the data for the pie
        self.chart.addSeries(self.series)
        # generate the pie by the animation
        self.chart.setAnimationOptions(QtCharts.QtCharts.QChart.SeriesAnimations)
        # align the legend on the right
        self.chart.legend().setAlignment(QtCore.Qt.AlignRight)
        # sav =e the graphic widget in chartview
        self.chartview = QtCharts.QtCharts.QChartView(self.chart)
        self.chartview.setAlignment(QtCore.Qt.AlignTop)
        self.chart.layout().setContentsMargins(2, 2, 2, 2)
        
    def fill_pie(self, data, labels, colors):
        # append data and labels
        for s in range(data.shape[0]):
            _slice = QtCharts.QtCharts.QPieSlice(labels[s], data[s])
            _slice.setBrush(QtGui.QBrush(QtGui.QColor(colors[s])))
            self.series.append(_slice)
        self.series.setPieSize(10)
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

    def fill_legend(self, musics, colors):
        painter = QtGui.QPainter(self.pixmap())
        pen = QtGui.QPen()
        pen.setWidth(8)
        for k in range(len(musics)):
            pen.setColor(QtGui.QColor(colors[k]))
            painter.setPen(pen)
            painter.drawPoint(30, 20+k*20)
            pen.setColor(QtGui.QColor('black'))
            painter.setPen(pen)
            painter.drawText(50, 25+k*20, musics[k])
        painter.end()
        self.update()

    def clear_legend(self):
        canvas = QtGui.QPixmap(500, 100)
        canvas.fill(QtCore.Qt.white)
        self.setPixmap(canvas)


class Errors(QtWidgets.QLabel):
    def __init__(self):
        super().__init__()
        canvas = QtGui.QPixmap(350, 180)
        canvas.fill(QtCore.Qt.white)
        self.setPixmap(canvas)

    def fill_errors(self, l2err, l2err_rel):
        painter = QtGui.QPainter(self.pixmap())
        pen = QtGui.QPen()
        pen.setColor(QtGui.QColor('black'))
        painter.setPen(pen)

        painter.drawText(25, 60, ("absolute L2-error = " + str("{:.2f}".format(l2err))))
        painter.drawText(25, 100, ("relative L2-error = " + str("{:.2f}".format(l2err_rel))))
        painter.end()
        self.update()

    def clear_errors(self):
        canvas = QtGui.QPixmap(350, 180)
        canvas.fill(QtCore.Qt.white)
        self.setPixmap(canvas)


class Player(QtMultimedia.QMediaPlayer):
    def __init__(self, directory, music):
        super().__init__()
        filename = os.path.join(directory, music)
        url = QtCore.QUrl.fromLocalFile(filename)
        self.setMedia(QtMultimedia.QMediaContent(url))

    def my_play(self):
        self.play()
        if self.state() == QtMultimedia.QMediaPlayer.PlayingState:
            self.stop()



class PlayerToggle(Toggle):
    def __init__(self):
        super().__init__(checked_color="#FFB000")


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, musics_dir, paintings_dir):
        super().__init__()
        self.setWindowTitle("Playing paintings")
        self.setGeometry(100, 100, 1300, 800)

        self.paintings_dir = paintings_dir
        self.musics_dir = musics_dir
        self.selected_musics = list()
        self.n_selected_musics = 0
        self.painting_name = " "
        self.mother_wavelet = "db5"
        self.wave_nlevels = 7
        self.transform = 0

        self.color_musics = ["#99ccff", "#3399ff", "#0066cc", "#003366"]
        self.color_painting = ["#ff0000", "#ff8000"]

        self.sample_rate = 44100
        self.counter_go = 0
        self.toggle_musics_list = list()
        self.player_musics_list = list()
        self.player_painting = Player(CURRENT_DIR, "sound.mp3")
        self.toggle_painting = PlayerToggle()
        self.toggle_painting.statusTip()
        self.toggle_painting.pressed.connect(self.player_painting.my_play)

        self.alpha = np.zeros([4])
        self.alpha_percento = np.zeros([4])
        # local variables
        painting_name = ""

        ############################################################
        # global layout
        layout = QtWidgets.QHBoxLayout()

        # left layout
        widget_left = QtWidgets.QGroupBox()
        layout_left = QtWidgets.QVBoxLayout()
        layout_left.setContentsMargins(0, 0, 0, 0)
        widget_left.setLayout(layout_left)
        widget_left.setMaximumWidth(250)

        # right layout
        widget_right = QtWidgets.QGroupBox()
        layout_right = QtWidgets.QVBoxLayout()
        layout_right.setContentsMargins(0, 0, 0, 0)
        widget_right.setLayout(layout_right)
        widget_right.setMinimumWidth(1050)

        # put the left and right widgets inside the global layout
        layout.addWidget(widget_left)
        layout.addWidget(widget_right)

        # the global cointainer  of the main window
        container = QtWidgets.QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
# ##########################################################
#       # left layout: input
# ##########################################################
        widget_left_list = []
        layout_left_list = []
# ############################################################
#       # Select the painting
        lay = QtWidgets.QVBoxLayout()
        label = QtWidgets.QLabel("Step 1. Choose the painting")
# read the list of the paintings from a file
        menu = PaintingListComboBox("paintings.csv")
        lay.addWidget(menu)
        widget_left_list.append(label)
        widget_left_list.append(menu)
        layout_left_list.append(lay)
# create the object for the scaled image (empty image)
        self.pixmap = ScaledPixmap(painting_name, self.musics_dir)
        # send the signal to the slot self.load_image_gui to show the image in the gui
        menu.currentTextChanged.connect(self.load_image_gui)
        self.selected_painting = painting_name
# define a widget Qlabel to load the scaled image
        self.image = QtWidgets.QLabel(self)
        self.image.setAlignment(QtCore.Qt.AlignHCenter)
#  define the layout and store the widget
        lay = QtWidgets.QHBoxLayout()
        lay.setContentsMargins(10, 10, 10, 10)
        self.image.setLayout(lay)
        widget_left_list.append(self.image)
        layout_left_list.append(lay)
# ############################################################
# Choice of the music tracks
        label = QtWidgets.QLabel("Step 2. Choose up to 4 tracks")
        self.musics_list = MusicListWidget("musics.csv")
        self.musics_list.itemSelectionChanged.connect(self.set_selected_musics)

# define the layout and store the widget
        lay = QtWidgets.QVBoxLayout()
        self.musics_list.setLayout(lay)
        widget_left_list.append(label)
        widget_left_list.append(self.musics_list)
        layout_left_list.append(lay)
# ############################################################
# Choice of the discrete transform
        wid = QtWidgets.QGroupBox()
        label = QtWidgets.QLabel("Step 3. Select the transform")
# define the layout and store the widget
        self.transform_widget = TransformButtonGroup(self)
        lay = QtWidgets.QVBoxLayout()
        wid.setLayout(lay)
        wid.setMaximumHeight(160)
        lay.addWidget(label)
        lay.addWidget(self.transform_widget.w)
        widget_left_list.append(wid)
        layout_left_list.append(lay)
# ############################################################
# Choice of the mother wavelet
        wid = QtWidgets.QGroupBox()
        lay = QtWidgets.QHBoxLayout()
        wid.setLayout(lay)
        wid.setMaximumHeight(80)
        lay.setContentsMargins(10, 10, 10, 10)
        self.wavelet = WaveletComboBox()
        label = WaveletLabel("Mother wavelet", "(only for DWT)")
        label.setMaximumWidth(140)
        default = self.mother_wavelet
        self.wavelet.setCurrentText(default)
        self.wavelet.currentTextChanged.connect(self.select_wavelet)
        lay.addWidget(label)
        lay.addWidget(self.wavelet)
        widget_left_list.append(wid)
        layout_left_list.append(lay)

# ############################################################
#   Set the number of levels for DWT
        wid = QtWidgets.QGroupBox()
        lay = QtWidgets.QHBoxLayout()
        wid.setLayout(lay)
        label = WaveletLabel("Number of levels", "(only for DWT)")
        wid.setMaximumHeight(80)
        lay.setContentsMargins(1, 1, 1, 1)
        self.wavelet_levels = QtWidgets.QSpinBox()
        self.wavelet_levels.setMinimum(2)
        self.wavelet_levels.setMaximum(15)
        nlevels = self.wave_nlevels
        self.wavelet_levels.setValue(nlevels)
        self.wavelet_levels.valueChanged.connect(self.select_wavelet_level)
        lay.addWidget(label)
        lay.addWidget(self.wavelet_levels)
        widget_left_list.append(wid)
        layout_left_list.append(lay)

# ############################################################
        # left layout (input)
        for item in widget_left_list:
            layout_left.addWidget(item)

# ############################################################
#        # right layout
# ############################################################
#
        widget_grid = QtWidgets.QGroupBox()
        self.layout_grid = QtWidgets.QGridLayout()
        self.layout_grid.setContentsMargins(0, 0, 0, 0)

        widget_grid.setLayout(self.layout_grid)

        widget_output = QtWidgets.QGroupBox()
        widget_output.setMinimumHeight(180)
        widget_output.setMaximumHeight(180)
        self.layout_output = QtWidgets.QHBoxLayout()
        self.layout_output.setContentsMargins(0, 0, 0, 0)
        widget_output.setLayout(self.layout_output)

        layout_right.addWidget(widget_grid)
        layout_right.addWidget(widget_output)

########################################################
        # layout_grid
        self.signal_widget_list = []
        self.toggle_audio_list = []
        self.dt_widget_list = []
        
        # button for playing sounds
        for k in range(4):
            wid = PlayerToggle()
            wid.setMaximumWidth(60)
            wid.setMinimumWidth(60)
            wid.setEnabled(False)
            self.toggle_audio_list.append(wid)
            self.layout_grid.addWidget(self.toggle_audio_list[-1], k, 0)

        k = 5
        self.toggle_painting = PlayerToggle()
        self.toggle_painting.setEnabled(False)
        self.toggle_painting.statusTip()
        # self.toggle_painting.pressed.connect(self.player_painting.my_play)
        self.toggle_painting.setMaximumWidth(60)
        self.toggle_painting.setMinimumWidth(60)
        self.layout_grid.addWidget(self.toggle_painting, k, 0)

        # signals
        for k in range(4):
            wid = SignalMplCanvas(width=5, height=1)
            wid.setMaximumHeight(100)
            wid.setMinimumHeight(100)
            self.signal_widget_list.append(wid)
            self.layout_grid.addWidget(self.signal_widget_list[-1], k, 1)

        self.legend_widget = Legend()
        self.legend_widget.setMaximumHeight(100)
        self.legend_widget.setMinimumHeight(100)
        self.signal_widget_list.append(self.legend_widget)
        self.layout_grid.addWidget(self.signal_widget_list[-1], 4, 1)

        wid = SignalMplCanvas()
        wid.setMaximumHeight(100)
        wid.setMinimumHeight(100)
        self.signal_widget_list.append(wid)
        self.layout_grid.addWidget(self.signal_widget_list[-1], 5, 1)

        # transforms
        for k in range(6):
            wid = TransformMplCanvas(width=5, height=1)
            wid.setMaximumHeight(100)
            wid.setMinimumHeight(100)
            self.dt_widget_list.append(wid)
            self.layout_grid.addWidget(self.dt_widget_list[-1], k, 2)

# layout_output
        wid = QtWidgets.QLabel()
        wid.setMaximumWidth(200)
        wid.setMinimumWidth(200)
        wid.setMinimumHeight(100)
        lay = QtWidgets.QVBoxLayout()
        wid.setLayout(lay)
        # Go
        self.gobutton = QtWidgets.QPushButton("Go")
        self.gobutton.setFixedSize(QtCore.QSize(100, 30))
        self.gobutton.setEnabled(False)
        lay.addWidget(self.gobutton, alignment=QtCore.Qt.AlignHCenter)
        self.gobutton.pressed.connect(self.numeric_elaboration)
        # Clear
        self.clearbutton = QtWidgets.QPushButton("clear")
        self.clearbutton.setFixedSize(QtCore.QSize(100, 30))
        self.clearbutton.setEnabled(False)
        self.clearbutton.setStyleSheet('QPushButton')
        lay.addWidget(self.clearbutton, alignment=QtCore.Qt.AlignHCenter)
        self.clearbutton.pressed.connect(self.clear_all)

        self.layout_output.addWidget(wid, alignment=QtCore.Qt.AlignHCenter)
        self.layout_output.addWidget(wid, alignment=QtCore.Qt.AlignVCenter)

# pie Chart
        pie = PieChart()
        self.pie_widget = pie
        self.layout_output.addWidget(self.pie_widget.chartview)

# errors
        err = Errors()
        err.setMinimumWidth(350)
        err.setMaximumWidth(350)
        self.err_widget = err
        self.layout_output.addWidget(self.err_widget)


# end constructor of the main class
#     ############################################################

    def load_image_gui(self, text):
        # update the image name from the ComboBox
        # load the image and save it in self.image.
        self.pixmap = ScaledPixmap(text, self.paintings_dir)
        self.image.setPixmap(self.pixmap.scaled_pixmap)
        self.set_selected_painting(text)
        self.counter_go += 10
        if self.counter_go >= 21:
            self.gobutton.setEnabled(True)
            self.gobutton.setStyleSheet('QPushButton {background-color: #0066CC; color: white;}')

    def set_selected_painting(self, selected_painting):
        self.painting_name = selected_painting
        # print("selected painting", self.painting_name)

    def set_selected_musics(self):
        # save the list of selected musics in selected_musics
        # print("selected musics")
        # print([item.text() for item in self.musics_list.selectedItems()])
        selected_musics_list = list()
        for item in self.musics_list.selectedItems():
            selected_musics_list.append(item.text())
        self.save_selected_musics(selected_musics_list)
        self.counter_go += 1
        if self.counter_go >= 21:
            self.gobutton.setEnabled(True)
            self.gobutton.setStyleSheet('QPushButton {background-color: #0066CC; color: white;}')
        widget = 0
        for item in self.selected_musics:
            player = Player(self.musics_dir, item)
            self.player_musics_list.append(player)
            # connect the toggle with the music
            self.toggle_audio_list[widget].setEnabled(True)
            self.toggle_audio_list[widget].statusTip()
            self.toggle_audio_list[widget].pressed.connect(self.player_musics_list[-1].my_play)
            widget += 1

    def save_selected_musics(self, selected_musics):
        if len(selected_musics) > 4:
            selected_musics = selected_musics[0:4]
        self.selected_musics = selected_musics
        self.n_selected_musics = len(selected_musics)  # number of selected musics
        # print("number of selected musics", self.n_selected_musics)
        # print("save_selected_musics: selected musics", self.selected_musics)

    def set_selected_transform(self):
        self.set_transform(self.transform_widget.checkedId())
        self.counter_go += 10
        if self.counter_go >= 21:
            self.gobutton.setEnabled(True)
            self.gobutton.setStyleSheet('QPushButton {background-color: #0066CC; color: white;}')
        # print("selected transform ", self.transform)

    def set_transform(self, value):
        self.transform = value

    def select_wavelet(self, text):
        self.mother_wavelet = text
        # print("mother wavelet", self.mother_wavelet)

    def select_wavelet_level(self, level):
        self.wave_nlevels = level
        # print("number of levels for wavelet=", self.wave_nlevels)

    def play(self, k):
        QtMultimedia.QSound.play(self.musics_list[k])

    def clear_all(self):
        self.clearbutton.setEnabled(False)
        self.clearbutton.setStyleSheet('QPushButton')
        self.pie_widget.clear_pie()

        self.err_widget.clear_errors()
        self.legend_widget.clear_legend()
        self.player_musics_list = list()
        for k in range(6):
            if k != 4:
                self.signal_widget_list[k].clear_plot()

        for k in range(6):
            self.dt_widget_list[k].clear_plot_dt()

    def numeric_elaboration(self):
        # fill the legend
        self.legend_widget.fill_legend(self.selected_musics, self.color_musics)
        # read yhe image
        data = PIL.Image.open(self.paintings_dir + self.painting_name)
        # save the image intensity
        image_intensity = np.asarray(data)[:, :, 0]
        n_pixels = image_intensity.size
        total_time = float(n_pixels)/self.sample_rate
        # compute the discrete transform of the image
        coeffs_image, len_coeffs_image = self.transform_image(image_intensity)
        # print("number of pixels", n_pixels)

        # read the audio signals, plot and transform
        ma = []
        widget = 0
        for item in self.selected_musics:
            # read
            audio_signal, samplerate = librosa.load(self.musics_dir + item,
                                                    sr=None, mono=False,
                                                    duration=total_time)
            # reread the audio signal if it is needed
            if samplerate != self.sample_rate:
                print("Please wait, the track" + item + "will be reread")
                audio_signal, samplerate = librosa.load(self.musics_dir + item,
                                                        sr=None, mono=False)
            # select the first trace
            if audio_signal.shape.__len__() > 1:
                audio_signal = audio_signal[0]

            # normalize
            audio_signal = audio_signal / np.linalg.norm(audio_signal, np.inf)
            # plot
            self.signal_widget_list[widget].my_plot(audio_signal, samplerate, item,
                                                    self.color_musics[widget])
            self.signal_widget_list[widget].draw()

            # print("audio_signal.shape", audio_signal.shape, "len_data", len_data)
            # compute (and normalize) the transform of the audio signal
            coeffs_audio, len_coeffs_audio = self.transform_audio(audio_signal, n_pixels)
            # save coeff_audio in the matrix
            ma.append(coeffs_audio)
            # align 2d-dwt coefficients of the image
            self.dt_widget_list[widget].axes.grid(True)
            if self.transform <= 1:
                self.dt_widget_list[widget].my_plot_dwt(coeffs_audio, item,
                                                        self.color_musics[widget])
            else:
                self.dt_widget_list[widget].my_plot_dft(coeffs_audio, item,
                                                        self.color_musics[widget],
                                                        self.sample_rate)
            self.dt_widget_list[widget].draw()
            widget += 1
        n_audio = widget

        if self.transform == 1:
            coeffs_image = self.align_dwt2_to_dwt1(coeffs_audio,
                                                   len_coeffs_audio,
                                                   coeffs_image,
                                                   len_coeffs_image)
        elif self.transform == 3:
            coeffs_image = coeffs_image.flatten()

        # plot the transform of the original image
        if self.transform <= 1:
            self.dt_widget_list[4].my_plot_dwt(coeffs_image,
                                               "transform of the painting",
                                               self.color_painting[0])
        else:
            self.dt_widget_list[4].my_plot_dft(coeffs_image,
                                               "transform of the painting",
                                               self.color_painting[0],
                                               self.sample_rate)
        self.dt_widget_list[4].draw()

        # solve the least square problem
        nmatrix = np.sum(len_coeffs_audio)

        matrix = np.zeros([nmatrix, n_audio], dtype=complex)
        k = 0
        for item in ma:
            matrix[:, k] = np.array(item, dtype=complex)
            k += 1
        del ma
        matrix = np.matrix(matrix)
        # a = np.matmul(matrix.T, matrix)
        a = np.matmul(matrix.H, matrix)
        # align the size of coeffs_audio and coeffs_image (to the first one)
        cis = coeffs_image.size
        cas = coeffs_audio.size
        if cis < cas:
            coeffs_image = np.r_[coeffs_image, np.zeros([1, cas-cis])]
        elif cis > cas:
            coeffs_image = coeffs_image[0:cas]
        c = np.matrix(coeffs_image)

        b = np.matmul(matrix.H, c.T)
        # alpha = a \ b
        self.alpha = np.linalg.solve(a, b)
        # reconstruct painting signal
        coeffs_projection_image = np.matmul(matrix, self.alpha)
        coeffs_projection_image = coeffs_projection_image/np.linalg.norm(coeffs_projection_image, np.inf)
        coeffs_projection_image = np.array(coeffs_projection_image).flatten()
        coeffs_projection_image_real = coeffs_projection_image.real
        if self.transform <= 1:
            self.dt_widget_list[5].my_plot_dwt(coeffs_projection_image_real,
                                               "projection of the painting transform",
                                               self.color_painting[1])
        else:
            self.dt_widget_list[5].my_plot_dft(coeffs_projection_image_real,
                                               "projection of the painting transform",
                                               self.color_painting[1],
                                               self.sample_rate)
        self.dt_widget_list[5].draw()
        del matrix

        painting_signal = self.reconstruct_audio_signal(coeffs_projection_image,
                                                        len_coeffs_audio)
        # plot the signal of the image
        self.signal_widget_list[5].my_plot(painting_signal, samplerate,
                                           "audio signal of the painting",
                                           self.color_painting[1])
        self.signal_widget_list[5].draw()
        soundfile.write("sound1.wav", painting_signal, samplerate, format='WAV')

        # save the produced  music and connect it to the toggle
        self.player_painting = Player(CURRENT_DIR, "sound1.wav")
        self.toggle_painting.setEnabled(True)
        self.toggle_painting.statusTip()
        self.toggle_painting.pressed.connect(self.player_painting.my_play)

        # plot the piechart
        self.alpha_percento = abs(self.alpha)/np.sum(abs(self.alpha))
        self.pie_widget.fill_pie(self.alpha_percento, self.selected_musics, self.color_musics)

        # compute the error between the transform of the image and the least square solution

        l2err = np.linalg.norm(coeffs_image-coeffs_projection_image)
        l2err_rel = l2err / np.linalg.norm(coeffs_image)

        self.err_widget.fill_errors(l2err, l2err_rel)

        # activate the clear button
        self.clearbutton.setEnabled(True)
        self.clearbutton.setStyleSheet('QPushButton {background-color: #0066CC; color: white;}')
        # deactivate the go button
        self.gobutton.setEnabled(False)
        self.gobutton.setStyleSheet('QPushButton')

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
                coeffs = np.r_[coeffs, v]  # concatenate
            coeffs = coeffs / np.linalg.norm(coeffs, np.inf)
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
            # print("len_coeffs",len_coeffs)
            # print("coeffs.size", coeffs.size,np.sum(len_coeffs))
            coeffs = coeffs / np.linalg.norm(coeffs, np.inf)
        elif self.transform == 2:
            #   2d --> 1d -->DFT
            x = image_intensity.T.flatten()
            coeffs = fft.fft(x)
            coeffs = coeffs/np.amax(np.amax(np.abs(coeffs)))
            len_coeffs = coeffs.size
        elif self.transform == 3:
            #   2d --> DFT --> 1d
            coeffs = fft.fft2(image_intensity)
            # print("coeffs.shape", coeffs.shape)
            coeffs.flatten()
            coeffs = coeffs/np.amax(np.amax(np.abs(coeffs)))
            len_coeffs = coeffs.size

        return coeffs, len_coeffs

    def transform_audio(self, data, n_pixels):

        # print('dtw_coeff.shape ', dtw_coeff.shape)
        # pad the array with zero values
        len_data = data.size
        # print("before: N, len_data, data.size",N,len_data,data.size)
        if n_pixels < len_data:
            data = data[0:n_pixels]
        elif n_pixels > len_data:
            data = np.r_[data, np.zeros(len_data)]
        # print("after: N, len_data, data.size",N,len_data,data.size)

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
            # print("len_coeffs", len_coeffs)
            # print("coeffs.size", coeffs.size)
            coeffs = coeffs / np.linalg.norm(coeffs, np.inf)
        else:
            coeffs = fft.fft(data)
            coeffs = coeffs / np.linalg.norm(coeffs, np.inf)
            len_coeffs = coeffs.size
        # print("len_coeffs audio", len_coeffs)
        return coeffs, len_coeffs

    def align_dwt2_to_dwt1(self, coeffs_audio, len_coeffs_audio,
                           coeffs_image, len_coeffs_image):
        # print("leni, lena", np.sum(len_coeffs_image),np.sum(len_coeffs_audio))
        n = coeffs_audio.size
        coeffs1 = np.zeros(n)
        nci = len_coeffs_image[0]
        nca = len_coeffs_audio[0]
        nrep = nca // nci
        # print("nci, nca, nrep", nci,nca,nrep)
        krep = 0
        coeffs1[krep*nci:krep*nci+nci] = coeffs_image[0:nci]
        for krep in range(1,nrep):
            # coeffs1[krep*nci:krep*nci+nci] = coeffs_image[0:nci]
            coeffs1[krep*nci:krep*nci+nci] = np.zeros([nci])
        na = nca
        ni = nci
        # print("ni, na", ni,na)
        for l1 in range(self.wave_nlevels):
            nci = len_coeffs_image[1+l1*3]
            nca = len_coeffs_audio[1+l1]
            nrep = nca // (nci*3)
            # print("l1, nci, nca, nrep", l1, nci, nca, nrep)
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

            # print("ni, na", ni,na)
        return coeffs1

    def reconstruct_audio_signal(self, coeffs_projection_image, len_coeffs_audio):
        if self.transform <= 1:
            coeffs = []
            index = 0
            for item in len_coeffs_audio:
                v = np.array(coeffs_projection_image[index:index+item])
                coeffs.append(v)
                index = index+item
            x = pywt.waverec(coeffs, self.mother_wavelet)
        else:
            x = fft.ifft(coeffs_projection_image)
        signal = x.real / np.linalg.norm(x.real, np.inf)
        signal = signal.flatten()
        return signal


app = QtWidgets.QApplication(sys.argv)
# directory where the audio-files are stored (absolute path,
# it is needed for the reproduction of the sound)
musics_dir = "absolute_path_of_directory_of_musics"
# directory where the image-files are stored  (absolute or relative path)
paintings_dir = "path_of_directory_of_images"
warnings.filterwarnings('ignore')
window = MainWindow(musics_dir, paintings_dir)
window.show()
app.exec_()
