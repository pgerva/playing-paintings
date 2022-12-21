# Playing paintings

**<kbd>PlayingPaintings</kbd>** is a Python app for the analysis of similarities between a
digital image and a set of music tracks, by applying the least square approach described in [this](https://doi.org/10.48550/arXiv.2206.14142)
reference.  
For copyright reasons, **the digital images (.png) and the music tracks (.mp3
or .wav) needed to run the app have to be
supplied by you.**


# Getting started

- [download and install](#download)
- [the data](#data)
- [run the app](#run)
- [generated files](#newfiles)
- [warnings](#warnings)

<a name="download"></a>

# Download and install

- Clone the github repository or download the zip file and uncompress it.
- Install the following Python packages:

  pip install pathlib  
  pip install numpy  
  pip install scipy  
  pip install PyWavelets  
  pip install librosa  
  pip install soundfile  
  pip install Pillow  
  pip install PySide2  
  pip install matplot  


<a name="data"></a>

# The data

To run the app, a set of audio tracks (mp3 or wav) and a set of digital images
(png) have to be available on your storage space (not necessarily in the same directory in which the app is).


- Modify the value of
the variables *musics_dir* and *paintings_dir* at the lines 1241--1248 of the
PlayingPaintings.py script:

   - *musics_dir* must contain the (absolute) path of the directory in which your music tracks are stored,

   - *paintings_dir* must contain the (absolute or relative) path of the directory in which your digital images are stored.

- Modify the content of the files *musics.csv* and *paintings.csv* that are
  present in the directory.

   - The file *musics.csv* must contain the list of the .mp3 (or .wav) files
     of the musics that you want to consider for the analysis. These files must be present in the directory *musics_dir*.
   - The file *paintings.csv* must contain the list of the .png files (without
     extension) of the paintings that you want to consider for the analysis. These files must be present in the directory *paintings_dir*. The first line of the file *paintings.csv* must not contain a filename, but any string.

<a name="run"></a>

#  Run the app

When the app starts, select the input and run, more precisely:

- Step 1: select the painting from your list
- Step 2: select up to 4 musical pieces from your list
- Step 3: select the transform for the painting and the musics
- Click on the *Go* button and wait for the graphical output
- Listen to the musics
- Clear the graphical output, set new inputs and click on the *Go* button  again.

<a name="newfiles"></a>

#  Generated files

The directory *./_small* will be created by the app to store small (256x256)
reproductions of your images to display in the app panel. The original images
will be used by the numerical algorithm to perform the analysis and provide the
new music.

The new music is saved in the file *sound1.wav*.

<a name="warnings"></a>

# Warnings

The size of your files is free, however, bear in mind that the larger the image file, the longer the time to read it and the music tracks, the heavier the computation, and the longer the waiting time to see the results.  

If the music tracks are too short compared with the size of the image, i.e., if
the number of samples of the music track is smaller than the number of pixels of the image, then the music track will be padded by replicating the data.

 

# Referencing

If you write a paper using results obtained with the help of **<kbd>PlayingPaintings</kbd>**,
please cite [this](https://doi.org/10.48550/arXiv.2206.14142) reference:

P. Gervasio, A. Quarteroni, D. Cassani. 
<i>Let do paintings play.</i>  (2022)
https://doi.org/10.48550/arXiv.2206.14142
