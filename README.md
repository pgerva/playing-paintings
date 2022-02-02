# Playing paintings

This repository contains a Python program (app.py) to analyse similarities between a
digital image and a set of music tracks, by applying the least square approach described in 
Reference [1].  
The digital image and the music tracks have to be supplied by you.


# Files needed by the app

To run the app, a set of audio tracks (mp3 or wav) and a set of digital images (png or jpg) have to be available on the disk (not necessarily in the same directory in which the app is).
Moreover, two files must be present in the same directory where the app is:
<i>musics.csv</i> and <i>paintings.csv</i>

The file musics.csv must contain the list of the audio tracks available in your storage that you want to consider.
The file paintings.csv must contain the list of the paintings available in your storage that you want to compare with the audio tracks.
Examples of these two files are present in this repository.


# Mandatory changes to the file app.py

Modify lines 968 and 970 of app.py, by replacing the string "absolute_path_of_directory_of_musics" with the directory in which your music tracks are stored and the string "path_of_directory_of_images" with the directory in which the digital images are stored.

While the path of the directory with the musics has to be absolute, the path of the directory with the images can be relative or absolute.
 
 
# Required packages

os  
sys  
warnings  
csv  
numpy  
scipy  
pywt  (pip install PyWavelets)  
pywt.data  
librosa  
soundfile  
PIL (pip install Pillow)  
PySide2  
qtwidgets  
matplotlib  
  

# References

[1] P. Gervasio, A. Quarteroni, D. Cassani. "Let do paintings play."  (2022)
