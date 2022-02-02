# Playing paintings

This repository contains a Python program (app.py) to analyse similarities between a
painting and a set of music tracks, by applying the least square approach described in 
Reference [1].

To run the app, a set of audio tracks (mp3 or wav) and a set of digital images (.png or .jpg) have to be available on the disk (not necessarily in the same directory where the app is).
Moreover, two files must be present in the same directory where the app is:
musics.csv and paintings.csv

The file musics.csv contains the list of the audio tracks available in your storage that you want to consider.
The file paintings.csv contains the list of the paintings available in your storage that you want to compare with the audio tracks.
Examples of these two files are present in this repository.

Finally, modify lines 968 and 970 of app.py, by replacing the string "absolute_path_of_directory_of_musics" with the directory where your music tracks are stored and the string "path_of_directory_of_images" with the directory where the digital images are stored.

Notice that the path of the images directory can be relative or absolute.

 
 
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
