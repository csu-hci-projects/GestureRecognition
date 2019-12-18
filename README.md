# GestureRecognition

The intent of this repository is to make use of gesture recognition as part of a 3D user interface, and evaluate the 3D user interface. The project builds on top the [CNN Gesture Recognizer](https://github.com/asingh33/CNNGestureRecognizer). The game is developed using a [PyGame tutorial](https://www.youtube.com/watch?v=PjgLeP0G5Yw), and the images are downloaded from the [linked GitHub resource](https://github.com/techwithtim/Side-Scroller-Game/tree/master/images) in the tutorial. The alternate gesture system, is from another [gesture recognition repository](https://github.com/Gogul09/gesture-recognition). All codes were changed to use gestures recognized as inputs to the game. 


Key Requirements:
- Python 3.6.1
- OpenCV 3.4.1
- Keras 2.0.2
- Tensorflow 1.2.1
- PyGame 1.9.6
- imutils 0.5.3

# Repo contents
- **recognition.py** : The main script launcher for Gesture Recognition. This file contains all the code for UI options and OpenCV code to capture camera contents. This script internally calls interfaces to gestureCNN.py.
- **Game.py**: The 2D side-scroller game, designed with the help of the aforementioned tutorial.
- **gestureCNN.py** : This script file holds all the CNN specific code to create CNN model, load the weight file (if model is pretrained), train the model using image samples present in **./imgfolder_b**, visualize the feature maps at different layers of NN (of pretrained model) for a given input image present in **./imgs** folder.
- **imgfolder_b** : This folder contains all the 4015 gesture images previously taken in order to train the model.
- **_pretrained_weights_MacOS.hdf5_** : This is the pretrained file for MacOS. GitHub does not allow files over 100MB; find the link to the actual file here: https://drive.google.com/file/d/1j7K96Dkatz6q6zr5RsQv-t68B3ZOSfh0/view
- **_pretrained_weights_WinOS.hdf5_** : This is the pretrained file for Windows. GitHub does not allow files over 100MB; find the link to the actual file here: https://drive.google.com/file/d/1PA7rJxHYQsW5IvcZAGeoZ-ExYSttFuGs/view
- **_imgs_** - This is an optional folder of few sample images that one can use to visualize the feature maps at different layers. These are few sample images from imgfolder_b only.
- **_ori_4015imgs_acc.png_** : This is just a pic of a plot depicting model accuracy Vs validation data accuracy for the pretrained model.
- **_ori_4015imgs_loss.png_** : This is just a pic of a plot depicting model loss Vs validation loss for the pretrained model.
- **_images_** : This is a repository for game resources, downloaded from linked GitHub resource.
- **gesture.p**: Pickle file, used for linking gesture recognition to game actions in real-time.
- **scores.txt**: Simple text file that stores all game scores till date.
- **recognize.py**: Alternate gesture recognition script, focuses on segmentation to count the number of fingers, which are then fed as inputs to the game.
- **resources**: This folder contains all resources necessary for _recognize.py_ to function.

# Usage
**On Mac**
To use gesture detection
```
$ python recognition.py
```
To use alternate gesture detection
```
$ python recognize.py
```
To run game
```
$ python Game.py
```

# Features

## game
![game.py](https://github.com/ApoorvDP/GestureRecognition/blob/master/imgs/game.png)

## recognition.py
The pre-trained gestures are:
- OK
- PEACE
- STOP
- PUNCH
- NOTHING (i.e. when none of the above gestures are input)

The OK gesture is the least consistent, often getting confused with STOP. The gesture recognition requires outdoor light, or bright indoor light to work well.

OpenCV is being used for capturing the user's hand gestures. Post processing is being done on the captured images, such as binary thresholding, blurring, gray scaling, etc. to highlight the contours & edges.

The CNN model used had the following 12 layers -
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 32, 198, 198)      320       
_________________________________________________________________
activation_1 (Activation)    (None, 32, 198, 198)      0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 32, 196, 196)      9248      
_________________________________________________________________
activation_2 (Activation)    (None, 32, 196, 196)      0         
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 32, 98, 98)        0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 32, 98, 98)        0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 307328)            0         
_________________________________________________________________
dense_1 (Dense)              (None, 128)               39338112  
_________________________________________________________________
activation_3 (Activation)    (None, 128)               0         
_________________________________________________________________
dropout_2 (Dropout)          (None, 128)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 5)                 645       
_________________________________________________________________
activation_4 (Activation)    (None, 5)                 0         
=================================================================
```

![recognition.py](https://github.com/ApoorvDP/GestureRecognition/blob/master/imgs/recognition.png)
![prediction.py](https://github.com/ApoorvDP/GestureRecognition/blob/master/imgs/prediction.png)

## recognize.py
![recognize.py](https://github.com/ApoorvDP/GestureRecognition/blob/master/imgs/recognize.png)

# Conclusion
Both gesture detection approaches work well, and are used to feed input to the game.
[Demo](https://www.youtube.com/watch?v=T8GZPrZsqYU)