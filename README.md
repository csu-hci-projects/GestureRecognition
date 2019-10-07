# GestureRecognition

The intent of this repository is to make use of gesture recognition as part of a 3D user interface, and evaluate the 3D user interface. The project builds on top the [CNN Gesture Recognizer](https://github.com/asingh33/CNNGestureRecognizer).


Key Requirements:
- Python 3.6.1
- OpenCV 3.4.1
- Keras 2.0.2
- Tensorflow 1.2.1
- Theano 0.9.0

# Repo contents
- **trackgesture.py** : The main script launcher. This file contains all the code for UI options and OpenCV code to capture camera contents. This script internally calls interfaces to gestureCNN.py.
- **gestureCNN.py** : This script file holds all the CNN specific code to create CNN model, load the weight file (if model is pretrained), train the model using image samples present in **./imgfolder_b**, visualize the feature maps at different layers of NN (of pretrained model) for a given input image present in **./imgs** folder.
- **imgfolder_b** : This folder contains all the 4015 gesture images previously taken in order to train the model.
- **_ori_4015imgs_weights.hdf5_** : This is pretrained file. GitHub does not allow files over 100MB; find the link to the actual file here - https://drive.google.com/open?id=0B6cMRAuImU69SHNCcXpkT3RpYkE
- **_imgs_** - This is an optional folder of few sample images that one can use to visualize the feature maps at different layers. These are few sample images from imgfolder_b only.
- **_ori_4015imgs_acc.png_** : This is just a pic of a plot depicting model accuracy Vs validation data accuracy for the pretrained model.
- **_ori_4015imgs_loss.png_** : This is just a pic of a plot depicting model loss Vs validation loss for the pretrained model.

# Usage
**On Mac**
```bash
eg: With Theano as backend
$ KERAS_BACKEND=theano python trackgesture.py 
```
**On Windows**
```bash
eg: With Tensorflow as backend
> set "KERAS_BACKEND=tensorflow"
> python trackgesture.py 
```

# Features
The pre-trained gestures are:
- OK
- PEACE
- STOP
- PUNCH
- NOTHING (i.e. when none of the above gestures are input)

This application provides following functionalities:
- Prediction: Which allows the app to guess the user's gesture against pretrained gestures.
- New Training: Which allows the user to retrain the NN model. User can change the model architecture or add/remove new gestures. This app has inbuilt options to allow the user to create new image samples of user defined gestures if required.
- Visualization: Which allows the user to see feature maps of different NN layers for a given input gesture image. Interesting to see how NN works and learns things.


# Demo 
Youtube link - https://www.youtube.com/watch?v=Et6JcMyF7SU

# Gesture Input
OpenCV is being used for capturing the user's hand gestures. Post processing is being done on the captured images, such as binary thresholding, blurring, gray scaling, etc. to highlight the contours & edges.

There are two modes of capture:
- Binary Mode: The image is converted to grayscale, then a gaussian blur effect with adaptive threshold filtering is applied. This mode is useful with an empty background like a wall, whiteboard etc.
- SkinMask Mode: The input image is converted to HSV and a range is put on the H,S,V values based on skin color range. This is followed by applying errosion, followed by dilation. Then a gaussian blur to smoothen out the noise, and using that output as a mask on original input to mask out everything other than skin colored. The output is then grayscaled. This mode is useful when there is good amount of light and the background is not very empty.

**Binary Mode processing**
```python
gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray,(5,5),2)   
th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
ret, res = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
```

![OK gesture in Binary mode](https://github.com/ApoorvDP/GestureRecognition/blob/master/imgfolder_b/iiiok160.png)


**SkindMask Mode processing**
```python
hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
#Apply skin color range
mask = cv2.inRange(hsv, low_range, upper_range)

mask = cv2.erode(mask, skinkernel, iterations = 1)
mask = cv2.dilate(mask, skinkernel, iterations = 1)

#blur
mask = cv2.GaussianBlur(mask, (15,15), 1)
#cv2.imshow("Blur", mask)

#bitwise and mask original frame
res = cv2.bitwise_and(roi, roi, mask = mask)
# color to grayscale
res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
```
![OK gesture in SkinMask mode](https://github.com/ApoorvDP/GestureRecognition/blob/master/imgfolder_b/iiok44.png)


# CNN Model
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

# Training
The existing model was trained using an image set of 4015 images i.e. 803 image samples per class, for 15 epochs.

![Training Accuracy Vs Validation Accuracy](https://github.com/asingh33/CNNGestureRecognizer/blob/master/ori_4015imgs_acc.png)

![Training Loss Vs Validation Loss](https://github.com/asingh33/CNNGestureRecognizer/blob/master/ori_4015imgs_loss.png)

Attempting to re-train the model resulted in a weird macOS-specific bug at the end, which caused the program to hang before saving the model, thereby requiring a restart and erasing the freshly trained model.

![The bug](https://github.com/ApoorvDP/GestureRecognition/blob/Current/macOS_bug.png)


# Conclusion
As of now, the gesture recognition is currently flawed, and the model needs to be re-trained. Currently collaborating with the repository owner to work towards improving the gesture recognition.
