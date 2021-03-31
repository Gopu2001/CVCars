# Gopu2001-cvcars
Computer Vision for Finding specific Cars in an image / video

## Progress by 2/25/2021
Completion of the execute file... As far as functionality, I have reached the bare minimum of my project plan (predicting on a set of images, but there may be some output modifications of the code that I may change), but there is still more to do:
TO DO:
 * Increase the accuracy (and slow the runtime) of the Classification net by using a more accurate neural net
 * Increase the speed (and decrease the accuracy) of the Detection net by using, perhaps, a YOLO network

## Okay... So, here's what I got (2/18/2021).
After some time of analyzing the detector, I decided that working with the Faster R-CNN Object Detector seems to work best for my case because of the smaller amount of image processing that I would have to do. Furthermore, it is recommended by Matlab to start with the Faster R-CNN Object Detector because of its simplicity to use. Finally, after some work with the R-CNN, the detection code words! YAY!

Next steps: Linking the detector with the classifier and possibly linking it with a live camera feed / video.

## Progress by 2/11/2021
While working with Matlab for training models seems much easier, I find that I am a lot of the time copying the code from tutorials when I don't know what they are doing. So, after copying some code that trains a pretrained car detector (BTW, this uses resnet50), I realized that I should not run the code right away, but try to figure out how whether the decisions and assumptions made in the tutorial that I noted down best reflect what my project needs. For instance, a better decision may be to train the R-CNN object detector using a different model, maybe the resnet101, or perhaps alexnet as used in the recognizer. Also, if I want code generation support for an NVIDIA GPU, I would need to use a different detection algorithm.

## Using Matlab
After spending some time playing around with Matlab and the various tools, it seems that Matlab may be a decent tool for setting up this project and working on it.

### Initial Success (but with Alexnet — super fast, but can be inaccurate)!
![alt text](Screenshot%20of%20Initial%20Output.jpg "Sorry ... you missed out on a Screenshot of the Matlab code starting to run and train the model!")

### FIFTEEN MINUTES LATER...
![alt text](Screenshot%20Slow%20Fail.jpg "[Yeah, that initial success from above was short-lived.]")

Yeah, that initial success from above was short-lived. Well, I removed the code where the fault was occurring and training may take longer, but, I guess I'll see what happens.

## Notes Regarding Data Annotations
 * Data annotations include bounding boxes for the car images. Will make it more accurate to train the data, but will have to figure out another way to semantically segment the data
 * Data annotations depict 196 classes, where each class is: Make, Model, Year. Will make it more difficult to expand the model to newer cars.
 * Dataset was published in 2013, so newer cars are not included in the dataset. Will make it more difficult to expand the model to newer cars.

***
## Plan for CVCars
The general idea for this project is to train or fine-tune a deep learning computer vision model, specifically a semantically segmented model to find and display different cars (by make, model, and possibly year) in different colors. Semantic segmentation algorithms work to differentiate between objects and each pixel would be associated with an object, whether it be classified as part of the background, or an object, such as a person.

## Potential Applications (assuming the model is perfect)
 * For use by federal officers searching for a specific car (the suspect) during rush hour or in a parking lot.
 * Searching for your car in a packed parking lot
 * Helps 3rd-party automobile shops by seamlessly register a car with only an image
 * Semantic segmentation helps with humans deciphering output data so two cars next to each other would not be confused as the same car if placing boxes around the cars

## Research Plan
A lot of the code for this project will come from applying concepts from various tutorials (linked below).

## Information Sources
 * [Tutorial: Semantic Segmentation with OpenCV and deep learning using ENet](https://www.pyimagesearch.com/2018/09/03/semantic-segmentation-with-opencv-and-deep-learning/)
 * [A Research Paper on ENet Architecture (popular with SemSeg)](https://arxiv.org/pdf/1606.02147.pdf)
 * [Tutorial: Image Segmentation using PixelLib](https://towardsdatascience.com/image-segmentation-with-six-lines-0f-code-acb870a462e8)
 * [Documentation on the PixelLib Module](https://pixellib.readthedocs.io/en/latest/)
 * [Cars Image Database](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)
 * [[Matlab] Image Training Models](https://www.mathworks.com/help/deeplearning/ug/pretrained-convolutional-neural-networks.html)
 * [[Matlab] What is Resnet50?](https://www.mathworks.com/help/deeplearning/ref/resnet50.html)
 * [[Matlab] Fine-tuning ResNet-50](https://www.mathworks.com/matlabcentral/mlc-downloads/downloads/submissions/62558/versions/2/previews/DEMOS/Demo2_TransferLearning/TransferLearningDemo.m/index.html)
 * [[Matlab] Classifying images using a CNN](https://www.mathworks.com/help/deeplearning/ref/seriesnetwork.classify.html)
 * [[Matlab] Demo code for Fine-tuning](https://www.mathworks.com/matlabcentral/mlc-downloads/downloads/submissions/62558/versions/2/previews/DEMOS/Demo2_TransferLearning/TransferLearningDemo.m/index.html)
 * [[Matlab] Choosing an object detector](https://www.mathworks.com/help/vision/ug/choose-an-object-detector.html)
 * [[Matlab] Documentation / Guide for Object Detection using Deep Learning](https://www.mathworks.com/help/vision/object-detection-using-deep-learning.html)
 * [[Matlab] Tutorial For Object Detection using a Faster R-CNN Deep Learning](https://www.mathworks.com/help/vision/ug/object-detection-using-faster-r-cnn-deep-learning.html)

## Steps to Download and Use this Repo
 * Clone this Repository
 * Download the car image database from [here](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)
 * Install Matlab R2020B
 * Install "Deep Learning Toolbox Model for AlexNet Network Support Package" from the Add-on Manager
 * Install "Deep Learning Toolbox Model for Resnet50 Network Support Package" from the Add-on Manager
 * Open this repository in Matlab R2020B
 * Run the matlab_needed_functions.m to generate the alexnet classifier file
 * To run the classifier, load the classifier and run classify(alexnet, imageset)
Currently, the car detector.m does not work and is not linked with the classifier.