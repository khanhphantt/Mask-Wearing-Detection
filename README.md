<h1 align="center">Face Mask Detection</h1>

[Demo on Streamlit](https://khanhphantt-face-mask-detection-app-8w33p9.streamlitapp.com/)

Face Mask Detection System built with OpenCV, Keras/TensorFlow using Deep Learning and Computer Vision concepts in order to detect face masks in static images as well as in real-time video streams.

<p align="center"><img src="https://github.com/khanhphantt/Face-Mask-Detection/blob/master/images/out.jpeg" width="700" height="400"></p>

## :warning: TechStack/framework used
- [OpenCV](https://opencv.org/)
- [Caffe-based face detector](https://caffe.berkeleyvision.org/)
- [Keras](https://keras.io/)
- [TensorFlow](https://www.tensorflow.org/)
- [MobileNetV2](https://arxiv.org/abs/1801.04381)

### :file_folder: Dataset
This dataset consists of __4095 images__ belonging to two classes:
*	__with_mask: 2165 images__
*	__without_mask: 1930 images__

The images used were real images of faces wearing masks. The images were collected from the following sources:

* __Bing Search API__ ([See Python script](https://github.com/khanhphantt/Face-Mask-Detection/blob/master/search.py))
* __Kaggle datasets__
* __RMFD dataset__ ([See here](https://github.com/X-zhangyang/Real-World-Masked-Face-Dataset))


## Installation
1. Clone the repo
```
$ git clone https://github.com/khanhphantt/Face-Mask-Detection
```

2. Change your directory to the cloned repo
```
$ cd Face-Mask-Detection
```

3. Create a Python virtual environment named 'test' and activate it
```
$ python -m venv .venv
$ .venv/bin/activate
```

4. Install the libraries required
```
$ pip install -r requirements.txt
```

## Working

1. Train data:
```
$ python3 train_mask_detector.py --dataset dataset
```

2. Detect face masks in an image:
```
$ python3 detect_mask_image.py --image images/pic1.jpeg
```

3. Detect face masks in real-time video (webcam):
```
$ python3 detect_mask_video.py
```

## Streamlit app

Face Mask Detector webapp using Tensorflow & Streamlit:
```
$ streamlit run app.py
```
