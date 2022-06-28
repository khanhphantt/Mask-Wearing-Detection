from pyrsistent import m
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
from PIL import Image, ImageEnhance
import numpy as np
import cv2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

# Setting custom Page Title and Icon with changed layout and sidebar state
st.set_page_config(page_title='Face Mask Detector', page_icon='😷', layout='centered', initial_sidebar_state='expanded')

# Load models
prototxtPath = ".\\face_detector\\deploy.prototxt"
weightsPath = ".\\face_detector\\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
print("[INFO] loading face mask detector model...")
maskNet = load_model("mask_detector.model")

#RTC_CONFIGURATION = RTCConfiguration(
#    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
#)

class VideoProcessor:
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = mask_image(img)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

def local_css(file_name):
    # Method for reading styles.css and applying necessary changes to HTML
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def mask_image(image):
    # dimensions
    #image = cv2.imread("./images/out.jpg")
    (h, w) = image.shape[:2]

    # construct a blob from the image
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    print("[INFO] computing face detections...")
    faceNet.setInput(blob)
    detections = faceNet.forward()
    
    face_count = 0
    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence associated with the detection
        confidence = detections[0, 0, i, 2]
        #print(f"[INFO] face {i}: {confidence}")
        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > 0.5:
            face_count += 1
            # compute the (x, y)-coordinates of the object's bbox
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI
            # ordering, resize it to 224x224, and preprocess it
            face = image[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            # pass the face through the model to determine if the face
            # has a mask or not
            (mask, withoutMask) = maskNet.predict(face)[0]

            # determine the class label and color we'll use to draw
            # the bounding box and text
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (255, 0, 0)

            # include the probability in the label
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            # display the label and bounding box rectangle on the output
            # frame
            cv2.putText(image, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
            #RGB_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            break
    text = f"[INFO] Detect {face_count} face(s)."
    print(text)
    cv2.putText(image, text, (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 255, 0), 2)
    return image

def mask_detection():
    local_css("css/styles.css")
    st.markdown('<h1 align="center">😷 Face Mask Detection</h1>', unsafe_allow_html=True)
    activities = ["Image", "Webcam"]
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.sidebar.markdown("# Mask Detection on?")
    #choice = st.sidebar.selectbox("Choose among the given options:", activities)
    choice = st.sidebar.radio("Choose among the given options:", activities)
    if choice == 'Image':
        st.sidebar.markdown("#### Upload your image here ⬇")
        image_file = st.sidebar.file_uploader("", type=["jpg", "jpeg", "png"])
        if image_file is not None:
            print("image_file", image_file)
            
            st.sidebar.markdown('Image uploaded successfully!', unsafe_allow_html=True)
            st.markdown('<h4 align="center">Detection result</h2>', unsafe_allow_html=True)
        
            PIL_image = Image.open(image_file)  # making compatible to PIL
            #st.image(PIL_image, use_column_width=True)
            
            image = np.array(PIL_image.convert("RGB"))
            image = mask_image(image)
            #im = our_image.save('./images/out.jpg')
            #saved_image = st.image(image_file, caption='', use_column_width=True)
            #if st.button('Process'):
            st.image(image, use_column_width=True)

    if choice == 'Webcam':
        st.sidebar.markdown('Click "START" button to begin', unsafe_allow_html=True)
        #img_file_buffer = st.camera_input("Try to put on/off the mask")
        webrtc_ctx = webrtc_streamer(
            key="WYH",
            mode=WebRtcMode.SENDRECV,
            #rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
            video_processor_factory=VideoProcessor,
            async_processing=True,
        )

            
mask_detection()

