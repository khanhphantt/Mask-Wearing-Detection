import os

import av
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_webrtc import (
    RTCConfiguration,
    WebRtcMode,
    webrtc_streamer,
)
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Setting custom Page Title and Icon with changed layout and sidebar state
st.set_page_config(
    page_title="Face Mask Detector",
    page_icon="😷",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Load models
prototxtPath = os.path.sep.join(
    [
        "face_detector",
        "deploy.prototxt",
    ],
)
weightsPath = os.path.sep.join(
    [
        "face_detector",
        "res10_300x300_ssd_iter_140000.caffemodel",
    ],
)
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
maskNet = load_model("mask_detector.model")
example_image = "images/out.jpeg"
print("[INFO] loaded face mask detector model")

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
)


class VideoProcessor:
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = mask_image(img)
        return av.VideoFrame.from_ndarray(img, format="bgr24")


def local_css(file_name):
    # Method for reading styles.css and applying necessary changes to HTML
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def mask_image(image):
    # dimensions
    (h, w) = image.shape[:2]

    # construct a blob from the image
    blob = cv2.dnn.blobFromImage(
        image,
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0),
    )

    # pass the blob through the network and obtain the face detections
    print("[INFO] computing face detections...")
    faceNet.setInput(blob)
    detections = faceNet.forward()

    face_count = 0
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence associated with the detection
        confidence = detections[0, 0, i, 2]
        # print(f"[INFO] face {i}: {confidence}")
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
            label = f"{label}: {max(mask, withoutMask) * 100:.2f}%"

            # display the label & bbox rectangle on the output frame
            cv2.putText(
                image,
                label,
                (startX, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                2,
            )
            cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
        else:
            break
    text = f"[INFO] Detect {face_count} face(s)."
    print(text)
    cv2.putText(
        image,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        (0, 255, 0),
        2,
    )
    return image


def mask_detection():
    local_css("css/styles.css")
    st.markdown(
        '<h6 align="center">TT monthly challenge - Jun 2022</h6>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<h1 align="center">😷 Face Mask Detection</h1>',
        unsafe_allow_html=True,
    )
    st.set_option("deprecation.showfileUploaderEncoding", False)
    choice = st.sidebar.radio(
        "Select an input option:",
        ["Image", "Webcam"],
    )
    if choice == "Image":
        # st.sidebar.markdown('Upload your image ⬇')
        image_file = st.sidebar.file_uploader("", type=["jpg", "jpeg", "png"])

        if not image_file:
            text = """This is a detection example.
            Try your input from the left sidebar.
            """
            st.markdown(
                '<h6 align="center">' + text + "</h6>",
                unsafe_allow_html=True,
            )
            st.image(example_image, use_column_width=True)
        else:
            st.sidebar.markdown(
                "__Image is uploaded successfully!__",
                unsafe_allow_html=True,
            )
            st.markdown(
                '<h4 align="center">Detection result</h4>',
                unsafe_allow_html=True,
            )

            PIL_image = Image.open(image_file)

            image = np.array(PIL_image)
            image = mask_image(image)
            st.image(image, use_column_width=True)

    if choice == "Webcam":
        st.sidebar.markdown('Click "START" to connect this app to a server')
        st.sidebar.markdown("It may take a minute, please wait...")
        webrtc_streamer(
            key="WYH",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
            video_processor_factory=VideoProcessor,
            async_processing=True,
        )


mask_detection()
