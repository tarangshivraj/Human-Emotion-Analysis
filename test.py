import cv2
import streamlit as st
import mediapipe as mp
import os 
import csv
import numpy 

st.set_page_config(
     page_title="Minor Project CCE",
     page_icon="random",
     layout="wide",
     initial_sidebar_state="expanded",
     menu_items={
         'Get Help': 'https://www.extremelycoolapp.com/help',
         'Report a bug': "https://www.extremelycoolapp.com/bug",
         'About': "# This is a header. This is an *extremely* cool app!"
     }
 )

mpDraw = mp.solutions.drawing_utils
mpHolistic = mp.solutions.holistic

st.title("Human Behaviour Analysis by Body Language using Machine Learning")
run = st.checkbox('Run')
FRAME_WINDOW = st.image([])
capture = cv2.VideoCapture(0)
with mpHolistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
	while run and capture.isOpened():
	    _, frame = capture.read()
	    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	    FRAME_WINDOW.image(frame)
# else:
#     st.write('Stopped')
