import streamlit as st
import cv2
import numpy as np
import time

# set page config
st.set_page_config(page_title='Chatbot TLU', page_icon="🦈", layout='wide')

# sidebar
with st.sidebar:
	st.title('🦙💬 TLU Chatbot')
	st.write('This chatbot is created using the open-source LLM model and built from Ollama. ')

# main page
st.markdown("# Welcome to the Chatbot TLU! :sunglasses: :sunglasses:")


# # webcam
# enable = st.checkbox("Enable camera")

# # align the camera in the center and fit the width
# col1, col2, col3 = st.columns([1,10,1])
# with col2:
#     picture = st.camera_input("Take a picture", disabled=not enable)
    
# if picture:
#     st.image(picture, use_column_width=True)

# -----------------

# Stream video from webcam
st.title("Webcam Live Feed")

# Use session state to track camera state
if 'camera_running' not in st.session_state:
    st.session_state.camera_running = False

# Create buttons for camera control
if not st.session_state.camera_running:
    if st.button('Start Camera'):
        st.session_state.camera_running = True
        st.experimental_set_query_params(camera_running="true")

if st.session_state.camera_running:
    if st.button('Stop Camera'):
        st.session_state.camera_running = False
        st.experimental_set_query_params(camera_running="false")

FRAME_WINDOW = st.empty()

# Camera handling
if st.session_state.camera_running:
    camera = cv2.VideoCapture(0)
    
    try:
        while st.session_state.camera_running:
            ret, frame = camera.read()
            if not ret:
                st.error("Failed to capture image from camera")
                st.session_state.camera_running = False
                st.experimental_set_query_params(camera_running="false")
                break
                
            # Convert from BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame)
            # time.sleep(0.0083)  # ~120 FPS
            
    except Exception as e:
        st.error(f"Error accessing camera: {str(e)}")
        
    finally:
        camera.release()
else:
    st.write('Stopped')

# text input
st.markdown("Type your message below:")