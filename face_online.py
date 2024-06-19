# import streamlit as st
# import cv2
# import numpy as np
# from io import BytesIO
#
# # Load the pre-trained Haar Cascade classifier for face detection
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#
# def load_model(url):
#     if url.startswith('http'):
#         # Load video stream from URL
#         cap = cv2.VideoCapture(url)
#     else:
#         # Load video stream from local camera
#         cap = cv2.VideoCapture(int(url))
#
#     if not cap.isOpened():
#         raise Exception(f"Error: Unable to open camera at {url}")
#
#     return cap
#
# def detect_faces(frame, initial_distance=60):
#     if frame is not None:  # Check if the frame is valid
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
#
#         for (x, y, w, h) in faces:
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
#             distance = round(initial_distance * 200 / w, 2)  # Adjusting based on the size of the face detected
#             cv2.putText(frame, f'Distance: {distance} cm', (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
#
#         return frame
#     else:
#         return None
#
# def main():
#     st.title('Live Face Distance Detection')
#
#     camera_type = st.radio("Select Camera Type:", ('ESP32-CAM', 'Laptop/Desktop Camera'))
#     if camera_type == 'ESP32-CAM':
#         url = st.text_input('Enter ESP32-CAM video stream URL')
#     else:
#         url = st.text_input('Enter camera number (0 for default camera)')
#
#     if not url:
#         st.warning('Please enter the camera URL or number.')
#         return
#
#     cap = load_model(url)
#     st.write('Press "Start Face Detection" to begin')
#
#     start_detection = st.button('Start Face Detection')
#
#     if start_detection:
#         st.write('Face Detection started')
#         while True:
#             ret, frame = cap.read()  # Read frame from the camera
#
#             if not ret:
#                 st.error('Error: Unable to read frame from the camera.')
#                 break
#
#             # Perform face detection
#             frame_with_faces = detect_faces(frame)
#
#             if frame_with_faces is not None:
#                 # Display the frame with face detection and distance measurements
#                 st.image(frame_with_faces, channels='BGR', use_column_width=True)
#
#             if st.button('Stop Detection'):
#                 st.write('Face Detection stopped')
#                 break
#
#     cap.release()
#
# if __name__ == '__main__':
#     main()
#
# 
# 
# #
# import streamlit as st
# import cv2
# import numpy as np
# from io import BytesIO
# 
# # Load the pre-trained Haar Cascade classifier for face detection
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# 
# def load_model(url):
#     if url.startswith('http'):
#         # Load video stream from URL
#         cap = cv2.VideoCapture(url)
#     else:
#         # Load video stream from local camera
#         cap = cv2.VideoCapture(int(url))
# 
#     if not cap.isOpened():
#         raise Exception(f"Error: Unable to open camera at {url}")
# 
#     return cap
# def detect_faces(frame, initial_distance=60):
#     distance = None  # Initialize distance outside the loop
# 
#     if frame is not None:  # Check if the frame is valid
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
# 
#         for (x, y, w, h) in faces:
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
#             distance = round(initial_distance * 200 / w, 2)  # Update distance when face is detected
#             cv2.putText(frame, f'Distance: {distance} cm', (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
# 
#         return frame, distance  # Return the frame and calculated distance
#     else:
#         return None, None
# 
# #
# # def main():
# #     st.title('Live Face Distance Detection')
# #
# #     camera_type = st.radio("Select Camera Type:", ('ESP32-CAM', 'Laptop/Desktop Camera'))
# #     if camera_type == 'ESP32-CAM':
# #         url = st.text_input('Enter ESP32-CAM video stream URL')
# #     else:
# #         url = st.text_input('Enter camera number (0 for default camera)')
# #
# #     if not url:
# #         st.warning('Please enter the camera URL or number.')
# #         return
# #
# #     cap = load_model(url)
# #     st.write('Press "Start Face Detection" to begin')
# #
# #     start_detection = st.button('Start Face Detection')
# #
# #     if start_detection:
# #         st.write('Face Detection started')
# #         while True:
# #             ret, frame = cap.read()  # Read frame from the camera
# #
# #             if not ret:
# #                 st.error('Error: Unable to read frame from the camera.')
# #                 break
# #
# #             # Perform face detection and get the frame with faces and the distance
# #             frame_with_faces, distance = detect_faces(frame)
# #
# #             if frame_with_faces is not None:
# #                 # Display the frame with face detection and distance measurements
# #                 st.image(frame_with_faces, channels='BGR', use_column_width=True)
# #
# #                 # Display the live distance measurement
# #                 if distance is not None:
# #                     st.write(f'Live Distance: {distance} cm')
# #
# #             if st.button('Stop Detection', key='stop_btn'):
# #                 st.write('Face Detection stopped')
# #                 break
# #
# #     cap.release()
# #
# # if __name__ == '__main__':
# #     main()
# 
# 
# 
# 
# def main():
#     st.title('Live Face Distance Detection')
# 
#     camera_type = st.radio("Select Camera Type:", ('ESP32-CAM', 'Laptop/Desktop Camera'))
#     if camera_type == 'ESP32-CAM':
#         url = st.text_input('Enter ESP32-CAM video stream URL')
#     else:
#         url = st.text_input('Enter camera number (0 for default camera)')
# 
#     if not url:
#         st.warning('Please enter the camera URL or number.')
#         return
# 
#     cap = load_model(url)
#     st.write('Press "Start Face Detection" to begin')
# 
#     start_detection = st.button('Start Face Detection')
# 
#     if start_detection:
#         st.write('Face Detection started')
#         while True:
#             ret, frame = cap.read()
# 
#             if not ret:
#                 st.error('Error: Unable to read frame from the camera.')
#                 break
# 
#             frame_with_faces, distance = detect_faces(frame)
# 
#             if frame_with_faces is not None:
#                 st.image(frame_with_faces, channels='BGR', use_column_width=True)
# 
#                 if distance is not None:
#                     st.write(f'Live Distance: {distance} cm')
# 
#             if st.button('Stop Detection', key=f'stop_btn_{start_detection}'):
#                 st.write('Face Detection stopped')
#                 break
# 
#     cap.release()
# 
# if __name__ == '__main__':
#     main()
# 
# 




import streamlit as st
import cv2
import numpy as np
from io import BytesIO
import time

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def load_model(url):
    if url.startswith('http'):
        # Load video stream from URL
        cap = cv2.VideoCapture(url)
    else:
        # Load video stream from local camera
        cap = cv2.VideoCapture(int(url))

    if not cap.isOpened():
        raise Exception(f"Error: Unable to open camera at {url}")

    return cap

def detect_faces(frame, initial_distance=60):
    distance = None  # Initialize distance outside the loop

    if frame is not None:  # Check if the frame is valid
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            distance = round(initial_distance * 200 / w, 2)  # Update distance when face is detected
            cv2.putText(frame, f'Distance: {distance} cm', (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        return frame, distance  # Return the frame and calculated distance
    else:
        return None, None

def main():
    st.title('Live Face Distance Detection')

    camera_type = st.radio("Select Camera Type:", ('ESP32-CAM', 'Laptop/Desktop Camera'))
    if camera_type == 'ESP32-CAM':
        url = st.text_input('Enter ESP32-CAM video stream URL')
    else:
        url = st.text_input('Enter camera number (0 for default camera)')

    if not url:
        st.warning('Please enter the camera URL or number.')
        return

    cap = load_model(url)
    st.write('Press "Start Face Detection" to begin')

    start_detection = st.button('Start Face Detection')

    if start_detection:
        st.write('Face Detection started')
        while True:
            ret, frame = cap.read()  # Read frame from the camera

            if not ret:
                st.error('Error: Unable to read frame from the camera.')
                break

            # Perform face detection and get the frame with faces and the distance
            frame_with_faces, distance = detect_faces(frame)

            if frame_with_faces is not None:
                # Display the frame with face detection and distance measurements
                st.image(frame_with_faces, channels='BGR', use_column_width=True)

                # Display the live distance measurement
                if distance is not None:
                    st.write(f'Live Distance: {distance} cm')

            unique_key = f'stop_btn_{time.time()}'  # Generate a unique key using timestamp
            if st.button('Stop Detection', key=unique_key):
                st.write('Face Detection stopped')
                break

    cap.release()

if __name__ == '__main__':
    main()


#
#
# import streamlit as st
# import cv2
# import numpy as np
# from io import BytesIO
# import time
#
# # Load the pre-trained Haar Cascade classifier for face detection
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#
# def load_model(url):
#     if url.startswith('http'):
#         # Load video stream from URL
#         cap = cv2.VideoCapture(url)
#     else:
#         # Load video stream from local camera
#         cap = cv2.VideoCapture(int(url))
#
#     if not cap.isOpened():
#         raise Exception(f"Error: Unable to open camera at {url}")
#
#     return cap
#
# def detect_faces(frame, initial_distance=60):
#     distance = None  # Initialize distance outside the loop
#
#     if frame is not None:  # Check if the frame is valid
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
#
#         for (x, y, w, h) in faces:
#             distance = round(initial_distance * 200 / w, 2)  # Update distance when face is detected
#             cv2.putText(frame, f'Distance: {distance} cm', (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
#
#         return frame, distance  # Return the frame and calculated distance
#     else:
#         return None, None
#
# def main():
#     st.title('Live Face Distance Detection')
#
#     camera_type = st.radio("Select Camera Type:", ('ESP32-CAM', 'Laptop/Desktop Camera'))
#     if camera_type == 'ESP32-CAM':
#         url = st.text_input('Enter ESP32-CAM video stream URL')
#     else:
#         url = st.text_input('Enter camera number (0 for default camera)')
#
#     if not url:
#         st.warning('Please enter the camera URL or number.')
#         return
#
#     cap = load_model(url)
#     st.write('Press "Start Face Detection" to begin')
#
#     start_detection = st.button('Start Face Detection')
#
#     if start_detection:
#         st.write('Face Detection started')
#         while True:
#             ret, frame = cap.read()  # Read frame from the camera
#
#             if not ret:
#                 st.error('Error: Unable to read frame from the camera.')
#                 break
#
#             # Perform face detection and get the frame with faces and the distance
#             frame_with_faces, distance = detect_faces(frame)
#
#             if frame_with_faces is not None:
#                 # Display the live video feed with face detection and distance measurements
#                 st.image(frame_with_faces, channels='BGR', use_column_width=True)
#
#                 # Display the live distance measurement
#                 if distance is not None:
#                     st.write(f'Live Distance: {distance} cm')
#
#             if st.button('Stop Detection'):
#                 st.write('Face Detection stopped')
#                 break
#
#     cap.release()
#
# if __name__ == '__main__':
#     main()

#
# import streamlit as st
# import cv2
# import numpy as np
# from io import BytesIO
#
# # Load the pre-trained Haar Cascade classifier for face detection
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#
# def load_model(url):
#     if url.startswith('http'):
#         # Load video stream from URL
#         cap = cv2.VideoCapture(url)
#     else:
#         # Load video stream from local camera
#         cap = cv2.VideoCapture(int(url))
#
#     if not cap.isOpened():
#         raise Exception(f"Error: Unable to open camera at {url}")
#
#     return cap
#
# def detect_faces(frame, initial_distance=60):
#     if frame is not None:  # Check if the frame is valid
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
#
#         for (x, y, w, h) in faces:
#             distance = round(initial_distance * 200 / w, 2)  # Calculate distance based on the size of the face detected
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
#             cv2.putText(frame, f'Distance: {distance} cm', (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
#
#         return frame
#     else:
#         return None
#
# def main():
#     st.title('Live Face Distance Detection')
#
#     camera_type = st.radio("Select Camera Type:", ('ESP32-CAM', 'Laptop/Desktop Camera'))
#     if camera_type == 'ESP32-CAM':
#         url = st.text_input('Enter ESP32-CAM video stream URL')
#     else:
#         url = st.text_input('Enter camera number (0 for default camera)')
#
#     if not url:
#         st.warning('Please enter the camera URL or number.')
#         return
#
#     cap = load_model(url)
#     st.write('Press "Start Face Detection" to begin')
#
#     start_detection = st.button('Start Face Detection')
#
#     if start_detection:
#         st.write('Face Detection started')
#         while True:
#             ret, frame = cap.read()  # Read frame from the camera
#
#             if not ret:
#                 st.error('Error: Unable to read frame from the camera.')
#                 break
#
#             # Perform face detection and get the frame with faces and the distance
#             frame_with_faces = detect_faces(frame)
#
#             if frame_with_faces is not None:
#                 # Display the frame with face detection and distance measurements
#                 st.image(frame_with_faces, channels='BGR', use_column_width=True)
#
#             if st.button('Stop Detection'):
#                 st.write('Face Detection stopped')
#                 break
#
#     cap.release()
#
# if __name__ == '__main__':
#     main()

#
# working as
# import streamlit as st
# import cv2
# import numpy as np
# from io import BytesIO
# import time
#
# # Load the pre-trained Haar Cascade classifier for face detection
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#
# def load_model(url):
#     if url.startswith('http'):
#         # Load video stream from URL
#         cap = cv2.VideoCapture(url)
#     else:
#         # Load video stream from local camera
#         cap = cv2.VideoCapture(int(url))
#
#     if not cap.isOpened():
#         raise Exception(f"Error: Unable to open camera at {url}")
#
#     return cap
#
# def detect_faces(frame, initial_distance=60):
#     if frame is not None:  # Check if the frame is valid
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
#
#         for (x, y, w, h) in faces:
#             distance = round(initial_distance * 200 / w, 2)  # Calculate distance based on the size of the face detected
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
#             cv2.putText(frame, f'Distance: {distance} cm', (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
#
#         return frame
#     else:
#         return None
#
# def main():
#     st.title('Live Face Distance Detection')
#
#     camera_type = st.radio("Select Camera Type:", ('ESP32-CAM', 'Laptop/Desktop Camera'))
#     if camera_type == 'ESP32-CAM':
#         url = st.text_input('Enter ESP32-CAM video stream URL')
#     else:
#         url = st.text_input('Enter camera number (0 for default camera)')
#
#     if not url:
#         st.warning('Please enter the camera URL or number.')
#         return
#
#     cap = load_model(url)
#     st.write('Press "Start Face Detection" to begin')
#
#     start_detection = st.button('Start Face Detection')
#
#     if start_detection:
#         st.write('Face Detection started')
#         while True:
#             ret, frame = cap.read()  # Read frame from the camera
#
#             if not ret:
#                 st.error('Error: Unable to read frame from the camera.')
#                 break
#
#             # Perform face detection and get the frame with faces and the distance
#             frame_with_faces = detect_faces(frame)
#
#             if frame_with_faces is not None:
#                 # Display the frame with face detection and distance measurements
#                 st.image(frame_with_faces, channels='BGR', use_column_width=True)
#
#             unique_key = f'stop_btn_{time.time()}'  # Generate a unique key using timestamp
#             if st.button('Stop Detection', key=unique_key):
#                 st.write('Face Detection stopped')
#                 break
#
#     cap.release()
#
# if __name__ == '__main__':
#     main()


#
# import streamlit as st
# import cv2
# import numpy as np
# from deepface import DeepFace
# import time
#
# # Load the pre-trained Haar Cascade classifier for face detection
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#
# def load_model(url):
#     if url.startswith('http'):
#         # Load video stream from URL
#         cap = cv2.VideoCapture(url)
#     else:
#         # Load video stream from local camera
#         cap = cv2.VideoCapture(int(url))
#
#     if not cap.isOpened():
#         raise Exception(f"Error: Unable to open camera at {url}")
#
#     return cap
#
# def analyze_face(frame):
#     if frame is not None:
#         # Analyze the face attributes using DeepFace
#         results = DeepFace.analyze(frame, actions=['age', 'gender', 'race', 'emotion'], enforce_detection=False)
#         return results
#     else:
#         return None
#
# def suggest_styling(attributes):
#     recommendations = []
#
#     # Gender-based recommendations
#     if attributes['gender'] == 'Man':
#         recommendations.append("Try a clean-shaven look or a well-groomed beard.")
#         recommendations.append("Consider using a matte finish for skincare products.")
#         recommendations.append("For clothing, try neutral colors like black, grey, or navy.")
#     else:
#         recommendations.append("Experiment with different lipstick shades to find what suits you best.")
#         recommendations.append("Consider using a foundation that matches your skin tone.")
#         recommendations.append("For clothing, try pastel colors like light pink, lavender, or baby blue.")
#
#     # Emotion-based recommendations
#     if attributes['dominant_emotion'] == 'happy':
#         recommendations.append("Your smile is your best accessory, keep smiling!")
#     elif attributes['dominant_emotion'] == 'sad':
#         recommendations.append("Brighten up your look with some bold accessories.")
#
#     # Skin tone-based clothing color recommendations
#     dominant_race = attributes['dominant_race'].lower()
#     if dominant_race == 'white':
#         recommendations.append("Clothing Color: Try bold colors like red, blue, and emerald green.")
#     elif dominant_race == 'black':
#         recommendations.append("Clothing Color: Try vibrant colors like yellow, orange, and pink.")
#     elif dominant_race == 'asian':
#         recommendations.append("Clothing Color: Try rich colors like purple, deep blues, and greens.")
#     elif dominant_race == 'indian':
#         recommendations.append("Clothing Color: Try warm colors like gold, bronze, and rich reds.")
#     else:
#         recommendations.append("Clothing Color: Experiment with various shades to see what complements your skin tone.")
#
#     # Placeholder for face structure analysis
#     # Note: DeepFace does not provide face structure, so you may need a custom model for detailed face shape analysis
#     face_structure = "Oval"  # Example placeholder
#
#     if face_structure == 'Oval':
#         recommendations.append("Hair Style: Try layers or waves, most hairstyles suit you.")
#         recommendations.append("Makeup: Highlight cheekbones and use bold lip colors.")
#     elif face_structure == 'Round':
#         recommendations.append("Hair Style: Try long, straight hair to elongate your face.")
#         recommendations.append("Makeup: Contour to create definition and use natural lip colors.")
#     elif face_structure == 'Square':
#         recommendations.append("Hair Style: Try soft, wavy hair to soften the angles.")
#         recommendations.append("Makeup: Highlight cheekbones and use bold eye makeup.")
#     elif face_structure == 'Heart':
#         recommendations.append("Hair Style: Try chin-length bobs or shoulder-length waves.")
#         recommendations.append("Makeup: Focus on eyes and use lighter shades for lips.")
#
#     return recommendations
#
# def main():
#     st.title('Live Facial Styling Suggestions')
#
#     camera_type = st.radio("Select Camera Type:", ('ESP32-CAM', 'Laptop/Desktop Camera'))
#     if camera_type == 'ESP32-CAM':
#         url = st.text_input('Enter ESP32-CAM video stream URL')
#     else:
#         url = st.text_input('Enter camera number (0 for default camera)')
#
#     if not url:
#         st.warning('Please enter the camera URL or number.')
#         return
#
#     cap = load_model(url)
#     st.write('Press "Start Face Detection" to begin')
#
#     start_detection = st.button('Start Face Detection')
#
#     if start_detection:
#         st.write('Face Detection started')
#         while True:
#             ret, frame = cap.read()  # Read frame from the camera
#
#             if not ret:
#                 st.error('Error: Unable to read frame from the camera.')
#                 break
#
#             # Perform face detection
#             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#             faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
#
#             for (x, y, w, h) in faces:
#                 face = frame[y:y + h, x:x + w]
#                 attributes = analyze_face(face)
#
#                 if attributes:
#                     recommendations = suggest_styling(attributes)
#                     st.write(f"Age: {attributes['age']}")
#                     st.write(f"Gender: {attributes['gender']}")
#                     st.write(f"Race: {attributes['dominant_race']}")
#                     st.write(f"Emotion: {attributes['dominant_emotion']}")
#                     st.write("Styling Recommendations:")
#                     for rec in recommendations:
#                         st.write(f"- {rec}")
#
#                 cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
#
#             st.image(frame, channels='BGR', use_column_width=True)
#
#             unique_key = f'stop_btn_{time.time()}'  # Generate a unique key using timestamp
#             if st.button('Stop Detection', key=unique_key):
#                 st.write('Face Detection stopped')
#                 break
#
#     cap.release()
#
# if __name__ == '__main__':
#     main()
