from retinaface import RetinaFace
import cv2

# Open the camera
cap = cv2.VideoCapture(0)

while True:
    # Capture a frame
    ret, frame = cap.read()
    if not ret:
        break

    # frame = cv2.imread("noface.jpg")

    # Detect faces using the pretrained RetinaFace model
    faces = RetinaFace.detect_faces(frame)

    # Iterate over the detected faces and draw bounding boxes
    if faces:
        print("Face detected")
        for face in faces.values():
            x1, y1, x2, y2 = face['facial_area']
            # Draw a rectangle around the face
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    else:
        print("No face detected")

    # Show the frame with detected faces
    cv2.imshow('Face Detection', frame)

    # # Save the result to a new file if desired
    # cv2.imwrite('output_image.jpg', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
