import cv2

# Load the pre-trained Haar Cascade Classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the image
image_path = 'images.jpg'  # Replace with your image file path
image = cv2.imread(image_path)

# Check if the image was loaded successfully
if image is None:
    print("Error: Image not found.")
    exit()

# Convert the image to grayscale (Haar Cascade works better on grayscale images)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Draw rectangles around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue rectangle

# Display the image with face detection results
cv2.imshow('Face Detection', image)

# Save the result to a new file
cv2.imwrite('output_image.jpg', image)

# Wait for any key press to close the window
cv2.waitKey(0)
cv2.destroyAllWindows()
