import cv2

# Use OpenCV's built-in path to the Haar cascade XML
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_haar_cascade = cv2.CascadeClassifier(cascade_path)

# Initialize video capture from the default webcam
capture = cv2.VideoCapture(0)

# Check that the video capture and cascade loaded properly
if not capture.isOpened():
    print("Error: Could not open video device.")
    exit()

if face_haar_cascade.empty():
    print("Error: Failed to load Haar cascade.")
    exit()

while True:
    ret, image = capture.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Face Detection", image)

    if cv2.waitKey(30) & 0xFF == 27:  # Exit on ESC key
        break

# Release everything properly
capture.release()
cv2.destroyAllWindows()
