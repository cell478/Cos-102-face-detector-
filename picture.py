import cv2

face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

image = cv2.imread(r"C:\Users\Daniel Terwase Ajayi\PycharmProjects\pythonProject33\Execl.jpg")
image = cv2.resize(image, (600, 400))  # Resize for consistent detection
grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

face = face_haar_cascade.detectMultiScale(grey, scaleFactor=1.05, minNeighbors=3)

for (x, y, w, h) in face:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 5)

cv2.imshow("Face", image)
cv2.waitKey(0)
cv2.destroyAllWindows()


