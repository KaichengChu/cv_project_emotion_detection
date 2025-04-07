import cv2
import time
import os

new_path = r"C:\Users\chuka\Desktop\CS4501\project\real_time_face"
os.chdir(new_path)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

webcam = cv2.VideoCapture(0)

snapshot_count = 0
last_saved_time = 0
save_interval = 3

while True:
    ret, img = webcam.read()

    if not ret:
        print("Failed to grab frame")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=6,
        minSize=(50, 50),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    current_time = time.time()

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

        # Save face snapshot
        if current_time - last_saved_time >= save_interval:
            face_roi = img[y:y + h, x:x + w]
            snapshot_filename = os.path.join(new_path,  f"face_snapshot_{snapshot_count}.png")
            cv2.imwrite(snapshot_filename, face_roi)
            print(f"Snapshot saved as {snapshot_filename}")
            snapshot_count += 1
            last_saved_time = current_time

    cv2.imshow("Face Detection", img)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

webcam.release()
cv2.destroyAllWindows()
