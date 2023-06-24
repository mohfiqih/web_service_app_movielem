from keras.models import load_model
import cv2
import numpy as np

# Load the model
model = load_model("model/image/keras_model.h5", compile=False)

# Load the labels
class_names = open("model/image/labels.txt", "r").readlines()

# Initialize the webcam stream
stream_url = "http://10.36.169.99:8080/video"
# stream_url = "video/vid_testing.mp4"
camera = cv2.VideoCapture(stream_url)

# Load the face cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    # Read the next frame from the stream
    ret, frame = camera.read()

    # Check if the frame is valid
    if not ret:
        print("Failed to retrieve frame from the stream.")
        break

    # Resize the frame to half its size
    width = int(frame.shape[1] * 0.5)
    height = int(frame.shape[0] * 0.5)
    resized_frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

    # Convert the frame to grayscale
    gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract the face region and resize it to (224, 224)
        face = resized_frame[y:y+h, x:x+w]
        resized_face = cv2.resize(face, (224, 224), interpolation=cv2.INTER_AREA)

        # Make the resized face a numpy array and reshape it to the model's input shape
        image = np.asarray(resized_face, dtype=np.float32).reshape(1, 224, 224, 3)

        # Normalize the image array
        image = (image / 127.5) - 1

        # Predict the model
        prediction = model.predict(image)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        # Draw the bounding box around the face
        cv2.rectangle(resized_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display the label near the bounding box
        label = f"{class_name[2:]}: {np.round(confidence_score * 100)}%"
        cv2.putText(resized_frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        print(label)

    # Show the frame in a window
    cv2.imshow("Webcam Image", resized_frame)

    # Listen to the keyboard for presses.
    keyboard_input = cv2.waitKey(1)

    # 'q' key to exit the program
    if keyboard_input == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
