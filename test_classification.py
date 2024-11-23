import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import time

# Path to your TensorFlow Lite model
MODEL_PATH = "model.tflite"

# Classes (update these with your waste categories)
CLASSES = ['Plastic', 'Paper', 'Metal', 'Organic']

# Load the TensorFlow Lite model
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Get input and output details of the model
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Function to preprocess the input image
def preprocess_image(image, input_shape):
    # Resize the image to the model's expected input size
    img = cv2.resize(image, (input_shape[1], input_shape[2]))
    # Normalize pixel values to [0, 1]
    img = img.astype(np.float32) / 255.0
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    return img

# Function to classify the image
def classify_image(image):
    # Preprocess the image
    input_shape = input_details[0]['shape']
    input_data = preprocess_image(image, input_shape)

    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    start_time = time.time()
    interpreter.invoke()
    end_time = time.time()

    # Get the output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    class_index = np.argmax(output_data)
    confidence = output_data[0][class_index]

    # Display classification results
    print(f"Class: {CLASSES[class_index]} | Confidence: {confidence:.2f} | Time: {end_time - start_time:.3f}s")
    return CLASSES[class_index], confidence

# Initialize the camera
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Could not access the camera.")
    exit()

print("Starting real-time image classification. Press 'q' to quit.")

while True:
    # Capture a frame from the camera
    ret, frame = camera.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Display the frame
    cv2.imshow("Camera Feed", frame)

    # Perform classification
    waste_type, confidence = classify_image(frame)

    # Add classification result on the frame
    cv2.putText(frame, f"{waste_type} ({confidence:.2f})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame with the classification result
    cv2.imshow("Classified Frame", frame)

    # Quit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
camera.release()
cv2.destroyAllWindows()
