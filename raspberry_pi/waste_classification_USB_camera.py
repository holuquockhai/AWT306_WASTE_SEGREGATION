import time
import numpy as np
import cv2
import tensorflow as tf
from picamera2 import Picamera2

# Load the TensorFlow Lite model
MODEL_PATH = "awt306_model.tflite"  # Path to your TFLite file
LABELS = ['glass', 'metal', 'paper', 'plastic']  # Modify based on your classes
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)

print(interpreter.get_signature_list())

classify_lite = interpreter.get_signature_runner('serving_default')
print(classify_lite)

# Get input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

import time
import numpy as np
import cv2
import tensorflow as tf
from picamera2 import Picamera2

# Load the TensorFlow Lite model
MODEL_PATH = "awt306_model.tflite"  # Path to your TFLite file
LABELS = ['glass', 'metal', 'paper', 'plastic', 'unknown']  # Modify based on your classes
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)

print(interpreter.get_signature_list())

classify_lite = interpreter.get_signature_runner('serving_default')
print(classify_lite)

# Get input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Open the default camera
cam = cv2.VideoCapture(8)

# Get the default frame width and height
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
#fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))

# Function to preprocess the image
def preprocess_image(frame, input_shape):
	image = cv2.resize(frame, (input_shape[1], input_shape[2]))  # Resize to model input size

	img_array = tf.keras.utils.img_to_array(image)
	img_array = tf.expand_dims(img_array, 0) # Create a batch
	# img_array = img_array/255.0
	return img_array

# Function to perform inference and get predictions
def predict(image):
	input_shape = input_details[0]['shape']
	preprocessed_image = preprocess_image(image, input_shape)
	predictions_lite = classify_lite(keras_tensor_329=preprocessed_image)['output_0']
	score_lite = tf.nn.softmax(predictions_lite)
	return score_lite

# Main loop for real-time image classification
try:
	while True:
		
		# Capture frame from the camera
		ret, frame = cam.read()

		image = cv2.flip(frame, 1)

		# Convert the image from BGR to RGB as required by the TFLite model.
		rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

		# Perform inference
		predictions = predict(rgb_image)
		class_name = LABELS[np.argmax(predictions)]
		confidence = 100 * np.max(predictions)
		if confidence >= 25:
			# Overlay prediction on the camera feed
			text = f"{class_name}: {confidence:.2f}%"
		else:
			text = "No class identified"

		cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

		# Display the frame with prediction
		cv2.imshow("Waste Classification", frame)

		# Exit on pressing 'q'
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

except KeyboardInterrupt:
    print("Stopped by User")

finally:
    # Cleanup
    cam.release()
    cv2.destroyAllWindows()
