import socketio
import eventlet
import numpy as np
from flask import Flask
from tensorflow.keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import cv2
from tensorflow.keras.utils import register_keras_serializable
import tensorflow as tf

# Register the mse metric, if required
@register_keras_serializable()
def mse(y_true, y_pred):
    return tf.keras.losses.mean_squared_error(y_true, y_pred)

# Initialize the Socket.IO server
sio = socketio.Server()

# Initialize the Flask app
app = Flask(__name__)  # '__main__'

# Define speed limit
speed_limit = 10

# Image preprocessing function
def img_preprocess(img):
    img = img[60:135, :, :]  # Crop the image to remove irrelevant parts
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)  # Convert to YUV color space
    img = cv2.GaussianBlur(img, (3, 3), 0)  # Apply Gaussian blur to reduce noise
    img = cv2.resize(img, (200, 66))  # Resize to the model input shape
    img = img / 255.0  # Normalize pixel values
    return img

# Fine-tune the steering angle for stability
def adjust_steering(steering_angle, previous_angle, smoothing_factor=0.2):
    return smoothing_factor * steering_angle + (1 - smoothing_factor) * previous_angle

# Initialize the previous steering angle
previous_steering_angle = 0

# Socket.IO telemetry event listener
@sio.on('telemetry')
def telemetry(sid, data):
    global previous_steering_angle
    try:
        # Parse telemetry data
        speed = float(data['speed'])
        image = Image.open(BytesIO(base64.b64decode(data['image'])))
        image = np.asarray(image)
        image = img_preprocess(image)  # Preprocess the image
        image = np.array([image])

        # Predict the steering angle
        steering_angle = float(model.predict(image)[0][0])  # Use correct indexing for predictions
        steering_angle = adjust_steering(steering_angle, previous_steering_angle)  # Smooth steering angle
        previous_steering_angle = steering_angle

        # Restrict steering angle to [-1, 1]
        steering_angle = max(min(steering_angle, 1.0), -1.0)

        # Compute throttle
        throttle = max(0.3, 1.0 - speed / speed_limit)  # Ensure minimum throttle

        # Log telemetry data
        print(f'Steering Angle: {steering_angle:.4f}, Throttle: {throttle:.4f}, Speed: {speed:.2f}')

        # Send control commands
        send_control(steering_angle, throttle)
    except Exception as e:
        print(f"Error processing telemetry data: {e}")

# Socket.IO connect event listener
@sio.on('connect')
def connect(sid, environ):
    print('Client connected')
    send_control(0, 0)  # Initialize with zero steering and throttle

# Function to send control commands
def send_control(steering_angle, throttle):
    sio.emit('steer', data={
        'steering_angle': str(steering_angle),
        'throttle': str(throttle)
    })

if __name__ == '__main__':
    try:
        # Load the pre-trained model
        model = load_model('model/model.h5', custom_objects={'mse': mse}, compile=False)

        # Wrap Flask app with Socket.IO middleware
        app = socketio.Middleware(sio, app)

        # Start the server
        print("Starting server on port 4567...")
        eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
    except OSError as e:
        print(f"OSError: {e}. Port 4567 may already be in use.")
    except Exception as e:
        print(f"Error: {e}")
