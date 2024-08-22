from numpy import loadtxt
from keras.models import model_from_json
import numpy as np

# Load model architecture
try:
    with open('model.json', 'r') as json_file:
        loaded_model_json = json_file.read()
except FileNotFoundError:
    print("Model architecture file not found.")
    exit()

# Load model from JSON
try:
    model = model_from_json(loaded_model_json)
except Exception as e:
    print(f"Error loading model from JSON: {e}")
    exit()

# Load model weights
try:
    model.load_weights("model_weights.weights.h5")
except FileNotFoundError:
    print("Model weights file not found.")
    exit()
except Exception as e:
    print(f"Error loading model weights: {e}")
    exit()

print("Loaded model from disk")

# Function to take user input and predict
def predict_new_data():
    features = []
    print("Please enter the 8 features for prediction:")

    # Collect user input with error handling
    for i in range(8):
        while True:
            try:
                value = float(input(f"Feature {i+1}: "))
                features.append(value)
                break
            except ValueError:
                print("Invalid input. Please enter a numeric value.")

    # Convert features to a numpy array
    features_array = np.array(features).reshape(1, -1)  # Reshape for a single prediction

    # Predict
    try:
        prediction = model.predict(features_array)
        prediction_class = (prediction > 0.5).astype(int)
        print(f"Predicted class: {int(prediction_class[0][0])}")
    except Exception as e:
        print(f"Error during prediction: {e}")

# Call the function to make a prediction
predict_new_data()
