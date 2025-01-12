import numpy as np
from flask import Flask, request, render_template
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from io import BytesIO  # To convert the file to a stream

# Load the saved model
model = load_model('best_model.keras')

# Define the categories (class names)
categories = [
    'Apple', 'Banana', 'Broccoli', 'Carrots', 'Cauliflower', 'Chili',
    'Coconut', 'Cucumber', 'Custard Apple', 'Dates', 'Dragon fruit',
    'Eggplant', 'Garlic', 'Grape', 'Green lemon', 'Jackfruit', 'Kiwi',
    'Mango', 'Okra', 'Onion', 'Orange', 'Papaya', 'Peanut', 'Pineapple',
    'Pomegranate', 'Star Fruit', 'Strawberry', 'Sweet Potato', 'Watermelon',
    'White mushroom'
]

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict')
def predict_page():
    return render_template('predict.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the uploaded image
        f = request.files['image']
        if not f:
            return render_template('output.html', prediction="Error: No file uploaded.")

        # Convert the file to a BytesIO object
        img = load_img(BytesIO(f.read()), target_size=(224, 224))  # Resize to 224x224
        x = img_to_array(img)
        x = x / 255.0  # Normalize pixel values to range [0, 1]

        # Add batch dimension
        x = np.expand_dims(x, axis=0)

        # Make predictions
        prediction = np.argmax(model.predict(x), axis=1)
        predicted_class = categories[prediction[0]] if prediction[0] < len(categories) else "Unknown"

        return render_template('output.html', prediction=predicted_class)

    except Exception as e:
        return render_template('output.html', prediction=f"Error: {e}")

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
