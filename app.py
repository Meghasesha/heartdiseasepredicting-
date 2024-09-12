from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

app = Flask(__name__)

# Load data from CSV
data = pd.read_csv('heart_disease_data.csv')

# Features and target variable
X = data.drop('output', axis=1)  # Drop the 'output' column to get features
y = data['output']  # The 'output' column is the target variable

# Train a model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the model to a pickle file
pickle.dump(model, open('heart_disease_model.pkl', 'wb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract features from form submission (ensure the order matches your form in index.html)
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    # Load the trained model
    model = pickle.load(open('heart_disease_model.pkl', 'rb'))
    prediction = model.predict(final_features)

    # Interpret the result
    output = "No Heart Disease" if prediction[0] == 0 else "Heart Disease"
    return render_template('index.html', prediction_text=f'Prediction: {output}')

if __name__ == "__main__":
    app.run(debug=True)
from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

print("Starting the Flask application...")  # Debugging print

app = Flask(__name__)

print("Loading the CSV file...")  # Debugging print
data = pd.read_csv('heart_disease_data.csv')

# Features and target variable
print("Processing the data...")  # Debugging print
X = data.drop('output', axis=1)  # Drop the 'output' column to get features
y = data['output']  # The 'output' column is the target variable

# Train a model
print("Training the model...")  # Debugging print
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the model to a pickle file
print("Saving the model...")  # Debugging print
pickle.dump(model, open('heart_disease_model.pkl', 'wb'))

@app.route('/')
def home():
    print("Rendering home page...")  # Debugging print
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    print("Processing prediction...")  # Debugging print
    # Extract features from form submission
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    # Load the trained model
    model = pickle.load(open('heart_disease_model.pkl', 'rb'))
    prediction = model.predict(final_features)

    # Interpret the result
    output = "No Heart Disease" if prediction[0] == 0 else "Heart Disease"
    return render_template('index.html', prediction_text=f'Prediction: {output}')

if __name__ == "__main__":
    print("Running the Flask app...")  # Debugging print
    app.run(debug=True, host='0.0.0.0')

