from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load and train the machine learning model
def train_model():
    # Load the sanitation data
    data = pd.read_csv('sanitation_progress_2007_2012.csv')
    
    # Preprocess the data (for simplicity, using Total IHHL and School Toilets for prediction)
    X = data[['Total IHHL', 'School Toilets']]
    y = data['School Toilets'].apply(lambda x: 'Requires Sanitation' if x < 1 else 'Sufficient Sanitation')

    # Label encode the target variable
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a decision tree classifier
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    return clf, le

# Load and train the model once
clf, label_encoder = train_model()

# Define the route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Handle file upload and prediction
@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file uploaded"

    file = request.files['file']
    if file.filename == '':
        return "No file selected"
    
    # Read the uploaded CSV file
    data = pd.read_csv(file)

    # Extract the relevant columns for prediction
    X_new = data[['Total IHHL', 'School Toilets']]

    # Make predictions
    predictions = clf.predict(X_new)
    results = label_encoder.inverse_transform(predictions)

    # For simplicity, we'll just display the result for the first school
    result = results[0]

    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
