import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

try:
    # Load the dataset
    print("Loading dataset...")
    df = pd.read_csv('dataset.csv')
    print(f"Dataset loaded successfully. Shape: {df.shape}")

    # Prepare features and target
    x = df.iloc[:, :-1]
    y = df.iloc[:,-1]
    print(f"Features shape: {x.shape}, Target shape: {y.shape}")

    # Split the data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=100)
    print(f"Training set shape: {x_train.shape}, Test set shape: {x_test.shape}")

    # Create preprocessing pipeline
    trf = ColumnTransformer([
        ('trf', OneHotEncoder(sparse_output=False, drop='first'), ['batting_team', 'bowling_team', 'city'])
    ], remainder='passthrough')

    # Create and train the model pipeline
    ra_pipe = Pipeline([
        ('step1', trf),
        ('step2', RandomForestClassifier(random_state=42))
    ])

    print("Training the model...")
    ra_pipe.fit(x_train, y_train)

    # Make predictions and evaluate
    ra_y_pred = ra_pipe.predict(x_test)
    accuracy = accuracy_score(y_test, ra_y_pred)
    print(f"Model accuracy: {accuracy:.4f}")

    # Save the model
    print("Saving the model...")
    with open('ra_pipe.pkl', 'wb') as f:
        pickle.dump(ra_pipe, f)
    
    # Verify the model was saved
    if os.path.exists('ra_pipe.pkl'):
        print("Model saved successfully!")
    else:
        print("Error: Model file was not saved properly")

except Exception as e:
    print(f"An error occurred: {str(e)}")