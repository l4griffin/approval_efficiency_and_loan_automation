import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC




def run_ml_pipeline(filepath="C:/Users/griff/OneDrive/Documentos/Machine Learning/END Sem Project/Merged_Dataset.csv"):
    print("Loading and preparing data...")
    # 1. Training Data Construction
    df = pd.read_csv(filepath)
    
    df = df.dropna(subset=['Default_Status'])
    df = df.drop(['country', 'year', 'uniqueid'], axis=1)

    # Feature Selection
    X = df.drop('Default_Status', axis=1) # Features
    y = df['Default_Status'] # Target variable

    # Encoding of categorical demographic variables
    X = pd.get_dummies(X, drop_first=True)

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2. Machine Learning Implementation
    models = {
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "ANN": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42),
        "SVM": SVC(kernel='linear', random_state=42)
    }

    # 3. Model Evaluation and Comparison
    print("\n--- 2-Fold Cross-Validation Accuracy Results ---")
    results = {}
    
    for name, model in models.items():
        
        scores = cross_val_score(model, X_scaled, y, cv=2, scoring='accuracy')
        mean_accuracy = np.mean(scores)
        results[name] = mean_accuracy
        print(f"{name}: {mean_accuracy * 100:.2f}%")

if __name__ == "__main__":

    try:
        run_ml_pipeline('Merged_Dataset.csv')
    except FileNotFoundError:
        print("Error: 'Merged_Dataset.csv' not found. Please check the file path.")