# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib
import onnxruntime as ort

# Step 1: Load and preprocess the dataset
def load_and_preprocess_data(filepath):
    # Load dataset
    df = pd.read_csv(filepath)

    # Drop Patient ID
    df = df.drop("Patient ID", axis=1)

    # Encode categorical columns
    df["Sex"] = df["Sex"].map({"Female": 0, "Male": 1})
    df["Risk classification"] = df["Risk classification"].map({"High": 1, "Moderate": 1, "Low": 0})

    # Split features and target
    X = df.drop("Risk classification", axis=1)
    y = df["Risk classification"]

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Normalize numerical features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Save scaler for deployment
    joblib.dump(scaler, "hiv_scaler.pkl")

    return X_train, X_test, y_train, y_test

# Step 2: Define the PyTorch model
class HIVClassifier(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
    
    def forward(self, x):
        return self.layers(x)

# Step 3: Train the model
def train_model(X_train, y_train, X_test, y_test, epochs=50, batch_size=32):
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

    # Create DataLoader
    train_data = TensorDataset(X_train_tensor, y_train_tensor)
    test_data = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    # Initialize model
    model = HIVClassifier(X_train.shape[1])

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for inputs, labels in test_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            print(f"Epoch {epoch+1}/{epochs} | Val Accuracy: {correct/total:.4f}")
    
    return model

# Step 4: Export the model to ONNX
def export_to_onnx(model, X_train, output_path="hiv_model.onnx"):
    dummy_input = torch.randn(1, X_train.shape[1])  # Batch size 1, input features
    input_names = ["blood_features"]
    output_names = ["hiv_risk_probability"]

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={
            "blood_features": {0: "batch_size"},
            "hiv_risk_probability": {0: "batch_size"}
        },
        opset_version=11
    )
    print(f"Model exported to {output_path}")

# Step 5: Validate the ONNX model
def validate_onnx_model(onnx_path, X_test):
    ort_session = ort.InferenceSession(onnx_path)

    # Compare PyTorch and ONNX outputs
    model.eval()
    with torch.no_grad():
        pt_output = model(torch.FloatTensor(X_test[:1])).numpy()

    # ONNX prediction
    onnx_inputs = {ort_session.get_inputs()[0].name: X_test[:1].astype(np.float32)}
    onnx_output = ort_session.run(None, onnx_inputs)[0]

    print("PyTorch Output:", pt_output)
    print("ONNX Output:", onnx_output)
    assert np.allclose(pt_output, onnx_output, atol=1e-6), "Outputs mismatch!"

# Main script
if __name__ == "__main__":
    # Load and preprocess data
    X_train, X_test, y_train, y_test = load_and_preprocess_data("dataset.csv")

    # Train the model
    model = train_model(X_train, y_train, X_test, y_test)

    # Export the model to ONNX
    export_to_onnx(model, X_train)

    # Validate the ONNX model
    validate_onnx_model("hiv_model.onnx", X_test)