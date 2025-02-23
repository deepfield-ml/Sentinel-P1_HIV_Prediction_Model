import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import joblib
import onnxruntime as ort
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler  # Import StandardScaler

# Define the HIVClassifier model (same as in the training script)
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


class ModelEvaluationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("HIV Risk Prediction - Model Evaluation")

        # Load scaler and ONNX model (initialize to None)
        self.scaler = None
        self.ort_session = None
        self.feature_names = ['Age', 'Sex', 'CD4+ T-cell count', 'Viral load', 'WBC count', 'Hemoglobin', 'Platelet count']
        self.model_loaded = False
        self.scaler_loaded = False
        # UI elements
        self.create_widgets()

    def create_widgets(self):
        # Load Model Button
        self.load_model_button = ttk.Button(self.root, text="Load ONNX Model", command=self.load_model)
        self.load_model_button.pack(pady=10)

        # Load Scaler Button
        self.load_scaler_button = ttk.Button(self.root, text="Load Scaler", command=self.load_scaler)
        self.load_scaler_button.pack(pady=10)
        # Input fields (Age, Sex, CD4+ T-cell count, Viral load, WBC count, Hemoglobin, Platelet count)
        self.input_frame = ttk.Frame(self.root)
        self.input_frame.pack(pady=10)

        self.input_vars = {}  # Dictionary to hold input variables
        for i, feature in enumerate(self.feature_names):
            label = ttk.Label(self.input_frame, text=f"{feature}:")
            label.grid(row=i, column=0, padx=5, pady=5, sticky=tk.W)
            if feature == "Sex":
                self.input_vars[feature] = tk.StringVar()
                sex_combobox = ttk.Combobox(self.input_frame, textvariable=self.input_vars[feature], values=["Female", "Male"])
                sex_combobox.grid(row=i, column=1, padx=5, pady=5, sticky=tk.E)
                sex_combobox.set("Female")  # Default value
            else:
                self.input_vars[feature] = tk.StringVar()
                entry = ttk.Entry(self.input_frame, textvariable=self.input_vars[feature])
                entry.grid(row=i, column=1, padx=5, pady=5, sticky=tk.E)

        # Predict Button
        self.predict_button = ttk.Button(self.root, text="Predict Risk", command=self.predict_risk, state=tk.DISABLED)
        self.predict_button.pack(pady=10)

        # Output Label
        self.output_label = ttk.Label(self.root, text="Prediction will appear here")
        self.output_label.pack(pady=10)

    def load_model(self):
        filepath = filedialog.askopenfilename(title="Select ONNX Model", filetypes=[("ONNX files", "*.onnx")])
        if filepath:
            try:
                self.ort_session = ort.InferenceSession(filepath)
                self.model_loaded = True
                messagebox.showinfo("Model Loaded", "ONNX model loaded successfully.")
                self.update_predict_button_state()
            except Exception as e:
                messagebox.showerror("Error Loading Model", f"Error loading model: {e}")

    def load_scaler(self):
        filepath = filedialog.askopenfilename(title="Select Scaler", filetypes=[("Pickle files", "*.pkl")])
        if filepath:
            try:
                self.scaler = joblib.load(filepath)
                self.scaler_loaded = True
                messagebox.showinfo("Scaler Loaded", "Scaler loaded successfully.")
                self.update_predict_button_state()
            except Exception as e:
                messagebox.showerror("Error Loading Scaler", f"Error loading scaler: {e}")

    def update_predict_button_state(self):
        if self.model_loaded and self.scaler_loaded:
            self.predict_button.config(state=tk.NORMAL)
        else:
            self.predict_button.config(state=tk.DISABLED)

    def predict_risk(self):
        try:
            # 1. Data Validation and Conversion
            input_data = {}
            for feature in self.feature_names:
                value = self.input_vars[feature].get()
                if not value:
                    raise ValueError(f"Please enter a value for {feature}.")

                if feature == "Sex":
                    if value not in ["Female", "Male"]:
                        raise ValueError("Sex must be 'Female' or 'Male'.")
                    input_data[feature] = 0 if value == "Female" else 1  # Encode Sex
                else:
                    try:
                        input_data[feature] = float(value)
                    except ValueError:
                        raise ValueError(f"Invalid input for {feature}.  Must be a number.")

            # Create DataFrame
            input_df = pd.DataFrame([input_data])  # DataFrame with a single row

            # Standardize the input data using the loaded scaler
            # NO need to drop Sex anymore!
            scaled_values = self.scaler.transform(input_df[self.feature_names]) # Scale ALL features.

            # Create a DataFrame from the scaled values
            scaled_df = pd.DataFrame(scaled_values, columns=self.feature_names)  #Use ALL feature names now.



            # Ensure the column order matches the training data
            input_array = scaled_df[self.feature_names].values.astype(np.float32)

            # 3. ONNX Prediction
            ort_inputs = {self.ort_session.get_inputs()[0].name: input_array}
            ort_outs = self.ort_session.run(None, ort_inputs)

            # 4.  Process Output (Probability)
            probabilities = ort_outs[0][0]  # Get the probabilities for both classes
            risk_probability = probabilities[1]  # Probability of high risk (class 1)

            if 0 <= risk_probability <= 100:
                self.output_label.config(text=f"HIV Probability: {risk_probability:.4f}")
            elif risk_probability < 0:
                self.output_label.config(text=f"HIV Probability: 0")
            elif risk_probability > 100:
                self.output_label.config(text=f"HIV Probability: 100")

        except ValueError as e:
            messagebox.showerror("Input Error", str(e))
        except Exception as e:
            messagebox.showerror("Prediction Error", f"An error occurred during prediction: {e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = ModelEvaluationApp(root)
    root.mainloop()
