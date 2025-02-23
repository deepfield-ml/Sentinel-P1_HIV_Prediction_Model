
# Sentinel-P1: AI-Powered HIV Risk Stratification from Blood Reports

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![Dependencies Status](https://img.shields.io/badge/Dependencies-Up%20to%20Date-brightgreen)](requirements.txt)

## Overview

Sentinel-P1 is an innovative, AI-driven solution designed to estimate an individual's risk of HIV based solely on readily available blood test reports. Unlike traditional HIV risk assessments that rely on behavioral factors, sexual history, and other self-reported data, Sentinel-P1 leverages the power of machine learning to identify subtle patterns within routine blood biomarkers that correlate with increased HIV susceptibility.  This marks a significant paradigm shift in early risk identification and opens avenues for proactive intervention.

## Key Innovations

*   **Novel Data Source:** Sentinel-P1 pioneers the use of blood test reports as the primary data source for HIV risk classification. Prior approaches have almost exclusively relied on behavioral and demographic information. This innovation offers a more objective and readily accessible approach to risk assessment.
*   **Early Risk Detection:** By identifying patterns in blood biomarkers, Sentinel-P1 has the potential to detect increased HIV susceptibility even before noticeable symptoms appear or high-risk behaviors are reported. This provides a window of opportunity for preventative measures and early testing.
*   **Objective and Accessible:** Sentinel-P1 reduces reliance on subjective self-reporting, potentially mitigating biases and inaccuracies inherent in traditional risk assessment methods. Blood tests are a common part of routine medical checkups, making this approach easily accessible to a broad population.
*   **AI-Powered Insights:** The model uses a neural network architecture, capable of capturing complex non-linear relationships between various blood biomarkers and HIV risk that might be missed by traditional statistical methods.
*   **Potential for Personalized Medicine:** As the model is refined with more data, Sentinel-P1 can contribute to a more personalized approach to HIV prevention, allowing healthcare providers to tailor interventions based on individual risk profiles derived from blood test results.

## Model Architecture

Sentinel-P1 is built upon a feedforward neural network architecture implemented using PyTorch. The model consists of the following layers:

1.  **Input Layer:**  Accepts blood biomarker data as input (Age, Sex (encoded), CD4+ T-cell count, Viral load, WBC count, Hemoglobin, Platelet count).
2.  **Hidden Layers:** Two hidden layers with ReLU activation functions and dropout regularization (20%) to prevent overfitting. The first layer contains 64 neurons, and the second contains 32 neurons.
3.  **Output Layer:** A fully connected layer outputting probabilities for each risk class (Low/Moderate Risk, High Risk) using CrossEntropyLoss and Adam optimizer.

## Data

The model was trained on a synthetic dataset generated for demonstration purposes.  While this dataset mimics real-world blood biomarker distributions, it should *not* be used for actual medical decision-making. **Real-world deployment requires training on a large, ethically sourced, and clinically validated dataset.**

The dataset includes the following features:

*   **Age:** Patient's age in years.
*   **Sex:** Patient's sex (encoded as Female: 0, Male: 1).
*   **CD4+ T-cell count:**  Number of CD4+ T-cells per microliter of blood.
*   **Viral load:**  Number of HIV RNA copies per milliliter of blood.
*   **WBC count:** White blood cell count per microliter of blood.
*   **Hemoglobin:** Hemoglobin concentration in grams per deciliter.
*   **Platelet count:** Platelet count per microliter of blood.
*   **Risk classification:** Target variable indicating HIV risk level (Low/Moderate: 0, High: 1).

## Getting Started

### Prerequisites

*   Python 3.7+
*   pip package manager

### Installation

1.  Clone the repository:

    ```bash
    git clone https://github.com/[your_username]/[your_repository].git
    cd [your_repository]
    ```

2.  Create a virtual environment (recommended):

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```

3.  Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

### Running the Evaluation App

1.  **Ensure you have the `hiv_model.onnx` and `hiv_scaler.pkl` files generated from the training script (after modifying it to include `Sex` in scaling).**

2.  Execute the GUI application:

    ```bash
    python hiv_evaluation_app.py
    ```

3.  **Using the GUI:**
    *   Click the "Load ONNX Model" button and select the `hiv_model.onnx` file.
    *   Click the "Load Scaler" button and select the `hiv_scaler.pkl` file.
    *   Enter patient data into the input fields.
    *   Click the "Predict Risk" button to obtain the high-risk probability.

## Code Structure

*   `hiv_evaluation_app.py`:  The main Python script containing the Tkinter-based GUI application for evaluating the ONNX model.
*   `requirements.txt`:  Lists the Python packages required to run the application.
*   `hiv_model.onnx`: (Generated by training script) The ONNX-format model file.
*   `hiv_scaler.pkl`: (Generated by training script) The `StandardScaler` object saved as a pickle file.
*   `dataset.csv`: (Example) The CSV file containing synthetic patient data.

## Usage

The `hiv_evaluation_app.py` script provides a user-friendly interface for evaluating the Sentinel-P1 model.  Users can input patient data and obtain a probability score indicating the likelihood of high HIV risk. This score can be used as a preliminary indicator and should be considered alongside other clinical factors.

## Training the Model (Important: Requires Modification)

**Critical:** The provided training script needs to be modified as follows to ensure the `StandardScaler` is correctly trained:

1.  **Include 'Sex' in Scaling:** Modify the training script to numerically encode 'Sex' *before* fitting the `StandardScaler`. The scaler should be fitted on *all* features, including the encoded 'Sex' column.

    ```python
    # In your training script (before exporting to ONNX):

    # Encode categorical columns
    df["Sex"] = df["Sex"].map({"Female": 0, "Male": 1})
    df["Risk classification"] = df["Risk classification"].map({"High": 1, "Moderate": 1, "Low": 0})

    # Split features and target
    X = df.drop("Risk classification", axis=1)
    y = df["Risk classification"]

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Normalize numerical features (now includes Sex)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)  # Fit on ALL columns
    X_test = scaler.transform(X_test)        # Transform ALL columns

    # Save scaler for deployment
    joblib.dump(scaler, "hiv_scaler.pkl")
    ```

2.  **Run the Training Script:**  After modifying the training script, execute it to generate the `hiv_model.onnx` and `hiv_scaler.pkl` files.

3.  **Use the New Files in the App:**  Replace the old `hiv_model.onnx` and `hiv_scaler.pkl` files with the newly generated ones in your app directory.

## Model Validation

The `validate_onnx_model` function in the original training script validates the exported ONNX model by comparing its output to the PyTorch model's output for a sample input.  This ensures that the ONNX conversion process was successful and that the model is performing as expected.

## Ethical Considerations

*   **Data Privacy:** Patient data must be handled with the utmost care and in compliance with all applicable privacy regulations (e.g., HIPAA, GDPR).
*   **Bias Mitigation:**  The training data should be carefully reviewed for potential biases that could lead to unfair or discriminatory outcomes.
*   **Transparency and Explainability:** Efforts should be made to understand and explain the model's decision-making process to healthcare professionals and patients.
*   **Clinical Validation:** **Sentinel-P1 is currently a proof-of-concept and must undergo rigorous clinical validation on diverse patient populations before being used in real-world clinical settings.**

## Future Directions

*   **Real-World Data Integration:** Training the model on a large, clinically validated dataset from diverse patient populations is a critical next step.
*   **Feature Engineering:** Exploring additional blood biomarkers and other relevant clinical data to improve model accuracy.
*   **Explainable AI (XAI):** Implementing techniques to enhance the model's interpretability and provide insights into which biomarkers are most influential in risk prediction.
*   **Integration with EHR Systems:** Developing seamless integration with electronic health record (EHR) systems to facilitate automated risk assessment.
*   **Longitudinal Analysis:**  Analyzing changes in blood biomarkers over time to track disease progression and personalize interventions.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

**Sentinel-P1 is intended for research purposes only and should not be used for self-diagnosis or treatment. Consult with a qualified healthcare professional for any health concerns.** The developers of Sentinel-P1 are not responsible for any decisions made based on the output of this model. The information provided by this model is not a substitute for professional medical advice.
