# 🧠 Breast Cancer Classification Neural Network 🏥

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)



## 📝 Project Description

Neural network classifier for breast cancer diagnosis using scikit-learn's breast cancer dataset. This implementation achieves >98% accuracy in classifying tumors as malignant or benign.

## 🏗️ Model Architecture

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(64, input_shape=(30,), activation='relu'),
    Dense(32, activation='relu'), 
    Dense(16, activation='relu'),
    Dense(2, activation='softmax')
])
```

## 🛠️ Installation
```python
pip install numpy pandas scikit-learn tensorflow matplotlib
```

## 📊 Dataset Features (Scikit-learn)
```python
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
print(cancer.DESCR)
```

Dataset characteristics:

Samples: 569 (212 malignant, 357 benign)
Features: 30 numeric predictive features
Target: Binary classification (0=malignant, 1=benign)

## 🚀 Usage
1. Preprocess data:
```
df = pd.DataFrame(data=cancer["data"], 
                 columns=cancer["feature_names"])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
```
2. Train model:
```
history = model.fit(
    X_train, y_train,
    epochs=60,
    batch_size=32,
    validation_data=(X_test, y_test)
)
```

3. Evaluate:
```
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Accuracy: {accuracy*100:.2f}%')
```

## 📂 Project Structure
/project
│   README.md
│   requirements.txt
│   cancer_classifier.py
│
└───/models
    │   best_model.h5
│   
└───/notebooks
    │   EDA.ipynb




