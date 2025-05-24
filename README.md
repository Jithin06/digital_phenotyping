````markdown
# EmoSurv: Digital Phenotyping with Typing Dynamics

This project explores emotion recognition using **keystroke dynamics** as part of a digital phenotyping system. It uses a deep neural network to classify user emotions based on low-level typing features like key hold times and inter-key delays.

## 📌 Project Overview

- **Goal**: Predict emotional state (e.g., happy, sad, angry) from typing patterns using supervised learning.
- **Input**: Typing features including key hold duration (`keyUp - keyDown`) and inter-key delay (`D1D2`).
- **Output**: Predicted emotion label.
- **Model**: Deep feedforward neural network with dropout, batch normalization, and ReLU activation.

---

## 📊 Dataset

The dataset used is `FixedTextTypingDataset.csv`, which includes:

- `keyDown`: Time a key was pressed.
- `keyUp`: Time the key was released.
- `D1D2`: Delay between consecutive keypresses.
- `Emotion`: The labeled emotional category for each sample.

During preprocessing:
- Non-numeric and invalid entries are converted to `NaN` and dropped.
- Features are standardized using `StandardScaler`.

---

## 🧠 Model Architecture

The neural network is defined as:

```python
nn.Sequential(
    nn.Linear(2, 64),
    nn.BatchNorm1d(64),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(64, 128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, num_classes)
)


* **Input size**: 2 (key hold duration and inter-key delay)
* **Output size**: Number of emotion classes
* **Loss function**: Cross-Entropy Loss
* **Optimizer**: Adam

---

## 📈 Training & Evaluation

* Trained for 20 epochs.
* Tracks training loss and test accuracy.
* Evaluation includes:

  * Loss/accuracy plots.
  * Confusion matrix for prediction visualization.

Example:
![Accuracy Plot](example_accuracy.png)
![Confusion Matrix](example_confusion_matrix.png)

---

## 🚀 Usage

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/emosurv.git
cd emosurv
```

### 2. Install dependencies

Create a conda environment or install using pip:

```bash
pip install -r requirements.txt
```

### 3. Run the notebook

Open `emosurv_dataset_updated.ipynb` in Jupyter and run all cells:

```bash
jupyter notebook emosurv_dataset_updated.ipynb
```

---

## 📦 Dependencies

* `pandas`
* `numpy`
* `torch`
* `sklearn`
* `matplotlib`
* `seaborn`

---

## 📚 References

* Digital phenotyping literature.
* Emotion classification using keystroke dynamics.
* Feature engineering in behavioral biometrics.

---

## 🧑‍💻 Author

**Jithin Rajan**
Undergraduate at NIT Trichy
Major: ECE | Minor: CSE

---

## 📝 License

This project is for academic and research use. For commercial or clinical use, contact the author.

```
````
