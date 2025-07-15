# 🧠 Fashion MNIST Classifier using Artificial Neural Network (ANN)

This project demonstrates the implementation of an Artificial Neural Network (ANN) to classify grayscale images of clothing items from the Fashion MNIST dataset using Pytorch.

---

## 📊 Dataset: Fashion MNIST

Fashion MNIST is a dataset of Zalando's article images—consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes:

| Label | Description    |
|-------|----------------|
| 0     | T-shirt/top    |
| 1     | Trouser        |
| 2     | Pullover       |
| 3     | Dress          |
| 4     | Coat           |
| 5     | Sandal         |
| 6     | Shirt          |
| 7     | Sneaker        |
| 8     | Bag            |
| 9     | Ankle boot     |

---

## 🧠 Model Architecture

The ANN model is designed with the following architecture:

- Input Layer: Flattens 28×28 image to 784 (input_dim)
- Hidden Layer 1: Linear(784 → 128) → BatchNorm → ReLU → Dropout(0.3)
- Hidden Layer 2: Linear(128 → 64) → BatchNorm → ReLU → Dropout(0.3)
- Output Layer: Linear(64 → 10) for classification into 10 classes
- Activation Function: ReLU (hidden layers), no activation after output
- Loss Function: nn.CrossEntropyLoss()
- Optimizer: torch.optim.SGD with weight decay
- Training Epochs: 100

## 🚀 How to Run

Follow these steps to train and evaluate the Fashion MNIST ANN model:

### 1. ✅ Prerequisites

Make sure Python is installed, then install the required packages using pip:

```bash
pip install pytorch matplotlib numpy scikit-learn
```

### 2. 📥 Get the Notebook

You can either:

- **Clone the repository** (if hosted on GitHub):
  ```bash
  git clone https://github.com/yourusername/fashion-mnist-ann.git
  cd fashion-mnist-ann
  ```

- **Or manually download** the file: `ann_using_fashion_mnist.ipynb`

### 3. 🧪 Run the Notebook

You can use Jupyter Notebook or Google Colab to run the project:

#### ✅ Option A: Run Locally using Jupyter Notebook

1. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
2. Open `ann_using_fashion_mnist.ipynb`
3. Run each cell in order

#### ✅ Option B: Run on Google Colab

1. Visit [https://colab.research.google.com](https://colab.research.google.com)
2. Upload the notebook
3. Click **Runtime → Run all**

---

## 📈 Evaluation & Metrics

The notebook includes:

- Accuracy and loss curves over epochs
- Final accuracy score on the test dataset

---

## 📚 Libraries Used

- Pytorch
- NumPy
- Matplotlib
- scikit-learn

---

## 🧠 Key Concepts

- Dataset preprocessing with Pandas
- Creating custom PyTorch Dataset
- Model training loop in PyTorch
- model optimisation techniques(dropout, Batch Normalisation, Regularization)
- GPU compatibility
