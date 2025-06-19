# 🧠 Neural Network from Scratch (NumPy only) – Digit Recognizer

This project implements a **simple feedforward neural network from scratch** using only **NumPy**, without any machine learning libraries like TensorFlow or PyTorch.

It is trained on the [Kaggle Digit Recognizer dataset](https://www.kaggle.com/c/digit-recognizer), which is a subset of the famous **MNIST handwritten digits** dataset.

## 📌 Features

- 2-layer neural network (1 hidden layer with ReLU, output with softmax)
- Manual implementation of:
  - Forward Propagation
  - Backpropagation
  - Gradient Descent
- One-hot encoding for labels
- Softmax output for multi-class classification
- Accuracy tracking during training

---

## 📁 Dataset

Load the dataset in Kaggle Notebooks:

data = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
🧮 Model Architecture
Input Layer: 784 neurons (28x28 pixels flattened)

Hidden Layer: 10 neurons with ReLU activation

Output Layer: 10 neurons with softmax (digits 0–9)

🔧 Training

W1, b1, W2, b2 = gradient_descent(X_train, Y_train, iterations=500, alpha=0.5)
Training output:


iteration: 0
Accuracy: 0.113
...
iteration: 500
Accuracy: 0.88 (varies depending on initialization and shuffle)
🧪 Evaluation Function

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size
📊 Example Accuracy Curve
You can extend the script to plot training accuracy like:


accuracies = []
for i in range(iterations):
    ...
    if i % 50 == 0:
        acc = get_accuracy(get_predictions(A2), Y)
        accuracies.append(acc)

plt.plot(accuracies)
plt.xlabel("Iterations (x50)")
plt.ylabel("Accuracy")
plt.title("Training Accuracy")
plt.show()
🛠 Technologies
Python

NumPy

Pandas

Matplotlib (optional, for plotting)

📌 How to Run (on Kaggle)
Fork the notebook or upload this script

Load the dataset using Kaggle's dataset tab

Run all cells

🚀 Future Improvements
Add multiple hidden layers

Implement dropout or batch normalization

Extend for testing on Kaggle test set

📷 Sample Output
You can visualize predictions using:

index = 0
img = X_dev[:, index].reshape(28, 28)
plt.imshow(img, cmap='gray')
print("Predicted:", get_predictions(forward_prop(W1,b1,W2,b2,X_dev)[-1])[index])
print("Actual:", Y_dev[index])
📄 License
This project is for educational purposes.
