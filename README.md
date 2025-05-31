# Image Classification with CNN Models

A tutorial for image classification on the CIFAR-10 dataset using Convolutional Neural Networks (CNNs) implemented in both TensorFlow/Keras and PyTorch.

![Cifar-10 dataset classes](https://github.com/user-attachments/assets/f8ee80b9-00ae-400c-813e-9a04a6d540be)

## Dataset

The [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class:

- airplane
- automobile
- bird
- cat
- deer
- dog
- frog
- horse
- ship
- truck

---

## Project Structure

```
Image-Classification-with-CNN-models-main/
│
├── README.md
├── LICENSE
├── with_tensorflow.ipynb
└── Pytorch_code/
    ├── with_pytorch.ipynb
    └── pytorch_parameters.pth
```

---

## Getting Started

### Requirements

- Python 3.x
- TensorFlow
- PyTorch
- scikit-learn
- matplotlib
- tqdm

---

## TensorFlow/Keras Implementation

See: [with_tensorflow.ipynb](with_tensorflow.ipynb)

### Workflow

1. **Data Loading & Preprocessing**
    - Load CIFAR-10 dataset.
    - Normalize pixel values to [0, 1].
    - Visualize sample images.

2. **Model Architectures**
    - **Feedforward ANN**: Baseline dense network.
    - **CNN (Decreasing Filters)**: Fewer filters in deeper layers.
    - **CNN (Increasing Filters)**: More filters in deeper layers.

3. **Training**
    - Models are trained for 50 epochs.
    - Loss and accuracy are tracked.

4. **Evaluation**
    - Classification reports and confusion matrices are generated.
    - Example code for visualizing predictions.

### Example Results

#### ANN Network

- **Accuracy**: ~56%
- **Classification Report**:  
  ```
    precision    recall  f1-score   support
    ...
    accuracy                           0.56     10000
    macro avg       0.58      0.56      0.57     10000
    weighted avg    0.58      0.56      0.57     10000
  ```

#### CNN (Decreasing Filters)

- **Accuracy**: ~67%
- **Classification Report**:  
  ```
    precision    recall  f1-score   support
    ...
    accuracy                           0.67     10000
    macro avg       0.68      0.67      0.67     10000
    weighted avg    0.68      0.67      0.67     10000
  ```

#### CNN (Increasing Filters)

- **Accuracy**: ~68%
- **Classification Report**:  
  ```
    precision    recall  f1-score   support
    ...
    accuracy                           0.68     10000
    macro avg       0.68      0.67      0.67     10000
    weighted avg    0.68      0.67      0.67     10000
  ```

---

## PyTorch Implementation

See: [Pytorch_code/with_pytorch.ipynb](Pytorch_code/with_pytorch.ipynb)

### Workflow

1. **Data Loading & Preprocessing**
    - Load CIFAR-10 dataset using TensorFlow's loader for consistency.
    - Normalize and convert to PyTorch tensors.
    - Prepare DataLoader for batching.

2. **Model Architecture**
    - Custom `Convolution_model` class:
        - 2 Conv2D layers (32 and 64 filters)
        - MaxPooling
        - 2 Fully Connected layers
        - ReLU and Sigmoid activations

3. **Training**
    - Trained for 200 epochs using Adam optimizer and CrossEntropyLoss.
    - Loss printed every 10 epochs.

4. **Saving & Loading**
    - Model parameters saved to `pytorch_parameters.pth`.
    - Demonstrates loading and evaluating the model.

### Example Results

- **Final Training Loss**: ~1.53 after 200 epochs
- **Model Structure**:
    ```
    Convolution_model(
      (conv_layer_01): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1))
      (pool_layer): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (conv_layer_02): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
      (fc1): Linear(in_features=2304, out_features=64, bias=True)
      (fc2): Linear(in_features=64, out_features=10, bias=True)
      (conv_activation): ReLU()
      (activation): Sigmoid()
    )
    ```

---

## Visualizations

- The notebooks include code to visualize sample images and predictions using matplotlib.

---

## How to Run

1. Clone the repository.
2. Install the required libraries.
3. Open and run the notebooks in Jupyter or Google Colab.

---

## References

- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- TensorFlow and PyTorch official documentation

---

## License

See [LICENSE](LICENSE) for details.
