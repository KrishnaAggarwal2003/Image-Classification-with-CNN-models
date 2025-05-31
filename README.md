# Image Classification with CNN Models

A tutorial for image classification on the CIFAR-10 dataset using Convolutional Neural Networks (CNNs) implemented in both TensorFlow/Keras and PyTorch.

![Cifar-10 dataset classes](https://github.com/user-attachments/assets/f8ee80b9-00ae-400c-813e-9a04a6d540be)

## Dataset

The [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) consists of 60,000 32x32 colour images in 10 classes, with 6,000 images per class:

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
    - Normalise pixel values.
    - Visualise sample images.

2. **Model Architectures**
    - **Feedforward ANN**: Baseline dense network.
    - **CNN (Decreasing Filters)**: Fewer filters in deeper layers.
    - **CNN (Increasing Filters)**: More filters in deeper layers.

3. **Training**
    - Models are trained for 50 epochs.
    - Loss and accuracy are tracked.

4. **Evaluation**
    - Classification reports and confusion matrices are generated.
    - Example code for visualising predictions.

### Example Results

#### ANN Network

- ** Training Accuracy**: ~91.75% at the end of 50 epochs
- **Classification Report** of test-data:  
  ```
                precision    recall  f1-score   support

           0       0.57      0.72      0.63      1000
           1       0.71      0.65      0.68      1000
           2       0.42      0.55      0.48      1000
           3       0.41      0.33      0.36      1000
           4       0.48      0.54      0.51      1000
           5       0.45      0.54      0.49      1000
           6       0.66      0.58      0.62      1000
           7       0.75      0.53      0.62      1000
           8       0.77      0.57      0.65      1000
           9       0.63      0.64      0.64      1000

    accuracy                           0.56     10000
   macro avg       0.58      0.56      0.57     10000
   weighted avg       0.58      0.56      0.57     10000
  ```

#### CNN (Decreasing Filters)

- **Training Accuracy**: ~95.49% at the end of 50 epochs
- **Classification Report** of test-data:  
  ```
                precision    recall  f1-score   support

           0       0.72      0.66      0.69      1000
           1       0.81      0.77      0.79      1000
           2       0.58      0.52      0.55      1000
           3       0.43      0.57      0.49      1000
           4       0.60      0.63      0.62      1000
           5       0.60      0.52      0.56      1000
           6       0.77      0.74      0.76      1000
           7       0.76      0.65      0.70      1000
           8       0.76      0.79      0.77      1000
           9       0.73      0.80      0.76      1000

    accuracy                           0.67     10000
    macro avg       0.68      0.67      0.67     10000
    weighted avg       0.68      0.67      0.67     10000
  ```

#### CNN (Increasing Filters)

- **Training Accuracy**: ~96.84% at the end of 50 epochs
- **Classification Report** of test-data:  
  ```
                precision    recall  f1-score   support

           0       0.72      0.66      0.69      1000
           1       0.81      0.77      0.79      1000
           2       0.58      0.52      0.55      1000
           3       0.43      0.57      0.49      1000
           4       0.60      0.63      0.62      1000
           5       0.60      0.52      0.56      1000
           6       0.77      0.74      0.76      1000
           7       0.76      0.65      0.70      1000
           8       0.76      0.79      0.77      1000
           9       0.73      0.80      0.76      1000

    accuracy                           0.67     10000
   macro avg       0.68      0.67      0.67     10000
  weighted avg       0.68      0.67      0.67     10000

  ```

---

## PyTorch Implementation

See: [Pytorch_code/with_pytorch.ipynb](Pytorch_code/with_pytorch.ipynb)

### Workflow

1. **Data Loading & Preprocessing**
    - Load the CIFAR-10 dataset using TensorFlow's loader for consistency.
    - Normalise and convert to PyTorch tensors.
    - Prepare DataLoader for batching.

2. **Model Architecture**
    - Custom `Convolution_model` class:
        - 2 Conv2D layers (32 and 64 filters)
        - MaxPooling
        - 2 Fully Connected layers
        - ReLU and Sigmoid activations

3. **Training**
    - Trained for 200 epochs using Adam optimiser and CrossEntropyLoss.
    - Loss is printed every 10 epochs.

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
- Performance **Results** on test-data:
  ```
     Correctly predicted: 6714 and Wrongly predicted: 3286
     Total test data: 10000
     Accuracy of the model: 67.14%
  ```    

---

## Visualizations

- The notebooks include code to visualise sample images and predictions using matplotlib.

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
