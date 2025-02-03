# Deep-Learning-Lab



Experiment-1 1.How to upload and handle files in Google Colab.

2.The importance of data preprocessing for machine learning applications.

3.How to convert binary image data into a structured CSV format for easier analysis.

4.Basic concepts of handling image pixel values and label data.

5.The significance of structured datasets in training deep learning models for digit recognition.

Experiment-2 1.Understanding Data Separation

Linearly separable data can be classified using a simple neural network.
Non-linearly separable data requires a more complex architecture.
2.Building a Neural Network with Keras

We used the Sequential model to create a simple classifier.
A single-layer network with a sigmoid activation function can classify linearly separable data.
3.Training and Evaluating the Model

The model was trained using binary cross-entropy loss and the Adam optimizer.
Accuracy is a key metric for evaluating classification performance.
4.Importance of Model Architecture

For simple tasks, a basic neural network can perform well.
 For complex data, adding hidden layers and using non-linear activation functions (like ReLU) can improve performance.
Experiment-3

1.Understanding CNNs for Image Classification

We explored how Convolutional Neural Networks (CNNs) work for image classification.
We built a custom CNN model with convolutional, pooling, and fully connected layers.
Impact of Hyperparameters
2.Different activation functions (ReLU, Tanh, Leaky ReLU) affected model performance.

Various weight initialization techniques (Xavier, Kaiming, Random) influenced convergence speed.
Optimizers (SGD, Adam, RMSProp) played a crucial role in training efficiency and accuracy.
3.Comparison Between Custom CNN and Transfer Learning (ResNet-18)

Custom CNN: Worked well but required extensive tuning for better accuracy.
ResNet-18 (Pretrained): Showed superior performance due to transfer learning, requiring less training effort.
4.Dataset Preparation and Augmentation

Preprocessing (resizing, normalization) improved training stability.
Splitting datasets into training (80%) and validation (20%) ensured proper model evaluation.
5.Performance Analysis & Evaluation

Training and validation loss/accuracy trends helped assess model effectiveness.
Transfer learning provided better generalization, proving its efficiency over training CNNs
