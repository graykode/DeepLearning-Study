## DeepLearning-Study

<p align="center">
    <img width="100" src="https://upload.wikimedia.org/wikipedia/commons/thumb/1/11/TensorFlowLogo.svg/225px-TensorFlowLogo.svg.png" />
    <img width="100" src="https://media-thumbs.golden.com/OLqzmrmwAzY1P7Sl29k2T9WjJdM=/200x200/smart/golden-storage-production.s3.amazonaws.com/topic_images/e08914afa10a4179893eeb07cb5e4713.png" />
    <img width="100" src="https://keras.io/img/keras-logo-small-wb.png" />
    <img width="100" src="http://dcomstudy.com/image/header.png" />
    <img width="100" src="https://avatars2.githubusercontent.com/u/37439249?s=200&v=4" />
</p>


This is repository for Deep Learning Study in Kyung Hee University Computer Engineering Club `D.COM`.

- I've created a course material that will be accessible to the first person to start Python.

- [Tae Hwan Jung(@graykode)](https://github.com/graykode) will lead this Study with **Pytorch** for DeepLearning Framework. But I will implement Tensorflow, Pytorch, Keras for beginner. 
- We deal with **basic mathematical theory** and **basic models in Deep Learning** such as ` DNN, CNN, RNN, LSTM ` in 1st Study **All of Code were implemented with less than 30 lines.** 
- We will use `Google Colaboratory` GPU for memory resource so you can run easily in Colab link.(Thank for Google!)
- First, I made lecture material in Korean, But prepare English soon!



## 1st Curriculum

#### 0. Review Basic mathematical Theory with Deep Learning Framework

- Probability Review
- Linear Regression

  - Univariate Linear Regression vs. Multivariate Linear Regression
  - **loss function and activation function** in Linear Regression
    - activation function : identity map
    - loss function : MSE function
  - Gradient Descent in Linear Regression
- Logistic Regression

  - What is different with Linear Regression?

  - **loss function and activation function** in Logistic Regression

    - activation function : sigmoid vs. tanh vs. ReLu vs. Softmax

    - loss function : Maximizing Likelihood is Minimizing Cross-Entropy

  - Gradient Descent in Logistic Regression

  - different with binary classification and multi classification(sigmoid vs. Softmax)
- Optimizing

  - manual gradient vs. auto gradient  in Deep Learning Framework

  - What is batch and mini-batch?
  - role of Momentum
  - SGD, Adagrad, RMSProp, AdaDelta, Adam optimizer
- Regularization

  - What is Overfitting?
  - Regularization : weight decay
  - Regularization : dropout
- Machine Learning Diagnostic
  - Train Set, Cross Validation Set, Test Set
  - Bias vs. Variance



#### 1.DNN(Deep Neural Network)

- Mathematical Back Propagation in Deep Neural Network
- Basic Classification using Deep Neural Network
  - Classification  : Linear Regression in Deep Neural Network
  - Classification  : Logistic Regression in Deep Neural Network
- Dropout in Deep Neural Network



#### 2.DataLoader and basic Dataset

- MNIST
- Cifar10, Cifar100
- Image Folder



#### 3.CNN(Convolution Neural Network)

- Structure of CNN
  - Convolutional Layer
    - Role of filter(=kernel)
    - Role of Padding
    - Weight sharing in Convolutional Layer
  - Role of Channel, Reason using Multi Channel
  - Weight sharing in CNN
  - Pooling Layer
    - Max Pooling
    - Average Pooling
- FeedForward in Convolution Neural Network
- Mathematical Back Propagation in Convolution Neural Network
- Practice : Classification MNIST with AlexNet(2012)



#### 5.RNN(Recurrent Neural Network)

- Structure of RNN
  - Hidden State
  - Output Layer
  - Weight sharing in RNN
- Teacher Forcing vs. No Teacher Forcing
- FeedForward in Recurrent Neural Network
- Mathematical Back Propagation in Recurrent Neural Network
- Practice : Predict Next word using RNN



#### 6.LSTM(Long Short Term Memory)

- Structure of LSTM
  - Hidden State, Cell State
  - Different of RNN with LSTM
  - Output Layer
  - Weight sharing in RNN
- FeedForward in LSTM
- Mathematical Back Propagation in LSTM
- Practice : Implementation difference with RNN



#### 7. Application Level

- Vision : Cat or Dog Image Classification.
- Natural Language Processing : Positive or Negative Classification with Naver Movie Review.