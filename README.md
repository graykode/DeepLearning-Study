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
- First, I made lecture with page link material in Korean, only wrote Contents in English



### Contribution

If you find English link, Please liking in Markdown like this.

> Linear Regression([Eng[(your contribution link), Kor)



## Curriculum 

Please see down Contents.

- 1 Weeks : Linear Regression, DNN(Classification  : Linear Regression in Deep Neural Network)
- 2 Weeks : Logistic Regression, DNN(Classification  : Logistic Regression in Deep Neural Network)



## 1st Contents

#### 0. Review Basic mathematical Theory with pure `Python`
- Probability Review(Eng, Kor)
- Linear Regression(Eng, [Kor](https://wikidocs.net/4212))
  - Univariate Linear Regression(Eng, [Kor](https://wikidocs.net/4213)) vs. Multivariate Linear Regression(Eng, [Kor](https://wikidocs.net/7639))
  - **loss function and activation function** in Linear Regression
    - activation function : identity map(Eng, [Kor](https://ko.wikipedia.org/wiki/%ED%95%AD%EB%93%B1_%ED%95%A8%EC%88%98))
    - loss function : MSE function([Eng, Kor](https://en.wikipedia.org/wiki/Mean_squared_error))
  - Gradient Descent in Linear Regression
    - manual : [Univariate Linear Regression](https://github.com/graykode/DeepLearning-Study/blob/master/0.Univariate_Linear_Regression.py)
    - manual : [Multivariate Linear Regression](https://github.com/graykode/DeepLearning-Study/blob/master/0.Multivariate_Linear_Regression.py)
- Logistic Regression
  - What is different with Linear Regression?(Eng, [Kor](https://wikidocs.net/4267))
  - **loss function and activation function** in Logistic Regression
    - activation function : [sigmoid ](https://en.wikipedia.org/wiki/Sigmoid_function)vs. [tanh](https://en.wikipedia.org/wiki/Hyperbolic_function) vs. [ReLu](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) vs. [Softmax](https://en.wikipedia.org/wiki/Softmax_function)
    - loss function : Maximizing Likelihood is Minimizing Cross-Entropy(Eng, [Kor](https://taeoh-kim.github.io/blog/cross-entropy%EC%9D%98-%EC%A0%95%ED%99%95%ED%95%9C-%ED%99%95%EB%A5%A0%EC%A0%81-%EC%9D%98%EB%AF%B8/))
  - Gradient Descent in Logistic Regression
    - manual : [binary Logistic Regression](https://github.com/graykode/DeepLearning-Study/blob/master/0.Logistic_Regression.py)
  - different with binary classification and multi classification(sigmoid vs. Softmax)(Eng, [Kor1](https://wikidocs.net/4291), [Kor2](https://taeoh-kim.github.io/blog/bayes-theorem%EA%B3%BC-sigmoid%EC%99%80-softmax%EC%82%AC%EC%9D%B4%EC%9D%98-%EA%B4%80%EA%B3%84/))
- Optimizing
  - auto gradient  in Deep Learning Framework
    - auto gradient : Linear Regression
    - auto gradient : Logistic Regression
  - What is batch and mini-batch?(Eng, [Kor](http://shuuki4.github.io/deep%20learning/2016/05/20/Gradient-Descent-Algorithm-Overview.html))
  - role of Momentum(Eng, [Kor](http://shuuki4.github.io/deep%20learning/2016/05/20/Gradient-Descent-Algorithm-Overview.html))
  - SGD, Adagrad, RMSProp, AdaDelta, Adam optimizer([Eng](http://ruder.io/optimizing-gradient-descent/?fbclid=IwAR3-EUWRXxLwNlGIEBaETVeVU9VOnDH8hIlp1PJvMG0StbM72gEKMpWA_VA), [Kor](http://shuuki4.github.io/deep%20learning/2016/05/20/Gradient-Descent-Algorithm-Overview.html))
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



## Reference

- Andrew NG - Machine Learning Lecture
- Korean Andrew Ng NoteBook : [WikiBook](https://wikidocs.net/book/587)



## Author

- Tae Hwan Jung(Jeff Jung) @graykode
- Author Email : [nlkey2022@gmail.com](mailto:nlkey2022@gmail.com)



## License


<p xmlns:dct="http://purl.org/dc/terms/">
  <a rel="license"
     href="https://creativecommons.org/licenses/by-nc-sa/2.0/kr/">
    <img src="https://wikidocs.net/static/img/by-nc-sa.png" style="border-style: none;" alt="CC0" />
  </a>
</p>