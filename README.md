## DeepLearning Basic Study

<p align="center">
    <img width="100" src="https://upload.wikimedia.org/wikipedia/commons/thumb/1/11/TensorFlowLogo.svg/225px-TensorFlowLogo.svg.png" />
    <img width="100" src="https://media-thumbs.golden.com/OLqzmrmwAzY1P7Sl29k2T9WjJdM=/200x200/smart/golden-storage-production.s3.amazonaws.com/topic_images/e08914afa10a4179893eeb07cb5e4713.png" />
    <img width="100" src="https://keras.io/img/keras-logo-small-wb.png" />
    <img width="100" src="http://dcomstudy.com/image/header.png" />
    <img width="100" src="https://avatars2.githubusercontent.com/u/37439249?s=200&v=4" />
</p>


This is repository for Deep Learning Study in Kyung Hee University Computer Engineering Club `D.COM`.

#### Recommend this study to those who want to review the Machine Learning concept again and to those who have just learned Python.

- I've created a course material that will be accessible to the first person to start Python.
- [Tae Hwan Jung(@graykode)](https://github.com/graykode) will lead this Study with **Pytorch** for DeepLearning Framework. But I will implement Tensorflow, Pytorch, Keras for beginner. 
- We deal with **basic mathematical theory** and **basic models in Deep Learning** such as ` DNN, CNN, RNN, LSTM ` in 1st Study **All of Code were implemented with less than 30 lines.** 
- We will use `Google Colaboratory` GPU for memory resource so you can run easily in Colab link.(Thank for Google!)
- First, I made lecture with page link material in Korean, only wrote Contents in English



### Contribution Guide

If you find English link or helpful link irrespective of language, Please give me contribution in README, Markdown like this.

> Linear Regression([Eng[(your contribution link), Kor)



## Curriculum 

Please see down Contents.
- 1 Weeks
  - Supervisor Learning vs. Un-supervisor Learning
  - Linear Regression, Logistic Regression `manual` Gradient Descent implementation using `pure python`
- 2 Weeks
  - method using Google Colaboratory.
  - Linear Regression, Logistic Regression Review, Convert `manual` to `auto` implementation using `Pytorch`
- 3 Weeks 
  - Classification with DNN(Deep Neural Network) in `Pytorch`
  - apply Regularization(DropOut) concept to DNN
  - Optimization function in `Pytorch`, mini-batch, SGD, Adagrad, RMSProp, AdaDelta, Adam optimizer
- 4 Weeks
  - Basic Convolution Neural Network
  - load dataset and use data loader with `torchvision`
  - apply Machine Learning Diagnostic concept to DNN
  - Implementation MNIST Classification using CNN
- 5 Weeks
  - Basic RNN(Recurrent Neural Network) and LSTM in Pytorch
  - Teacher Forcing vs. No Teacher Forcing
  - Practice : Predict Next word using RNN or LSTM
- 6 Weeks - Hackathon
  - Topic1 : Classification Cat , Dog Image, [Dataset](https://github.com/ardamavi/Dog-Cat-Classifier/tree/master/Data/Train_Data)
  - Topic2 : Classification Positive or Negative Korean Naver Movie, [Dataset](https://github.com/e9t/nsmc)

## Contents

#### 0. Review Basic mathematical Theory with pure `Python`
- **Supervisor Learning vs. Unsupervisor Learning : In this Study, We will deal with only supervisor concept model.**
- Probability Review(Eng, Kor)
- Linear Regression(Eng, [Kor](https://wikidocs.net/4212))
  - Univariate Linear Regression(Eng, [Kor](https://wikidocs.net/4213)) vs. Multivariate Linear Regression(Eng, [Kor](https://wikidocs.net/7639))
  - **loss function and activation function** in Linear Regression
    - activation function : identity map(Eng, [Kor](https://ko.wikipedia.org/wiki/%ED%95%AD%EB%93%B1_%ED%95%A8%EC%88%98))
    - loss function : MSE function([Eng, Kor](https://en.wikipedia.org/wiki/Mean_squared_error))
  - Gradient Descent in Linear Regression
    - manual : [0.LinearRegression(manual)-Univariate.py](https://github.com/graykode/DeepLearning-Study/blob/master/0.LinearRegression(manual)-Univariate.py)
    - manual : [0.LinearRegression(manual)-Multivariate.py](https://github.com/graykode/DeepLearning-Study/blob/master/0.LinearRegression(manual)-Multivariate.py)
  - Problem : XOR
- Logistic Regression
  - What is different with Linear Regression?(Eng, [Kor](https://wikidocs.net/4267))
  - **loss function and activation function** in Logistic Regression
    - activation function : [sigmoid ](https://en.wikipedia.org/wiki/Sigmoid_function)vs. [tanh](https://en.wikipedia.org/wiki/Hyperbolic_function) vs. [ReLu](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) vs. [Softmax](https://en.wikipedia.org/wiki/Softmax_function)
    - loss function : Maximizing Likelihood is Minimizing Cross-Entropy(Eng, [Kor](https://taeoh-kim.github.io/blog/cross-entropy%EC%9D%98-%EC%A0%95%ED%99%95%ED%95%9C-%ED%99%95%EB%A5%A0%EC%A0%81-%EC%9D%98%EB%AF%B8/))
  - Gradient Descent in Logistic Regression
    - manual : [0.LogisticRegression(manual)-Binary.py](https://github.com/graykode/DeepLearning-Study/blob/master/0.LogisticRegression(manual)-Binary.py)
    - manual : [0.LogisticRegression(manual)-Softmax.py](https://github.com/graykode/DeepLearning-Study/blob/master/0.LogisticRegression(manual)-Softmax.py)
  - different with binary classification and multi classification(sigmoid vs. Softmax)(Eng, [Kor1](https://wikidocs.net/4291), [Kor2](https://taeoh-kim.github.io/blog/bayes-theorem%EA%B3%BC-sigmoid%EC%99%80-softmax%EC%82%AC%EC%9D%B4%EC%9D%98-%EA%B4%80%EA%B3%84/))
  - different with Multi-Classification and Multi-labels Classification([Eng](https://stats.stackexchange.com/questions/11859/what-is-the-difference-between-multiclass-and-multilabel-problem), Kor)
- Optimizing
  - What is batch and mini-batch?(Eng, [Kor](http://shuuki4.github.io/deep%20learning/2016/05/20/Gradient-Descent-Algorithm-Overview.html))
  - role of Momentum(Eng, [Kor](http://shuuki4.github.io/deep%20learning/2016/05/20/Gradient-Descent-Algorithm-Overview.html))
  - SGD, Adagrad, RMSProp, AdaDelta, Adam optimizer([Eng](http://ruder.io/optimizing-gradient-descent/?fbclid=IwAR3-EUWRXxLwNlGIEBaETVeVU9VOnDH8hIlp1PJvMG0StbM72gEKMpWA_VA), [Kor](http://shuuki4.github.io/deep%20learning/2016/05/20/Gradient-Descent-Algorithm-Overview.html)) : [2.DNN-Optimization.py](https://github.com/graykode/DeepLearning-Study/blob/master/2.DNN-Optimization.py)
- Regularization
  - What is Overfitting?(Eng, [Kor](https://wikidocs.net/4269))
  - Regularization : weight decay
    - weight decay : Linear Regression(Eng, [Kor](https://wikidocs.net/4330))
    - weight decay : Logistic Regression(Eng, [Kor](https://wikidocs.net/4331))
  - Regularization : dropout(Eng, [Kor](https://pythonkim.tistory.com/42))
- Machine Learning Diagnostic
  - Train Set, Cross Validation Set, Test Set(Eng, [Kor](https://wikidocs.net/4656))
  - Bias vs. Variance(Eng, [Kor](https://wikidocs.net/4657))
  - Learning Curves(Eng, [Kor](https://wikidocs.net/4658))


#### 1.DeepLearning FrameWork Basic

- Abstract Model using Pytorch Class : [1.Pytorch-Basic.py](https://github.com/graykode/DeepLearning-Study/blob/master/1.Pytorch-Basic.py)
- method using Google Colaboratory
- Convert `manual gradient descent` to `auto graident descent`
  - [1.LinearRegression(auto)-Univariate.py](https://github.com/graykode/DeepLearning-Study/blob/master/1.LinearRegression(auto)-Univariate.py)
  - [1.LinearRegression(auto)-Multivariate.py](https://github.com/graykode/DeepLearning-Study/blob/master/1.LinearRegression(auto)-Multivariate.py)
  - [1.LogisticRegression(auto)-Binary.py](https://github.com/graykode/DeepLearning-Study/blob/master/1.LogisticRegression(auto)-Binary.py)
  - [1.LogisticRegression(auto)-Softmax.py](https://github.com/graykode/DeepLearning-Study/blob/master/1.LogisticRegression(auto)-Softmax.py)


#### 2.DNN(Deep Neural Network)
- Mathematical Back Propagation in Deep Neural Network(Eng, [Kor1](https://wikidocs.net/4262), [Kor2](https://wikidocs.net/4279))
- Basic Classification using Deep Neural Network
  - ~~Classification  : Linear Regression in Deep Neural Network~~
  - Classification  : Logistic Regression in Deep Neural Network
    - 1 Layer Classification : [2.DNN-LinearRegression1.py](https://github.com/graykode/DeepLearning-Study/blob/master/2.DNN-LinearRegression1.py)
    - 2 Layers Classification : [2.DNN-LinearRegression2.py](https://github.com/graykode/DeepLearning-Study/blob/master/2.DNN-LinearRegression2.py)
- Dropout in Deep Neural Network : [2.DNN-Dropout.py](https://github.com/graykode/DeepLearning-Study/blob/master/2.DNN-Dropout.py)


#### 3.DataLoader and basic Dataset and Image handler
- MNIST : [3.DataLoader-MNIST.py](https://github.com/graykode/DeepLearning-Study/blob/master/3.DataLoader-MNIST.py)
- Cifar10 : [3.DataLoader-Cifar10.py](https://github.com/graykode/DeepLearning-Study/blob/master/3.DataLoader-Cifar10.py)
- Cifar100 : [3.DataLoader-Cifar100.py](https://github.com/graykode/DeepLearning-Study/blob/master/3.DataLoader-Cifar100.py)
- Image Folder : [3.DataLoader-ImageFolder.py](https://github.com/graykode/DeepLearning-Study/blob/master/3.DataLoader-ImageFolder.py)


#### 4.CNN(Convolution Neural Network)

- [awesome lecture](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks?fbclid=IwAR21k7YvRmCC1RqAJznzLjDPEf8EaZ2jBGeevX4GkiXruocr1akBAIX9-4U)
- Structure of CNN
  - [4.CNN-Introduce.py](https://github.com/graykode/DeepLearning-Study/blob/master/4.CNN-Introduce.py)
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
- Practice : Classification MNIST



#### 5.RNN(Recurrent Neural Network)
- [awesome lecture](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-recurrent-neural-networks?fbclid=IwAR0rE5QoMJ3l005fhvqoer0Jo_6GiXAF8XM86iWCXD78e3Ud_nDtw_NGzzY)
- Structure of RNN
  - [5.RNN-Introduce.py](https://github.com/graykode/DeepLearning-Study/blob/master/5.RNN-Introduce.py)
  - One-to-one vs. One-to-many vs. Many-to-one vs. Many-to-many
  - Hidden State
  - Output Layer
  - Weight sharing in RNN
- [Teacher Forcing vs. No Teacher Forcing](https://machinelearningmastery.com/teacher-forcing-for-recurrent-neural-networks/)
- FeedForward in Recurrent Neural Network(Eng, [Kor](https://ratsgo.github.io/natural%20language%20processing/2017/03/09/rnnlstm/))
- Mathematical Back Propagation in Recurrent Neural Network(Eng, [Kor](https://ratsgo.github.io/natural%20language%20processing/2017/03/09/rnnlstm/))
- Practice : [Predict Next word using RNN](https://github.com/graykode/DeepLearning-Study/blob/master/5.RNN-PredictWord.py)


#### 6.LSTM(Long Short Term Memory)
- Structure of LSTM
  - [6.LSTM-Introduce.py](https://github.com/graykode/DeepLearning-Study/blob/master/6.LSTM-Introduce.py)
  - Hidden State, Cell State
  - Different of RNN with LSTM
  - Output Layer
  - Weight sharing in RNN
- FeedForward in LSTM(Eng, [Kor](https://ratsgo.github.io/natural%20language%20processing/2017/03/09/rnnlstm/))
- Mathematical Back Propagation in LSTM(Eng, [Kor](https://ratsgo.github.io/natural%20language%20processing/2017/03/09/rnnlstm/))
- Bi-directional LSTM(BiLSTM)(Eng, [Kor](https://ratsgo.github.io/natural%20language%20processing/2017/10/22/manning/))
- Practice : [LSTM-AutoComplete with LSTM](https://github.com/graykode/DeepLearning-Study/blob/master/6.LSTM-AutoComplete.py)



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
