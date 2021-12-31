# Heterogeneous Autoencoder Empowered by Quadratic Neurons
This is the repository of our submission for IJCAI 2022 "Heterogeneous Autoencoder Empowered by Quadratic Neurons". In this work,

1. We prototype a generic  fully-connected quadratic autoencoder;
2. We design a random ensemble mixed autoencoder;
3. We propose three heterogeneous autoencoders empowerd with quadratic neurons.

All experiments are conducted with Windows 10 on an AMD R5 3600 CPU at 3.60 GHz and one NVIDIA RTX 2070S 8GB GPU. We implement our model on Python 3.8 with the PyTorch package, an open-source deep learning framework.  

## Quadratic Autoencoder
A quadratic autoencoder is constructed by quadratic neurons. We prototype a fully-conneted quadratic autoencoder.
![An  overview  of  fully-connected  quadratic  autoencoder.The core is the employment of quadratic neurons](https://raw.githubusercontent.com/ljxstc/image_repository/master/小书匠/1640488416607.png)
### Quadratic Neurons
A quadratic neuron was proposed by [1], It computes two inner products  and  one  power  term  of  the  input  vector  and  integrates them for a nonlinear activation function. The output function of a quadratic neuron is expressed as 

![enter description here](https://raw.githubusercontent.com/ljxstc/image_repository/master/小书匠/1640488416614.png),

where $\sigma(\cdot)$ is a nonlinear activation function, $\odot$ denotes the Hadamard product, $\boldsymbol{w}^r,\boldsymbol{w}^g, \boldsymbol{w}^b\in\mathbb{R}^n$ are weight vectors, and $b^r, b^g, c\in\mathbb{R}$ are biases. When $\boldsymbol{w}^g=0$, $b^g=1$, and $\boldsymbol{w}^b=0$, a quadratic neuron degenerates to a conventional neuron:  $\sigma(f(\boldsymbol{x}))= \sigma(\boldsymbol{x}^\top\boldsymbol{w}^{r}+b^{r})$. 
## Random Ensemble Autoencoder
To get two ensemble mixed autoencoders (EMAE), we randomly swap the scores of quadratic autoencoders and conventional autoencoders by a rate $\beta$=0.2, 0.5, 0.8 and get two final scores.

![A scheme of random ensemble mixed autoencoder.](https://raw.githubusercontent.com/ljxstc/image_repository/master/小书匠/1640274878158.png)


## Heterogeneous Autoencoder
We propose three heterogeneous autoencoders integrating conventional  and  quadratic  neurons  in  one  model,  referred  to  as HAE-X, HAE-Y, and HAE-I, respectively.
![The scheme of HAE-X, HAE-Y, and HAE-I.](https://raw.githubusercontent.com/ljxstc/image_repository/master/小书匠/1640274878159.png)

# Repository organization
This repository is organized by three folders, which correspond to the three main experiments in our paper. Three programs (quadratic_autoencoder, ensemble_autoencoder, and heterogeneous_autoencoder) can be run independently.
## Requirements
We use PyCharm 2021.2 to be a coding IDE, if you use the same, you can run this program directly. Other IDE we have not yet tested, maybe you need to change some settings.
* Python == 3.8
* PyTorch == 1.10.1
* CUDA == 11.3 if use GPU
* pyod == 0.9.6
* anaconda == 2021.05
 
## Organization
```
HAE_Empowered_by_Quadratic_Neurons
│   README.md
└───data # Anomaly detection datasets  
└───quadratic_autoencoder # Implement code for QAE
│   │   │   train.py # Main program entry, to train a QAE or AE for classification
│   │   └─── AEModel # Base Model Structure
│   │	│  │	AE.py
│   │	│  │	QAE.py
│   │   └─── utils
│   │	│  │	Dataloader.py
│   │	│  │	QuadraticOperation.py
│   │	│  │	train_function.py
│   │   └─── model # Trained model will be saved to this file folder
│   │   └─── results 
│   │   └─── data
└───ensemble_autoencoder # Implement code for QAE
│   │   │   train.py # Main program entry, to train a ensemble autoencoder for anomaly detection
│   │   └─── Model # Base Model Structure
│   │	│  │	ensemble_autoencoder.py
│   │   └─── utils
│   │	│  │	QuadraticOperation.py
│   │	│  │	.utils.py
│   │   └─── scores # Intermediate results saving
│   │   └─── results 
└───heterogeneous_autoencoder # Implement code for QAE
│   │   │   train_ae.py # Main program entry, to train an autoencoder for anomaly detection
│   │   │   benchmark_method.py # some benchmark methods
│   │   └─── Model # Base Model Structure
│   │	│  │	HAutoEncoder.py
│   │   └─── utils
│   │	│  │	QuadraticOperation.py
│   │	│  │	train_function.py
│   │   └─── results 
```

# How to Use
## Quadratic Autoencoder
Run  ```quadratic_autoencoder/train.py``` to train a quadratic autoencoder or a conventional autoencoder. 

*MNIST* , *Fashion-MNIST* and *Olivetti Faces* will be automatically downloaded in ***'quadratic_autoencoder/data'*** when you run ***'train.py'*** for the first time. For *YaleB*, you need to visit [YaleB](http://vision.ucsd.edu/~leekc/ExtYaleDatabase/ExtYaleB.html) to download **Cropped Images**  and unzip to the ***'quadratic_autoencoder/data'***  folder. This program will save the models and results of your training at ***'quadratic_autoencoder/model'*** and ***'quadratic_autoencoder/results'*** , respectively.
## Ensemble Autoencoder
Run ```ensemble_autoencoder/train.py``` to train a random ensemble autoencoder. Set **swap_rate > 0** to train an ensemble mixed autoencoder (EMAE). 

We use a random ensemble autoencoder that was proposed in [2], and follow the code implementation in [Outlier-Dectection-AE-Ensemble](https://github.com/drguigui1/Outlier-Detection-AE-Ensemble) [3].  The results will be saved to the ***'ensemble_autoencoder/results'*** folder.

The datasets for models ensemble and heterogeneous autoencoders experiments are openly provided by the multi-dimensional point datasets of outlier detectiondatasets [(ODDS)](http://odds.cs.stonybrook.edu/) [4]. We have uploaded some of these datasets which we used in our paper at ***'/data'*** folder.


 
## Heterogeneous Autoencoder
Run ```heterogeneous_autoencoder/train.py``` to train an autoencoder. We provide three heterogeneous autoencoders, a quadratic and a conventional autoencoder. 

 Run ```heterogeneous_autoencoder/benchmark_method.py``` to train other benchmark algorithm.


We follow the implementation of some anomaly detection algorithm by [pyod](https://github.com/yzhao062/pyod) package [5].   All results will be saved to the ***'heterogeneous_autoencoder/results'*** folder.



# Reference
[1] Fenglei Fan, Wenxiang Cong, and Ge Wang.   A new type of neu-rons for machine learning.International journal for numericalmethods in biomedical engineering, 34(2):e2920, 2018.

[2] Chen, J., Sathe, S., Aggarwal, C., & Turaga, D. (2017, June). Outlier detection with autoencoder ensembles. In Proceedings of the 2017 SIAM international conference on data mining (pp. 90-98). Society for Industrial and Applied Mathematics.

[3] https://github.com/drguigui1/Outlier-Detection-AE-Ensemble

[4] Shebuti Rayana.  Odds library [http://odds.cs.stonybrook.edu]. stony brook, ny:  Stony brook university.Department of ComputerScience, 2016.

[5] Zhao, Y., Nasrullah, Z. and Li, Z., 2019. PyOD: A Python Toolbox for Scalable Outlier Detection. Journal of machine learning research (JMLR), 20(96), pp.1-7.

[6] https://github.com/avijit9/Contractive_Autoencoder_in_Pytorch

[7] https://github.com/Abhipanda4/Sparse-Autoencoders
