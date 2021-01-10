# Binary Dragonfly Algorithm for Feature Selection

[![View Binary Dragonfly Algorithm for Feature Selection on File Exchange](https://www.mathworks.com/matlabcentral/images/matlab-file-exchange.svg)](https://www.mathworks.com/matlabcentral/fileexchange/81578-binary-dragonfly-algorithm-for-feature-selection)
[![License](https://img.shields.io/badge/license-BSD_3-yellow.svg)](https://github.com/JingweiToo/Binary-Dragonfly-Algorithm-for-Feature-Selection/blob/main/LICENSE)
[![GitHub release](https://img.shields.io/badge/release-1.1-green.svg)](https://github.com/JingweiToo/Binary-Dragonfly-Algorithm-for-Feature-Selection)

![Wheel](https://www.mathworks.com/matlabcentral/mlc-downloads/downloads/2e802d69-43b0-49cd-8121-792c643de940/74bfd0a2-6577-477c-a419-bb920010a910/images/1603353704.JPG)

## Introduction
* This toolbox offers Binary Dragonfly Algorithm ( BDA ) method
* The `Main` file illustrates the example of how BDA can solve the feature selection problem using benchmark data-set.


## Input
* *`feat`*     : feature vector ( Instances *x* Features )
* *`label`*    : label vector ( Instances *x* 1 )
* *`N`*        : number of dragonflies
* *`max_Iter`* : maximum number of iterations


## Output
* *`sFeat`*    : selected features
* *`Sf`*       : selected feature index
* *`Nf`*       : number of selected features
* *`curve`*    : convergence curve
* *`Acc`*      : accuracy of validation model


### Example
```code
% Benchmark data set 
load ionosphere.mat; 

% Set 20% data as validation set
ho = 0.2; 
% Hold-out method
HO = cvpartition(label,'HoldOut',ho);

% Parameter setting
N        = 10; 
max_Iter = 100; 
% Perform feature selection 
[sFeat,Sf,Nf,curve] = jBDA(feat,label,N,max_Iter,HO);

% Accuracy 
Acc = jKNN(sFeat,label,HO);

% Plot convergence curve
plot(1:max_Iter,curve);
xlabel('Number of Iterations');
ylabel('Fitness Value'); 
title('BDA'); grid on;
```


## Requirement
* MATLAB 2014 or above
* Statistics and Machine Learning Toolbox


## Cite As
```code
@article{too2020hyper,
  title={A Hyper Learning Binary Dragonfly Algorithm for Feature Selection: A COVID-19 Case Study},
  author={Too, Jingwei and Mirjalili, Seyedali},
  journal={Knowledge-Based Systems},
  volume={212},
  pages={106553},
  year={2020},
  publisher={Elsevier}
}
```
