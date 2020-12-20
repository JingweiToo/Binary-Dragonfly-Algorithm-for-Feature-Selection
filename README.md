# Binary Dragonfly Algorithm for Feature Selection

![Wheel](https://www.mathworks.com/matlabcentral/mlc-downloads/downloads/2e802d69-43b0-49cd-8121-792c643de940/74bfd0a2-6577-477c-a419-bb920010a910/images/1603353704.JPG)

## Introduction
* This toolbox offers Binary Dragonfly Algorithm ( BDA ) method
* The < Main.m file > illustrates the example of how BDA can solve the feature selection problem using benchmark data-set.


## Input
* *feat*     : feature vector ( Instances *x* Features )
* *label*    : label vector ( Instances *x* 1 )
* *N*        : number of dragonflies
* *max_Iter* : maximum number of iterations


## Output
* *sFeat*    : selected features
* *Sf*       : selected feature index
* *Nf*       : number of selected features
* *curve*    : convergence curve


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


