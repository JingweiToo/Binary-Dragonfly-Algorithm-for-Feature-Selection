% Notation: This fitness function is for demonstration 

function cost = jFitnessFunction(feat,label,X,HO)
alpha   = 0.99; 
beta    = 0.01;
maxFeat = length(X); 
if sum(X == 1) == 0
  cost = inf;
else
  error    = jwrapperknn(feat(:, X == 1),label,HO);
  num_feat = sum(X == 1);
  cost     = alpha * error + beta * (num_feat / maxFeat); 
end
end


function error = jwrapperknn(sFeat,label,HO)
%---// Parameter setting for k-value of KNN //
k = 5;

trainIdx = HO.training;        testIdx  = HO.test;
xtrain   = sFeat(trainIdx,:);  ytrain   = label(trainIdx);
xvalid   = sFeat(testIdx,:);   yvalid   = label(testIdx);

KNN   = fitcknn(xtrain,ytrain,'NumNeighbors',k);
pred  = predict(KNN,xvalid); 
Acc   = jAccuracy(pred,yvalid);
error = 1 - Acc; 
end












