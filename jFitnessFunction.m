% Notation: This fitness function is for demonstration 

function cost=jFitnessFunction(feat,label,X,HO)
alpha=0.99; beta=0.01;
maxFeat=length(X); 
if sum(X==1)==0
  cost=inf;
else
  Error=jwrapperknn(feat(:,X==1),label,HO);
  Nsf=sum(X==1);
  cost=alpha*Error+beta*(Nsf/maxFeat); 
end
end


function Err=jwrapperknn(sFeat,label,HO)
%---// Parameter setting for k-value of KNN //
k=5; 
trainIdx=HO.training; testIdx=HO.test;
xtrain=sFeat(trainIdx==1,:); ytrain=label(trainIdx==1);
xvalid=sFeat(testIdx==1,:); yvalid=label(testIdx==1);
KNN=fitcknn(xtrain,ytrain,'NumNeighbors',k);
pred=predict(KNN,xvalid); 
Acc=jAccuracy(pred,yvalid);
Err=1-(Acc/100); 
end












