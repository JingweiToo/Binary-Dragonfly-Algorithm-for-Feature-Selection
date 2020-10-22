% Notation: This fitness function is for demonstration 

function cost=jFitnessFunction(feat,label,X,opts)
alpha=0.99; beta=0.01;
maxFeat=length(X); 
if sum(X==1)==0
  cost=inf;
else
  Error=jwrapperknn(feat(:,X==1),label,opts);
  Nsf=sum(X==1);
  cost=alpha*Error+beta*(Nsf/maxFeat); 
end
end


function Err=jwrapperknn(sFeat,label,opts)
if isfield(opts,'kfold'), kfold=opts.kfold; end
if isfield(opts,'k'), k=opts.k; end
if isfield(opts,'Model'), Model=opts.Model; end
Acc=zeros(1,kfold); 
for i=1:kfold
	trainIdx=Model.training(i); testIdx=Model.test(i);
  xtrain=sFeat(trainIdx==1,:); ytrain=label(trainIdx==1);
  xtest=sFeat(testIdx==1,:); ytest=label(testIdx==1);
  KNN=fitcknn(xtrain,ytrain,'NumNeighbors',k);
  pred=predict(KNN,xtest); clear KNN
  Acc(i)=jAccuracy(pred,ytest);
end
Err=1-mean(Acc/100); 
end












