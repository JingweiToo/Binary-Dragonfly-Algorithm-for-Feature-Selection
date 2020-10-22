function ML=jknn(sFeat,label,Model,opts)
if isfield(opts,'kfold'), kfold=opts.kfold; end
if isfield(opts,'k'), k=opts.k; end
ML=jKNNCrossValidation(sFeat,label,Model,k,kfold);
end


function acc=jKNNCrossValidation(sFeat,label,Model,k,kfold) 
Afold=zeros(1,kfold); 
for i=1:kfold
	trainIdx=Model.training(i); testIdx=Model.test(i);
  xtrain=sFeat(trainIdx==1,:); ytrain=label(trainIdx==1);
  xtest=sFeat(testIdx==1,:); ytest=label(testIdx==1);
  KNN=fitcknn(xtrain,ytrain,'NumNeighbors',k);
  pred=predict(KNN,xtest); clear KNN
  Afold(i)=jAccuracy(pred,ytest);
end
acc=mean(Afold); 
fprintf('\n Accuracy: %g %%',acc);
end



