function Acc=jKNN(sFeat,label,HO)
%---// Parameter setting for k-value of KNN //
k=5; 
trainIdx=HO.training; testIdx=HO.test;
xtrain=sFeat(trainIdx==1,:); ytrain=label(trainIdx==1);
xvalid=sFeat(testIdx==1,:); yvalid=label(testIdx==1);
KNN=fitcknn(xtrain,ytrain,'NumNeighbors',k);
pred=predict(KNN,xvalid);
Acc=jAccuracy(pred,yvalid); 
fprintf('\n Accuracy: %g %%',Acc);
end



