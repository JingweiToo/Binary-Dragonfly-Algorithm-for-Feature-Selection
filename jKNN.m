function Acc = jKNN(sFeat,label,HO)
%---// Parameter setting for k-value of KNN //
k = 5; 

trainIdx = HO.training;        testIdx  = HO.test;
xtrain   = sFeat(trainIdx,:);  ytrain   = label(trainIdx);
xvalid   = sFeat(testIdx,:);   yvalid   = label(testIdx);

KNN  = fitcknn(xtrain,ytrain,'NumNeighbors',k);
pred = predict(KNN,xvalid);
Acc  = jAccuracy(pred,yvalid); 

fprintf('\n Accuracy: %g %%',100 * Acc);
end



