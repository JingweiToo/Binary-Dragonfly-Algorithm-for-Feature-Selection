function CA=jAccuracy(pred,ytest)
True=0; nData=length(ytest);
for i=1:nData
  if isequal(pred(i),ytest(i))
    True=True+1;
  end
end
CA=100*(True/nData);
end

