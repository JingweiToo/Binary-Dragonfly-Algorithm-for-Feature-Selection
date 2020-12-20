function Acc = jAccuracy(pred,yvalid)
True     = 0; 
num_data = length(yvalid);
for i = 1:num_data
  if isequal(pred(i),yvalid(i))
    True = True + 1;
  end
end
Acc = True / num_data;
end

