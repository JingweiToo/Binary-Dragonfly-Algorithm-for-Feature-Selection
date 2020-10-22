function X=jInitialPopulation(N,D)
X=zeros(N,D);
for i=1:N
  for d=1:D
    if rand() > 0.5
      X(i,d)=1;
    end
  end
end
end

    