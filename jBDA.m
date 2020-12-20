function [sFeat,Sf,Nf,curve] = jBDA(feat,label,N,max_Iter,HO)

fun = @jFitnessFunction; 
dim = size(feat,2); 
X   = jInitialPopulation(N,dim);
DX  = zeros(N,dim); 

fitF = inf; 
fitE = -inf; 
fit  = zeros(1,N);
Xnew = zeros(N,dim); 
Dmax = 6;

curve = inf;
t = 1;
%---// Iteration Start
while t <= max_Iter
  for i = 1:N
    fit(i) = fun(feat,label,X(i,:),HO); 
    if fit(i) < fitF
      fitF = fit(i); 
      Xf   = X(i,:);
    end
    if fit(i) > fitE
      fitE = fit(i); 
      Xe   = X(i,:);
    end
  end
  w    = 0.9 - t * ((0.9 - 0.4) / max_Iter);
  rate = 0.1 - t * ((0.1 - 0) / (max_Iter / 2));
  if rate < 0
    rate = 0; 
  end
  s = 2 * rand() * rate;
  a = 2 * rand() * rate;
  c = 2 * rand() * rate;    
  f = 2 * rand();
  e = rate; 
  for i = 1:N
    index        = 0; 
    num_neighbor = 0; 
    Xn           = zeros(1,dim); 
    DXn          = zeros(1,dim);
    for j = 1:N
      if i ~= j
        index        = index + 1;
        num_neighbor = num_neighbor + 1;
        DXn(index,:) = DX(j,:);
        Xn(index,:)  = X(j,:);
      end
    end
    temp_S = repmat(X(i,:),num_neighbor,1) - Xn;
    S      = -sum(temp_S,1);
    A      = sum(DXn,1) / num_neighbor;
    temp_C = sum(Xn,1) / num_neighbor;
    C      = temp_C - X(i,:);
    F      = Xf - X(i,:);
    E      = Xe + X(i,:);
    for d = 1:dim
      dX = (s * S(d) + a * A(d) + c * C(d) + f * F(d) + e * E(d)) + ...
        w * DX(i,d);
      dX(dX > Dmax) = Dmax;  dX(dX < -Dmax) = -Dmax; 
      DX(i,d) = dX;
      TF      = abs(DX(i,d) / sqrt(((DX(i,d) ^ 2) + 1))); 
      if rand() < TF
        Xnew(i,d) = 1 - X(i,d);
      else
        Xnew(i,d) = X(i,d);
      end
    end
  end
  X = Xnew;
  curve(t) = fitF; 
  fprintf('\nIteration %d Best (BDA)= %f',t,curve(t))
  t = t + 1;
end
Pos   = 1:dim;
Sf    = Pos(Xf == 1); 
sFeat = feat(:,Sf); 
Nf    = length(Sf); 
end


function X = jInitialPopulation(N,dim)
X = zeros(N,dim);
for i = 1:N
  for d = 1:dim
    if rand() > 0.5
      X(i,d) = 1;
    end
  end
end
end


