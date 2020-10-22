%-------------------------------------------------------------------------%
%  Binary Dragonfly Algorithm (BDA) source codes demo version             %
%                                                                         %
%  Programmer: Jingwei Too                                                %
%                                                                         %
%  E-Mail: jamesjames868@gmail.com                                        %
%-------------------------------------------------------------------------%


function [sFeat,Sf,Nf,curve]=jBDA(feat,label,opts)
if isfield(opts,'N'), N=opts.N; end
if isfield(opts,'T'), T=opts.T; end

fun=@jFitnessFunction; 
D=size(feat,2); 
X=jInitialPopulation(N,D); DX=zeros(N,D); 
curve=inf; fitF=inf; fitE=-inf; t=1;
fit=zeros(1,N); Xnew=zeros(N,D); Dmax=6;
%---// Iteration Start
while t <= T
  for i=1:N
    fit(i)=fun(feat,label,X(i,:),opts); 
    if fit(i) < fitF
      fitF=fit(i); Xf=X(i,:);
    end
    if fit(i) > fitE
      fitE=fit(i); Xe=X(i,:);
    end
  end
  w=0.9-t*((0.9-0.4)/T);
  rate=0.1-t*((0.1-0)/(T/2));
  if rate < 0, rate=0; end
  s=2*rand()*rate; a=2*rand()*rate; c=2*rand()*rate;    
  f=2*rand(); e=rate; 
  for i=1:N
    index=0; nNeighbor=0; Xn=zeros(1,D); DXn=zeros(1,D);
    for j=1:N
      if i~=j
        index=index+1; nNeighbor=nNeighbor+1;
        DXn(index,:)=DX(j,:); Xn(index,:)=X(j,:);
      end
    end
    S=repmat(X(i,:),nNeighbor,1)-Xn; S=-sum(S,1);
    A=sum(DXn,1)/nNeighbor;
    C=sum(Xn,1)/nNeighbor; C=C-X(i,:);
    F=Xf-X(i,:);
    E=Xe+X(i,:);
    for d=1:D
      dX=(s*S(d)+a*A(d)+c*C(d)+f*F(d)+e*E(d))+w*DX(i,d);
      dX(dX > Dmax)=Dmax; dX(dX < -Dmax)=-Dmax; DX(i,d)=dX;
      TF=abs(DX(i,d)/sqrt(((DX(i,d)^2)+1))); 
      if rand() < TF
        Xnew(i,d)=1-X(i,d);
      else
        Xnew(i,d)=X(i,d);
      end
    end
  end
  X=Xnew;
  curve(t)=fitF; 
  fprintf('\nIteration %d Best (BDA)= %f',t,curve(t))
  t=t+1;
end
Pos=1:D; Sf=Pos(Xf==1); sFeat=feat(:,Sf); Nf=length(Sf); 
end


