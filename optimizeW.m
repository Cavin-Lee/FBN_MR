
function [f,g] = optimizeW(W_tmp,X,M,lambda,d)
% Data setup
% W FBN                 1*(nROI*nROI)
% X Data Matirx         D*nROI
% M index of modular    nROI
% lambda hyper-parameter
% d hyper-parameter in subgradient_nuclearnorm

[d,nROI]=size(X);
W=reshape(W_tmp,nROI,nROI);
% Function value (residual)
term1=1/2* norm(X-X*W, 'fro')^2;

Sabs=0;
for i=1:length(unique(M))
  S=svd(W(M==i,M==i));
  Sabs=Sabs+sum(abs(S));
end

term2=1/2*lambda*Sabs;


Normgradient=subgradient_nuclearnorm(W, d);
term3=1/2*lambda*trace(W'*Normgradient);

f = term1+term2-term3;

% First derivatives computed in matrix form
gterm1=X'*X-X'*X*W;

for i=1:length(unique(M))
   gterm1(M==i,M==i)=gterm1(M==i,M==i)+lambda*subgradient_nuclearnorm(W(M==i,M==i), d);
    
end



gterm3=lambda*subgradient_nuclearnorm(W,d);

g=gterm1-gterm3;

g=reshape(g, nROI*nROI, 1);


end