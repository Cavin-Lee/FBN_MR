


function [W,fval_vec,itr_counter] = accel_grad_mlr_cavintest2(A,Y,lambda,opt)

Xtrain = A;
Ytrain = Y;
clear A;
clear Y;

if nargin<3
    opt = [];
end

if isfield(opt, 'L0')
    L0 = opt.L0;
else
    L0 = 100;
end

if isfield(opt, 'gamma')
    gamma = opt.gamma;
else
    gamma = 1.1;
end

if isfield(opt, 'W_init')
    W_init = opt.W_init;
else
    W_init = zeros(size(Xtrain,2),size(Ytrain,2));
end

if isfield(opt, 'epsilon')
    epsilon = opt.epsilon;
else
    epsilon = 10^-5;
end

if isfield(opt, 'max_itr')
    max_itr = opt.max_itr;
else
    max_itr = 100;
end

if isfield(opt, 'z')
    z = opt.z;
else
    z = 0;
end

alpha = 1;
Z_old = W_init;
Wp = W_init;
L = L0;
fval_old = rand(1,1);
fval = rand(1,1);
itr_counter = 0;

fval_vec = [];
while abs((fval_old-fval)/fval_old)>epsilon
   itr_counter = itr_counter+1;
    fval_old = fval;
    W_old = Wp;
    [Wp,P,sval] = ComputeQP(Xtrain,Ytrain,Z_old,L,z);
    f = 0.5*norm(Wp-Ytrain'*Xtrain,'fro')^2;
    fval = f+lambda*sval;
    Q = P+lambda*sval;
    while fval>Q
        L = L*gamma;
        [Wp,P,sval] = ComputeQP(Xtrain,Ytrain,Z_old,L,lambda,z);
        f = 0.5*norm(Wp-Ytrain'*Xtrain,'fro')^2;
%         fval = f+lambda*sval;
%         Q = P+lambda*sval;
        fval = f+sval;
        Q = P+sval;
    end
    
    alpha_old = alpha;
    alpha = (1+sqrt(1+4*alpha_old^2))/2;
    Z_old = Wp+((alpha_old-1)/alpha)*(Wp-W_old);
    
    W_old = Wp;
    fval_vec = [fval_vec,fval];
    itr_counter = itr_counter+1;
    if itr_counter>max_itr
        break;
    end
  
    
end
W = Wp;
return;

function [Wp,P,sval] = ComputeQP(X,Y,W,L,lambda)

[W1,delta_W] = ComputeGradStep(X,Y,W,L);
[U,D,V] = svd(W1,0);
D = diag(D);
D = D-(lambda/L);
idx = find(D>0);
sval = sum(D(idx));
Wp = U(:,idx)*diag(D(idx))*V(:,idx)';

P = 0.5*norm(W-X'*Y,'fro')^2+trace(delta_W'*(Wp-W))+0.5*L*norm(Wp-W,'fro')^2;
return;

function [W1,delta_W] = ComputeGradStep(X,Y,W,L)

delta_W = ComputeDerivative(X,Y,W);
W1 = W-(1/L)*delta_W;

return;

function dev = ComputeDerivative(X,Y,W)

dev = W-X'*Y;

return;



