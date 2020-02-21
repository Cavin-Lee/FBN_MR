function [W] = accel_grad_PSR_scale(A,lambda)

Xtrain = A;
W=accel_grad_PSR_cavin(A,lambda);
%W=corrcoef(Xtrain);
clear A;
L0 = 100;
gamma = 1.2;
W_init = W;
epsilon = 10^-5;
max_itr = 100;

alpha = 1;
Z_old = W_init;
Wp = W_init;
L = L0;
fval_old = rand(1,1);
fval = rand(1,1);
itr_counter = 0;
fval_vec = [];
n=size(sum(abs(W)),2);
%degree=ones(n)*diag(sum(abs(W)));

degree=ones(n)*diag(1./(sum(abs(W))+0.01));
gamma=exp(-(degree+degree'));


lambda_new=gamma*lambda;

while abs((fval_old-fval)/fval_old)>epsilon
    itr_counter = itr_counter+1;
    fval_old = fval;
    W_old = Wp;
    [Wp,P,sval] = ComputeQP(Xtrain,Z_old,L,lambda_new);
    f = 0.5*norm(Wp-corrcoef(Xtrain),'fro')^2;
    fval = f+sval;
    Q = P+sval;
    
    while fval>Q
        L = L*gamma;
        [Wp,P,sval] = ComputeQP(Xtrain,Ytrain,Z_old,L,lambda_new);
        f = 0.5*norm(Wp-corrcoef(Xtrain),'fro')^2;
        fval = f+sval;
        Q = P+sval;
    end

    fval_vec = [fval_vec,fval];
    alpha_old = alpha;
    alpha = (1+sqrt(1+4*alpha^2))/2;
    Z_old = Wp+((alpha_old-1)/alpha)*(Wp-W_old);
    
    if itr_counter>max_itr
        break;
    end
    
end
W = Wp;
return;

function [Wp,P,sval] = ComputeQP(X,W,L,lambda_new)

[W1,delta_W] = ComputeGradStep(X,W,L);

Wp=sign(W1).*max(abs(W1)-lambda_new/L,0);
sval=sum(sum(abs(Wp.*lambda_new)));
P = 0.5*norm(W-corrcoef(X),'fro')^2+trace(delta_W'*(Wp-W))+0.5*L*norm(Wp-W,'fro')^2;

return;

function [W1,delta_W] = ComputeGradStep(X,W,L)

delta_W = ComputeDerivative(X,W);
W1 = W-(1/L)*delta_W;

return;

function dev = ComputeDerivative(X,W)

dev = W-corrcoef(X);

return;