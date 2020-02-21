

function [W] = accel_grad_PSR_cavin_tes(A,lambda)

Xtrain = A;
clear A;
L0 = 100;
% gamma = 1.2;
% W_init = rand(size(Xtrain,2),size(Xtrain,2));
gamma = 1.1;
W_init = corrcoef(Xtrain);
epsilon = 10^-5;
max_itr = 200;

alpha = 1;
Z_old = W_init;
Wp = W_init;
L = L0;
fval_old = rand(1,1);
fval = zeros(1,1);
itr_counter = 0;
fval_vec = [];

while abs((fval_old-fval))>epsilon
    
    itr_counter = itr_counter+1;
    fval_old = fval;
    W_old = Wp;
    [Wp,P,sval] = ComputeQP(Xtrain,Z_old,L,lambda);
    f = 0.5*norm(Wp-corrcoef(Xtrain),'fro')^2;
    fval = f+sval;
    Q = P+sval;
    
    while fval>Q
        L = L*gamma;
        [Wp,P,sval] = ComputeQP(Xtrain,Ytrain,Z_old,L,lambda);
        f = 0.5*norm(Wp-corrcoef(Xtrain),'fro')^2;
        fval = f+sval;
        Q = P+sval;
    end

    
    alpha_old = alpha;
    alpha = (1+sqrt(1+4*alpha^2))/2;
    Z_old = Wp+((alpha_old-1)/alpha)*(Wp-W_old);
    
    if itr_counter>max_itr
        break;
    end
    
end
W = Wp;
return;

function [Wp,P,sval] = ComputeQP(X,W,L,lambda)

[W1,delta_W] = ComputeGradStep(X,W,L);

Wp=sign(W1).*max((abs(W1)-lambda/L),0);
sval=lambda*sum(sum(abs(Wp)));
P = 0.5*norm(W-corrcoef(X),'fro')^2+trace(delta_W'*(Wp-W))+0.5*L*norm(Wp-W,'fro')^2;

return;

function [W1,delta_W] = ComputeGradStep(X,W,L)

delta_W = ComputeDerivative(X,W);
W1 = W-(1/L)*delta_W;

return;

function dev = ComputeDerivative(X,W)

dev = W-corrcoef(X);

return;