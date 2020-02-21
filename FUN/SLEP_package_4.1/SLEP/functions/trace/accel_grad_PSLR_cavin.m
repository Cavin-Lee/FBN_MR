

function [W] = accel_grad_PSLR_cavin(A,lambda1,lambda2)

Xtrain = A;
clear A;
L0 = 100;
gamma = 1.1;
W_init = zeros(size(Xtrain,2),size(Xtrain,2));
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

while abs((fval_old-fval)/fval_old)>epsilon
    itr_counter = itr_counter+1;
    fval_old = fval;
    W_old = Wp;
    [Wp,P,sval] = ComputeQP(Xtrain,Z_old,L,lambda1,lambda2);
    f = 0.5*norm(Wp-Xtrain'*Xtrain,'fro')^2;
    fval = f+sval;
    Q = P+sval;
    
    while fval>Q
        L = L*gamma;
        [Wp,P,sval] = ComputeQP(Xtrain,Z_old,L,lambda1,lambda2);
        f = 0.5*norm(Wp-Xtrain'*Xtrain,'fro')^2;
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

function [Wp,P,sval] = ComputeQP(X,W,L,lambda,lambda2)

[W1,delta_W] = ComputeGradStep(X,W,L);


[U,D,V] = svd(W1,0);
D = diag(D);
D = D-(lambda/L);
idx = find(D>0);
Wp = U(:,idx)*diag(D(idx))*V(:,idx)';
Wp=sign(Wp).*max(abs(Wp)-lambda2/L,0);

sval=lambda*sum(D(idx))+lambda2*sum(sum(abs(Wp)));
P = 0.5*norm(W-X'*X,'fro')^2+trace(delta_W'*(Wp-W))+0.5*L*norm(Wp-W,'fro')^2;

return;

function [W1,delta_W] = ComputeGradStep(X,W,L)

delta_W = ComputeDerivative(X,W);
W1 = W-(1/L)*delta_W;

return;

function dev = ComputeDerivative(X,W)

dev = W-X'*X;

return;