% revised on September 21, 2010 to make parameters consistent

function [W, fval_vec, itr_counter] = accel_grad_mlr_cavin_test(A,Y,opt)

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
%     fval = f+lambda*sval+z*sum(sum(abs(Wp)));%%+z*sum(sum(abs(Wp)))
%     Q = P+lambda*sval+z*sum(sum(abs(Wp)));%+z*sum(sum(abs(Wp)))
    fval = f+sval;
    Q = P+sval;
    
    while fval>Q
%         fprintf('Searching step size (fval = %f, Q = %f)...\n',fval,Q);
        L = L*gamma;
        [Wp,P,sval] = ComputeQP(Xtrain,Ytrain,Z_old,L,z);
        f = 0.5*norm(Wp-Ytrain'*Xtrain,'fro')^2;
%         fval = f+lambda*sval;
%         Q = P+lambda*sval;
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
    
% 	fprintf('Iteration = %8d,  objective = %f\n',itr_counter, fval);
end
W = Wp;
return;

function [Wp,P,sval] = ComputeQP(X,Y,W,L,z)

[W1,delta_W] = ComputeGradStep(X,Y,W,L);

Wp=W;


Wp=sign(Wp).*max(abs(Wp)-z/L,0);


sval=z*sum(sum(abs(Wp)));

% tmp=ones(size(Wp))-alpha1*ones(size(Wp))./abs(Wp+1e-12);
% Wp=sign(sign(tmp)+ones(size(Wp))).*Wp; % proximal operator! Jun's code: S=sign(S).*max(abs(S)-alpha,0);
P = 0.5*norm(W-X'*Y,'fro')^2+trace(delta_W'*(Wp-W))+0.5*L*norm(Wp-W,'fro')^2;
return;

function [W1,delta_W] = ComputeGradStep(X,Y,W,L)

delta_W = ComputeDerivative(X,Y,W);
W1 = W-(1/L)*delta_W;

return;

function dev = ComputeDerivative(X,Y,W)

dev = W-X'*Y;

return;