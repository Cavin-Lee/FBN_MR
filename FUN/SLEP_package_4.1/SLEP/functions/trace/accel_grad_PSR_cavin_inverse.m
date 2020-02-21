

function [W] = accel_grad_PSR_cavin_inverse(A,lambda)
Xtrain=A;
Lamda=lambda;
W_init = zeros(size(Xtrain,2),size(Xtrain,2));
Wp=W_init;
fval_old=0.5*norm(Wp-(Xtrain'*Xtrain)^-1,'fro')^2;
Max_Iteration=1000;
epsilon=10^-5;
fval=0;
L=100;
Iteration=1;
while(abs((fval_old-fval)/fval_old)>epsilon||Iteration<Max_Iteration)
    fval_old=fval;
	[Wp sval]=ComputeQS(Xtrain,Wp,L,Lamda);
	fval=0.5*norm(Wp-(Xtrain'*Xtrain)^-1,'fro')^2+sval;
	Iteration=Iteration+1;
    
end
W=Wp;

return;

function [Wp sval]= ComputeQS(X,W,L,lambda)

[W1] = ComputeGradStep(X,W,L);

Wp=sign(W1).*max(abs(W1)-lambda/L,0);
sval=lambda*sum(sum(abs(Wp)));
return


%%
function [W1] = ComputeGradStep(X,W,L)

delta_W = ComputeDerivative(X,W);
W1 = W-(1/L)*delta_W;

return;
%%%
function dev = ComputeDerivative(X,W)

dev = W-(X'*X)^-1;

return;