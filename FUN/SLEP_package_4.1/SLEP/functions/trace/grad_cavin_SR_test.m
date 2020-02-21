function [W] = grad_cavin_SR_test(Xtrain,Lamda)

W_init = zeros(size(Xtrain,2),size(Xtrain,2));
Wp=W_init;
fval_old=0.5*norm(Wp-Xtrain'*Xtrain,'fro')^2;
Max_Iteration=1000;
epsilon=10^-5;
fval=0;
L=100;
Iteration=1;
while(abs((fval_old-fval)/fval_old)>epsilon||Iteration<Max_Iteration)
    fval_old=fval;
	[Wp sval]=ComputeGradStep(Xtrain,Wp,L,Lamda);
	fval=0.5*norm(Wp-Xtrain'*Xtrain,'fro')^2+Lamda*sval;
	Iteration=Iteration+1;
end
W=Wp;

return;


%%
function [W1 sval] = ComputeGradStep(X,W,L,lambda)

delta_W = ComputeDerivative(X,W);
W1 = W-(1/L)*delta_W;
W1=sign(W1).*max(abs(W1)-lambda/L,0);
sval=lambda*sum(sum(abs(W1)));

return;
%%%
function dev = ComputeDerivative(X,W)

dev = W-X'*X;

return;