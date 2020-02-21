

function [W] =  accel_grad_PSR_cavin_test(Xtrain,Lambda)

W_init = Xtrain'*Xtrain;
Wp=W_init;
fval_old=0.5*norm(Wp-Xtrain'*Xtrain,'fro')^2;
Max_Iteration=1000;
epsilon=10^-5;
fval=0;
L=10;
Iteration=1;

while(abs((fval_old-fval))>epsilon||Iteration<Max_Iteration)
    fval_old=fval;
	Wp=Wp-(Wp-Xtrain'*Xtrain)/L;
	Wp=sign(Wp).*max(abs(Wp)-Lambda/L,0);
    fval=0.5*norm(Wp-Xtrain'*Xtrain,'fro')^2+Lambda*sum(sum(abs(Wp)));
	Iteration=Iteration+1;
    
end
sum(sum(abs(Wp-Xtrain'*Xtrain)))
W=Wp;
return;
