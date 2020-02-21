
function[W_new]=optimize_surrogate(W,Data,M,lambda, d)

% Data setup
% W FBN                 nROI*nROI
% Data Data Matirx         D*nROI
% M index of modular    nROI
% lambda hyper-parameter
% d hyper-parameter in subgradient_nuclearnorm


[d, nROI]=size(Data);
x0=reshape(W, nROI*nROI ,1);
 out=ncg(@(x) optimizeW(x,Data,M,lambda,d), x0, 'RelFuncTol', 1e-5, 'StopTol', 1e-8, ...)
'MaxFuncEvals', 1000, 'Display', 'final');
W_new=reshape(out.X, nROI, nROI);

end

