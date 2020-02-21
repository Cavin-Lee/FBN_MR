function W=FBNMR(X ,J, lambda, d)

% Data setup
% W FBN                 nROI*nROI
% X Data Matirx         D*nROI
% J index of modular    nROI
% lambda hyper-parameter
% d hyper-parameter in subgradient_nuclearnorm


[n, m]=size(X);


Wt=rand(m);
Flag=true;
iteration=1;
while Flag && iteration <100
    
    Wtplus=optimize_surrogate(Wt,X,J,lambda, d);
    
    if norm(Wtplus-Wt, 'fro')/norm(Wt, 'fro') <10^-8
        Flag=false;
    else
       iteration=iteration+1; 
    end
     Wt=Wtplus;
    
end

W=Wt;
end


