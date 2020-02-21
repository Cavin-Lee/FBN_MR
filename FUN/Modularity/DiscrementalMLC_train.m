function[Wtplus]=DiscrementalMLC_train(Xtrain, Ytrain, J, lambda, s)
[n, m]=size(Ytrain);
[~, d]=size(Xtrain);
X=cell(m, 1);
for i=1:m
    tempindex=find(Ytrain(:, i)>0);
    X{i}=Xtrain(tempindex, :);    
end

Wt=eye(d, m);
Flag=true;
iteration=1;
while Flag && iteration <10
    
    Wtplus=optimize_surrogate(Xtrain, X, Ytrain, J, Wt, lambda, s);
    
    if norm(Wtplus-Wt, 'fro')/norm(Wt, 'fro') <10^-3
        Flag=false;
    else
       iteration=iteration+1; 
       Wt=Wtplus;
    end
     
    
end
end

