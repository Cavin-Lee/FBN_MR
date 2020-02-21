function[deltaA]=subgradient_nuclearnorm(A, s)
% An m * n matrix A
[m, n]=size(A);
[U, B, V]=svd(A);
% svalues=diag(B);
% s=sum(svalues>t);
% U1=U(:, 1:s);
% % U2=U(:, s+1:end);
% V1=V(1:s, :);
% % V2=V(s+1:end, :);
% 
% % B=zeros(m-s,n-s);
% % B=B/norm(B, 'fro');
% 
% % deltaA=U1*V1+U2*B*V2;
% deltaA=U1*V1;
if m <s || n <s
    deltaA=zeros(m, n);
else
    U1=U(:, 1:s);
    % U2=U(:, s+1:end);
    V1=V(1:s, :);
    % V2=V(s+1:end, :);

    % B=zeros(m-s,n-s);
    % B=B/norm(B, 'fro');

    % deltaA=U1*V1+U2*B*V2;
    deltaA=U1*V1;
end


end