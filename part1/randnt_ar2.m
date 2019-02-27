function [ X ] = randnt_ar2(m,sigma2,N)
% This function computes a vector following a normal law truncated

% M Computation
K=sqrt(pi*sigma2/2)*(1+erf(m/sqrt(sigma2*2)));
L=(sqrt(m^2+4*sigma2)-m)/(2*sigma2);
M=exp(-m^2/(2*sigma2)+sigma2*L^2)/(L*K);

% accept reject process
y=sqrt(sigma2)*(exprnd(m,N,1)/m);
f=truncated_normal(m,sigma2,y);
g = (1/L)*exp(-(1/L)*y);
U=unifrnd(0,1,N,1);
X=y(find(U<=(f./(g*M))));
end

