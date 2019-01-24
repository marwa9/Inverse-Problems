function [ f_x ] = truncated_normal(m,sigma2,X)
% This function computes a truncated normal law density

k=sqrt(pi*sigma2/2)*(1+erf(m/sqrt(2*sigma2)));
f_x= (1/k)*exp(-(X-m).^2/(2*sigma2));
f_x(find(X<0))=0;
end

