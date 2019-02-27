function [ X ] = randnt_inversion(m,sigma2,N)


X=erfinv(unifrnd(0,1,N,1)*(1+erf(m/sqrt(2*sigma2)))-erf(m/sqrt(2*sigma2)))*sqrt(2*sigma2)+m;
end 