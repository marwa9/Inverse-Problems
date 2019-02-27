function [ X ] = randnt_ar1(m,sigma2,N)

M=2/(1+erf(m/sqrt(2*sigma2)));
y=sqrt(sigma2)*randn(N,1)+m;
f=truncated_normal(m,sigma2,y);
g = normpdf(y,m,sqrt(sigma2));
U=unifrnd(0,1,N,1);
X=y(find(U<=(f./(g*M))));
end

