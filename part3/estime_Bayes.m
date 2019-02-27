function [ x_bayes ] = estime_Bayes(Nmc,y,H,sigma2,lambda)
% Bayes method in order to approximate x 
% N: signal length
% Nmc, number of iterations
% lambda:= gamma(k,theta)
% sigma= variance
N=length(y);
x_bayes= zeros(N,1);
etha2=sigma2./sum(H.^2);

for i=1: Nmc
    
    x_inter=exprnd(1/lambda,N,1);
    
for j=1:N 
    if j==1
        Ej=y-H(:,2:end)*x_inter(2:end);
    elseif j==N
        Ej=y-H(:,1:end-1)*x_inter(1:end-1);
    else
        Ej=y-H(:,1:j-1)*x_inter(1:j-1)-H(:,j+1:end)*x_inter(j+1:end);
    end
    
    m=(Ej'*H(:,j)-sigma2*lambda)/norm(H(:,j))^2;
    if m<0
        x_inter(j)=0;
    else
        x_vec=randnt_ar1(m,etha2(j),1000);
        x_inter(j)= x_vec(1);
    end
end
    x_bayes=x_bayes+x_inter;

end

x_bayes=x_bayes/Nmc;

end
