function [ x,nb ] =  proximal_gradient_descent( H,y,t,beta,T,positivity)
%T le nombre maximal d'itérations
% positivity: 1, activate the positivity; 0 desactivate the posivity.

x=zeros(size(y));
nb=0;
r = y;
alpha=0.1;
thres=norm(r);
while (norm(r)>=thres && nb<T)
    v=x-alpha*(-H'*y+H'*H*x);
    x0=x;
    x=Prox_iterativesoft_thres(v,t);
    if (positivity)
        x(x<0)=0;
    end
    thresh=norm(r)-norm(-2*H'*y+2*H'*H*(x-x0))+(1/2*alpha)*norm(x-x0);
    r=y-H*x;
    alpha=alpha*beta;
    nb=nb+1;
end
end

