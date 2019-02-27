function [ prox ] = Prox_iterativesoft_thres(u,t)
% This function compute the soft threshold h(x)= ||x||1
% prox= prox_lambda_h(u)

prox=zeros(size(u));
prox(abs(u)<abs(t))=0;
prox(u>=t)=u(find(u>=t))-t;
prox(u<=-t)=u(u<=-t)+t;

end

