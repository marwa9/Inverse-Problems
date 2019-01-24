function [ t ] = inv_CDF(m,sigma2,y )
%     a=1/(1+erf(m/sqrt(2*sigma2)));
%     b= erf(m/sqrt(2*sigma2))/a;
%     t1=(y-b)/a;
%     t= erfinv((t1-m)/sqrt(2*sigma2));

t =  (erfinv((y-m)/sqrt(2*sigma2))+ erf(m/sqrt(2*sigma2)))/(1+erf(m/sqrt(2*sigma2)));

end

