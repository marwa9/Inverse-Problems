clc

%1&2) Normal law troncatedrandnt inversion

% Signal parameters

N=50000; % signal length
m=0.5; % signal mean 
sigma2=1; % signal variance 

% theoretical probability density function
x_vector=linspace(0,10,1000);
[ f_x ] = truncated_normal(m,sigma2,x_vector);

% empirical histogram
X_tronc=randnt_inversion(m,sigma2,N);
[hist_X,x]=hist(X_tronc,50);


figure(2)
bar(x,hist_X/trapz(x,hist_X));hold on
plot(x_vector,f_x,'r')
title('Normal law troncated: emperical histogram vs law probability, m=0.5, sigma2=1')
legend('emperical histogram',' law probability')

%% 3) Accept-reject process with Normal distribution 

%a) We compute M in randnt_ar1



%b) accept reject with normal distribution 

X_accept_reject =randnt_ar1(m,sigma2,N);
[hist_accept_reject,x_accept_reject]=hist(X_accept_reject,50);

figure;
bar(x_accept_reject,hist_accept_reject/trapz(x_accept_reject,hist_accept_reject))
hold on
plot(x_vector,f_x,'r')
title('Normal law troncated: emperical histogram vs law probability, m=0.5, sigma2=1')
legend('emperical histogram',' law probability')


%% 4) Accept-reject process with exponential distribution 

%a) We compute M in randnt_ar2


%b) accept reject with exponential distribution 

X_exp=randnt_ar2(m,sigma2,N);
[hist_exp,x_exp]=hist(X_exp,50);

figure;
bar(x_exp,hist_exp/trapz(x_exp,hist_exp))
hold on
plot(x_vector,f_x,'r')
title('Normal law troncated: emperical histogram vs law probability, m=0.5, sigma2=1')
legend('emperical histogram',' law probability')


%% 5)
% We fix m=0.5 and we vary sigma2
sigma2_vector = linspace(0,10,500);
exp_vector = [];
norm_pdf_vector = [];
for i=1:length(sigma2_vector)
    norm_pdf_vector(i) =length(randnt_ar1(m,sigma2_vector(i),N));
    exp_vector(i) = length(randnt_ar2(m,sigma2_vector(i),N));
end
figure;
subplot(2,1,1)
plot(sigma2_vector,norm_pdf_vector,'r')
hold on 
plot(sigma2_vector,exp_vector,'b')
title('Comparison of acceptance ratio between normal and exponential laws for constant m and variable sigma2') 
legend('normal law','exponential law')

% We fix sigma2=1 and we vary m
m_vector = linspace(0,10,500);
exp_vector = [];
norm_pdf_vector = [];
for i=1:length(sigma2_vector)
    norm_pdf_vector(i) =length(randnt_ar1(m_vector(i),sigma2,N));
    exp_vector(i) = length(randnt_ar2(m_vector(i),sigma2,N));
end
subplot(2,1,2)
plot(m_vector,norm_pdf_vector,'r')
hold on 
plot(m_vector,exp_vector,'b')
title('Comparison of acceptance ratio between normal and exponential laws for constant sigma2 and variable m')
legend('normal law','exponential law')

% Now we vary m and sigma2 in the same time
m_matrix = meshgrid(linspace(0,10,100));
sigma2_matrix = meshgrid(linspace(0,10,100));
norm_pdf_matrix = zeros(size(m_matrix));
exp_matrix = zeros(size(m_matrix));
for i=1:100
    for j=1:100
        norm_pdf_matrix(i,j)=length(randnt_ar1(m_matrix(i,j),sigma2_matrix(i,j),N));
        exp_matrix(i,j)=length(randnt_ar2(m_matrix(i,j),sigma2_matrix(i,j),N));
    end
end
subplot(1,2,1)
surf(m_matrix,sigma2_matrix,norm_pdf_matrix)
title('Normal law with variable m and sigma2')
xlabel('mean')
ylabel('sigma2')
zlabel('acceptance ratio')
subplot(1,2,2)
surf(m_matrix,sigma2_matrix,exp_matrix)
title('Exponential law with variable m and sigma2')
xlabel('mean')
ylabel('sigma2')
zlabel('acceptance ratio')

