clc
syms x1 x2
a = 0.5;
b= 8;
c= 0.55;
d=2;
x0_1 = 0.5;
x0_2 = 1;
x0 = [x0_1;x0_2];
x_init = [-0.8;-2.2]; % initialization
alpha = 0.1; % learning rate
epsilon1 = 10^-3; 
epsilon2 = 10^-6;

%%
% Functions definitions:
f0(x1,x2)=a*atan(d*x1^2+x2^2)+b*(x1-x0_1)^2+c*(x2-x0_2)^2; % f0 function 
G=gradient(f0, [x1 x2]); % f0 gradient
grad_f0(x1,x2) = G;
H(x1,x2) = jacobian(G, [x1 x2]); % La Hessienne de f0

%% Gradient Descent
% epsilon1
[x1_GD_all,x1_GD,iter1_GD,J1_GD]=min_f0(f0,grad_f0,x_init,x0,alpha,epsilon1);
% epsilon1
[x2_GD_all,x2_GD,iter2_GD,J2_GD]=min_f0(f0,grad_f0,x_init,x0,alpha,epsilon2);

%% Gradient Descent with steep size for alpha
alpha_step = logspace(-3,1,1000);
% epsilon1
[x1_GD_step_all,x1_GD_step,iter1_GD_step,J1_GD_step]=min_f0_step(f0,grad_f0,x_init,x0,alpha_step,epsilon1);
% epsilon2
[x2_GD_step_all,x2_GD_step,iter2_GD_step,J2_GD_step]=min_f0_step(f0,grad_f0,x_init,x0,alpha_step,epsilon2);

%% Newton Method without using alpha
% epsilon1
[x1_Newton_all,x1_Newton,iter1_Newton,J1_Newton]=Newton_min_f0(f0,grad_f0,H,x_init,x0,alpha,epsilon1);
% epsilon2
[x2_Newton_all,x2_Newton,iter2_Newton,J2_Newton]=Newton_min_f0(f0,grad_f0,H,x_init,x0,alpha,epsilon2);
%% Newton Method using step size for alpha
% epsilon1
[x1_Newton_step_all,x1_Newton_step,iter1_Newton_step,J1_Newton_step]=Newton_step_min_f0(f0,grad_f0,H,x_init,x0,alpha_step,epsilon1);
% epsilon2
[x2_Newton_step_all,x2_Newton_step,iter2_Newton_step,J2_Newton_step]=Newton_step_min_f0(f0,grad_f0,H,x_init,x0,alpha_step,epsilon2);
%% Performance Analysis
% Gradiant Descent for epsilon = 10^-3
figure;
subplot(2,2,1)
plot(J1_GD,'r')
hold on
plot(J1_GD_step,'b')
title('Cost Function of Gradient Descent for Epsilon = 10^-3')
legend('constant alpha','steep size alpha')
xlabel('Number of iterations')
ylabel('Cost Function')
% Gradiant Descent for epsilon = 10^-6
subplot(2,2,2)
plot(J2_GD,'r')
hold on
plot(J2_GD_step,'b')
title('Cost Function of Gradient Descent for Epsilon = 10^-6')
legend('constant alpha','steep size alpha')
xlabel('Number of iterations')
ylabel('Cost Function')
% Newton Method for epsilon = 10^-3
subplot(2,2,3)
plot(J1_Newton,'r')
hold on
plot(J1_Newton_step,'b')
title('Cost Function of Newton Method for Epsilon = 10^-3')
legend('constant alpha','steep size alpha')
xlabel('Number of iterations')
ylabel('Cost Function')
% Newton Method for epsilon = 10^-6
subplot(2,2,4)
plot(J2_Newton,'r')
hold on
plot(J2_Newton_step,'b')
title('Cost Function of Newton Method for Epsilon = 10^-6')
legend('constant alpha','steep size alpha')
xlabel('Number of iterations')
ylabel('Cost Function')
% Comparison between Gradient Descent and Newton Method for epsilon = 10^-3
% with constant alpha
figure;
subplot(2,2,1)
plot(J1_GD,'r')
hold on
plot(J1_Newton,'b')
title('Comparison of the speed of convergence of the cost function for Epsilon = 10^-3 and constant alpha')
legend('Gradient Descent','Newton Method')
xlabel('Number of iterations')
ylabel('Cost Function')
% Comparison between Gradient Descent and Newton Method for epsilon = 10^-3
% with variable alpha
subplot(2,2,2)
plot(J1_GD_step,'r')
hold on
plot(J1_Newton_step,'b')
title('Comparison of the speed of convergence of the cost function for Epsilon = 10^-3 and variable alpha')
legend('Gradient Descent','Newton Method')
xlabel('Number of iterations')
ylabel('Cost Function')
% Comparison between Gradient Descent and Newton Method for epsilon = 10^-6
% with constant alpha 
subplot(2,2,3)
plot(J2_GD,'r')
hold on
plot(J2_Newton,'b')
title('Comparison of the speed of convergence of the cost function for Epsilon = 10^-6 and constant alpha')
legend('Gradient Descent','Newton Method')
xlabel('Number of iterations')
ylabel('Cost Function')
% Comparison between Gradient Descent and Newton Method for epsilon = 10^-6
% with variable alpha
subplot(2,2,4)
plot(J2_GD_step,'r')
hold on
plot(J2_Newton_step,'b')
title('Comparison of the speed of convergence of the cost function for Epsilon = 10^-6 and variant alpha')
legend('Gradient Descent','Newton Method')
xlabel('Number of iterations')
ylabel('Cost Function')
%% 3D plots of f0 function for epsilon = 10^-3
% Gradient descent with constant alpha
x = meshgrid(linspace(min(x1_GD_all(:,1)),max(x1_GD_all(:,1)),100));
y = meshgrid(linspace(min(x1_GD_all(:,2)),max(x1_GD_all(:,2)),100));
Z = double(f0(x,y'));
figure;
subplot(2,2,1)
surf(x,y,Z)
title('Gradient Descent with constant alpha')

% Gradient descent without variable alpha
x = meshgrid(linspace(min(x1_GD_step_all(:,1)),max(x1_GD_step_all(:,1)),100));
y = meshgrid(linspace(min(x1_GD_step_all(:,2)),max(x1_GD_step_all(:,2)),100));
Z = double(f0(x,y'));
subplot(2,2,2)
surf(x,y,Z)
title('Gradient Descent with variant alpha')

% Newton Method without optimization
x = meshgrid(linspace(min(x1_Newton_all(:,1)),max(x1_Newton_all(:,1)),100));
y = meshgrid(linspace(min(x1_Newton_all(:,2)),max(x1_Newton_all(:,2)),100));
Z = double(f0(x,y'));
subplot(2,2,3)
surf(x,y,Z)
title('Newton Method without optimization')

% Newton Method with optimization
x = meshgrid(linspace(min(x1_Newton_step_all(:,1)),max(x1_Newton_step_all(:,1)),100));
y = meshgrid(linspace(min(x1_Newton_step_all(:,2)),max(x1_Newton_step_all(:,2)),100));
Z = double(f0(x,y'));
subplot(2,2,4)
surf(x,y,Z)
title('Newton Method with optimization')

%% 3D plots of f0 function for epsilon = 10^-6
% Gradient descent with constant alpha
x = meshgrid(linspace(min(x2_GD_all(:,1)),max(x2_GD_all(:,1)),100));
y = meshgrid(linspace(min(x2_GD_all(:,2)),max(x2_GD_all(:,2)),100));
Z = double(f0(x,y'));
figure;
subplot(2,2,1)
surf(x,y,Z)
title('Gradient Descent with constant alpha')

% Gradient descent without variable alpha
x = meshgrid(linspace(min(x2_GD_step_all(:,1)),max(x2_GD_step_all(:,1)),100));
y = meshgrid(linspace(min(x2_GD_step_all(:,2)),max(x2_GD_step_all(:,2)),100));
Z = double(f0(x,y'));
subplot(2,2,2)
surf(x,y,Z)
title('Gradient Descent with variant alpha')

% Newton Method without optimization
x = meshgrid(linspace(min(x2_Newton_all(:,1)),max(x2_Newton_all(:,1)),100));
y = meshgrid(linspace(min(x2_Newton_all(:,2)),max(x2_Newton_all(:,2)),100));
Z = double(f0(x,y'));
subplot(2,2,3)
surf(x,y,Z)
title('Newton Method without optimization')

% Newton Method with optimization
x = meshgrid(linspace(min(x2_Newton_step_all(:,1)),max(x2_Newton_step_all(:,1)),100));
y = meshgrid(linspace(min(x2_Newton_step_all(:,2)),max(x2_Newton_step_all(:,2)),100));
Z = double(f0(x,y'));
subplot(2,2,4)
surf(x,y,Z)
title('Newton Method with optimization')