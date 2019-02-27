function [x_all,x,iter,cost_function]=min_f0_step(f0,grad_f0,x_init,x0,alpha,epsilon)
% x_init : get the initialization value of x
% x : is the value that minimizes f0
% diff : Boolean variable that indicates if the stop criterion is achived
% or not
% iter : number of iteration needed for convergence
% x_all : a Matrix of number of iterations by 2 where we save all the
% x values  computed during the while loop
% cost_function : an error function that controls the convergence of
% gradient descent
x=x_init;
diff = true;
iter = 0;
x_all = [];
cost_function = [];
while diff
    iter = iter+1;
    f0_k_1= double(f0(x(1),x(2)));
    grad_f0_val= double(grad_f0(x(1),x(2)));
    x_vect = double(f0(x(1) - alpha*grad_f0_val(1),x(2) - alpha*grad_f0_val(2)));
    alpha_min=  alpha(x_vect==min(x_vect));       
    x1 = x(1) - alpha_min*double(grad_f0_val(1));
    x2 = x(2) - alpha_min*double(grad_f0_val(2));
    f0_k = double(f0(x1,x2));
    cost_function(iter) = norm(f0_k - f0_k_1);
    diff = norm(f0_k - f0_k_1)>epsilon;
    x=[x1;x2];   
    x_all = [x_all; [x1 x2]];
end
    
end