function [x_all,x,iter,cost_function]= Newton_step_min_f0(f0,grad_f0,H,x_init,x0,alpha,epsilon)
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
    f0_iter = double(f0(x(1),x(2)));
    hess_f0= double(H(x(1),x(2)));
    grad_f0_iter= double(grad_f0(x(1),x(2)));
    d_iter = - inv(hess_f0)*grad_f0_iter;
    grad_f0_val= double(grad_f0(x(1),x(2)));
    x_vect = double(f0(x(1) - alpha*grad_f0_val(1),x(2) - alpha*grad_f0_val(2)));
    alpha_min =  alpha(x_vect==min(x_vect)); 
    x=x+alpha_min*d_iter;
    x_all = [x_all;x'];
    f0_iter_1 = double(f0(x(1),x(2)));
    cost_function(iter) = norm(f0_iter - f0_iter_1);
    diff = norm(f0_iter - f0_iter_1)>epsilon;
end
    
end