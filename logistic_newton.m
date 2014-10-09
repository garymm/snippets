% Stanford, CS 229, 2014 Autumn, Homework 1, Problem 1(b)

% Newton's method:
% theta := theta - (f(theta) / f'(theta))
% For maximum likelihood,
% theta := theta - l'(theta) / l''(theta)
% theta := theta - H^{-1} * grad(l(theta))

function [theta, ll] = logistic_newton(X,Y)

% rows of X are training samples
% rows of Y are corresponding 0/1 values

% output ll: vector of log-likelihood values at each iteration
% ouptut theta: parameters

[m,n] = size(X);

max_iters = 500;

X = [ones(size(X, 1), 1), X]; % append col of ones for intercept term

theta = zeros(size(X, 2), 1);  % initialize theta
ll = zeros(1, max_iters);

function p = partial(theta_index)
  p = sum(Y - sigmoid(X * theta) .* X(:, theta_index), 1);
end

function res = hess(r, c)
	hx = sigmoid(X * theta);
  res = sum(-X(:, r) .* X(:, c) .* hx .* (1 - hx), 1);
end

H_indices = square_array_indices(length(theta));

grad_norm = inf;

for k = 1:max_iters
	grad = transpose(arrayfun('partial', 1:length(theta)));
	hessian_array = arrayfun('hess', H_indices(:, 1), H_indices(:, 2));
	H = reshape(hessian_array, [length(theta), length(theta)]);
	theta = theta - inv(H) * grad;
	hx = sigmoid(X * theta)
	ll(k) = sum(Y .* log(hx) + (1 - Y) .* log(1 - hx));
end

end
