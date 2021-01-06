function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
alpha=1;
n=size(theta);
z=X*theta;%mx1
h=(1./(1+exp(-1*z)));
grad=((1/m)*(X'*(h-y)))+((lambda/m)*theta);
grad(1)=((1/m)*(X(:,1)'*(h-y)));
%J=(1/(2*m))*((X'*((h-y).^2))+(lambda*(theta.^2)));
J=((1/m)*((-y'*log(h))-((1-y)'*(log(1-h)))))+((lambda/(2*m))*sum(theta(2:n,1).^2));
%J(1)=((1/m)*((-y(1)'*log(h(1)))-((1-y(1))'*(log(1-h(1))))));
%theta=theta-(alpha*((1/m)*(X'*(h-y))+((lambda/m)*theta)));
%theta(1)=theta(1)-(alpha*(1/m)*(X(:,1)'*(h-y)));


% =============================================================

end
