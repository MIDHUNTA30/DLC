%% Description
% This code implements the Autoencoder with Ordered Variance (AEO) for a
% five-variable dataset. 
% For more details the AEO Paper can be referred: https://arxiv.org/abs/2402.14031

%% Notations
% Number of input variables:                 n 
% Number of neurons in the hidden layer:     h 
% Number of latent variables:                m 
% Number of observations or samples of data: N 
% Number of observations in training data:   Ntr 
% Number of observations in testing data:    Nts 
% Loss Function:                             J= J1+J2+J3 
% Reconstruction Error term :                J1 
% Variance Regularization term :             J2 
% Weight Regulairzation term :               J3

%% Tuning parameters in the algorithm
% Number of neurons in the hidden layer:                h 
% Scaling factor of the variance regularization term:   alpha 
% Scaling factor of the weight regularization term:     beta
% Weighting matrix of the variance regularization term: Q

%%  Generating the input data
clear all
close all
rand('state',1)
randn('state',1)
N=500;
for j=1:N
    Xraw(j,1:3)=1-2*rand(1,3);
    Xraw(j,4)=sin(3*Xraw(j,1))+0.1*randn(1);
    Xraw(j,5)=Xraw(j,2)+tan(0.5*Xraw(j,3))+0.1*randn(1);
end
Xrawtr=Xraw(1:300,:);    % training data
Xrawts=Xraw(301:500,:);  % testing data

% Normalizing the training data
[Ntr,n]=size(Xrawtr);
Etr=mean(Xrawtr);
Vtr=std(Xrawtr);
Xtr0=Xrawtr-Etr(ones(Ntr,1),:);  % Data is mean-centered
Xtrn=Xtr0./Vtr(ones(Ntr,1),:);   % Data is now normalized
Xtr=Xtrn';                    % This makes observations as column vectors and Xtr is of size n by Ntr

% Normalizing the testing data
[Nts,n]=size(Xrawts);
Ets=mean(Xrawts);
Vts=std(Xrawts);
Xts0=Xrawts-Ets(ones(Nts,1),:);  % Data is mean-centered
Xtsn=Xts0./Vts(ones(Nts,1),:);   % Data is now normalized
Xts=Xtsn';                    % This makes observations as column vectors and Xts is of size n by Nts

%% Selecting tuning parameters 
h=6;           
alpha=0.001;        % Hyperparameter 1
beta=0.5;           % Hyperparameter 2
w=10;               % Hyperparameter 3
epsilon=0.01; % Variance threshold
Q=diag([w,w^2,w^3,w^4,w^5]);  
m=n;
A0=rand(h,2*(n+m)); % Initializing A   % A contain the weights (and biases) of encoder and decoder: A=[A1 A2 A3 A4] 
%% Training AEO  
% Defining the loss function for AEO
fun = @(A)trace((Xtr-(A(:,n+2*m+1:2*n+2*m)'*tanh(A(:,n+m+1:n+2*m)*((A(:,n+1:n+m)'*tanh(A(:,1:n)*Xtr))))))'*(Xtr-(A(:,n+2*m+1:2*n+2*m)'*tanh(A(:,n+m+1:n+2*m)*((A(:,n+1:n+m)'*tanh(A(:,1:n)*Xtr)))))))+alpha*trace(((A(:,n+1:n+m)'*tanh(A(:,1:n)*Xtr)))'*Q*((A(:,n+1:n+m)'*tanh(A(:,1:n)*Xtr))))+beta*trace(A'*A);
options = optimoptions('fminunc','MaxIterations',1e6,'MaxFunctionEvaluations',1e6,'OptimalityTolerance',1e-5);
[A,fval,flag]=fminunc(fun,A0,options);   
Ytr= (A(:,n+1:n+m)'*tanh(A(:,1:n)*Xtr)); 
Vytr=[var(Ytr(1,:));var(Ytr(2,:));var(Ytr(3,:)),;var(Ytr(4,:));var(Ytr(5,:))];
for s=n:-1:1
   if Vytr(s)>epsilon
       p=s;      % Number of significant variables
       break;
   end    
end       
Eytr=mean(Ytr(p+1:m,:)')';    % Computing mean values of residual latent variables for the training data
% In Xhattr computation, we will be replacing the residual latent variables with its mean Eytr
Xhattr=(A(:,n+2*m+1:2*n+2*m)'*tanh(A(:,n+m+1:n+m+p)*Ytr(1:p,:)+A(:,n+m+p+1:n+2*m)*Eytr));

%% Prediction
% Prediction on training data
% Solving the nonlinear equation f(Xp,Xr)=0 
Xrtr0=Xtr(4:5,:)+0.1*rand(n-p,Ntr);
options1 = optimoptions('fsolve','MaxIterations',1e7,'MaxFunctionEvaluations',1e7,'FunctionTolerance',1e-2,'Algorithm','trust-region-dogleg')
funtr=@(Xrtr)A(:,n+p+1:n+m)'*tanh(A(:,1:p)*Xtr(1:p,:)+A(:,p+1:n)*Xrtr)-Eytr;
[Xrtr,fvaltr,flagtr] = fsolve(funtr,Xrtr0,options1);
MSEprtr= mse(Xtr(p+1:n,:),Xrtr);
MSEretr=mse(Xtr,Xhattr);

% Prediction on testing data
Yts= ((A(:,n+1:n+m)'*tanh(A(:,1:n)*Xts)));
Xhatts=(A(:,n+2*m+1:2*n+2*m)'*tanh(A(:,n+m+1:n+m+p)*Yts(1:p,:)+A(:,n+m+p+1:n+2*m)*Eytr));
Xrts0=Xts(p+1:n,:)+0.1*rand(n-p,Nts);
funts=@(Xrts)A(:,n+p+1:n+m)'*tanh(A(:,1:p)*Xts(1:p,:)+A(:,p+1:n)*Xrts)-Eytr;
[Xrts,fvalts,flagts] = fsolve(funts,Xrts0,options1);
MSEprts= mse(Xts(p+1:n,:),Xrts);
MSErets=mse(Xts,Xhatts);


%% Comparing the model extraction performance with PCA
p=3;     % Number of independent variables in the input data
corr_mat=cov(Xtr');
[vec,val]=eig(corr_mat);
eigen_val = diag(val);
for i=1:n
    eigenvalues(i)=eigen_val(n-i+1);
    v(:,i)=vec(:,n-i+1);
end

for i=1:n
    contrib(i)=eigenvalues(i)/sum(eigenvalues);
end
Pp= v(:,1:p);
Pr= v(:,p+1:n);
Xhat=Pp*Pp'*Xtr;

% Solving the nonlinear equation f(Xp,Xr)=0       
Xr0=Xtr(p+1:n,:)+0.1*randn(n-p,Ntr);
fun0=@(XrPCA)Pr'*[Xtr(1:p,:);XrPCA];
[XrPCA,fval0,flag0] = fsolve(fun0,Xr0,options1);
msep= mse(Xtr(p+1:n,:),XrPCA);
mser=mse(Xtr,Xhat);

%% Plotting results
figure(1)
    plot(Xtr(1,:),Xtr(4,:),'r.','LineWidth',.8)
    hold on
    plot(Xtr(1,:),Xrtr(1,:),'b.','LineWidth',.8)
    hold on
    plot(Xtr(1,:),XrPCA(1,:),'g.','LineWidth',.8)
    hold on
    xlabel('$x_{1}$','Interpreter','latex');ylabel('$x_{4}$','Interpreter','latex');
    legend('$x_{1}\hspace{.1cm} \mbox{vs} \hspace{.1cm} x_{4}$','$x_{1} \hspace{.1cm} \mbox{vs} \hspace{.1cm} x_{pr_4}(AEO)$', '$x_{1} \hspace{.1cm} \mbox{vs} \hspace{.1cm} x_{pr_4}(PCA)$', 'Interpreter','latex');
    grid on