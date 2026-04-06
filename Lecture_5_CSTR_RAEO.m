%% Description
% This code implements the ResNet Autoencoder with Ordered Variance (RAEO) for 
% continuous stirred tank reactor (CSTR) process. 
% The model extracted with RAEO is used for solving a 
% real-time optimization (RTO) problem for the CSTR system.
% For more details the AutoEncoder Paper can be referred: https://arxiv.org/abs/2402.14031

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
global A n m p Ez Vz
N=500;
V=48;Cd=1;k0=7.08*10^(10);E=29900;R=1.99;dH=-30000;
rho=50;Cp=0.75;U=150;Au=150;rhoj=62.3;Cpj=1;Vj=3.85;CA=0.2345;
D=randn(6,N);
 D(1,:)=40+kron(4*randn(1,N/100),ones(1,100));
 D(2,:)=0.5+kron(0.05*randn(1,N/50),ones(1,50));
 D(3,:)=530+0*kron(53*randn(1,N/20),ones(1,20));
 D(4,:)=56.626+kron(5.66*randn(1,N/50),ones(1,50));
 D(5,:)=530+0*kron(53*randn(1,N/50),ones(1,50));
 D(6,:)=10.6137+0*kron(1.06137*randn(1,N/20),ones(1,20));
Dn0=[0.2345*ones(1,N);600*ones(1,N);590*ones(1,N);0.3*ones(1,N)]+0.01*randn(4,N);
options1 = optimoptions('fsolve','MaxIterations',1e7,'MaxFunctionEvaluations',1e7,'FunctionTolerance',1e-1,'OptimalityTolerance',1e-2','Algorithm','trust-region-dogleg')
fun0=@(Dn)[(D(1,:)./(Au*Dn(4,:))).*(D(2,:)-Dn(1,:))-Cd*k0*Dn(1,:).*exp(-E./(R*Dn(2,:)));
 (D(1,:)./(Au*Dn(4,:))).*(D(3,:)-Dn(2,:))+(-dH/(rho*Cp))*Cd*k0*Dn(1,:).*exp(-E./(R*Dn(2,:)))-(U*Au*(Dn(2,:)-Dn(3,:)))./(Au*Dn(4,:)*rho*Cp);
 (D(4,:)/Vj).*(D(5,:)-Dn(3,:))+(U*Au*(Dn(2,:)-Dn(3,:)))/(Vj*rhoj*Cpj);
 (Dn(1,:).*(k0*Au*Dn(4,:))).*exp(-E./(R*Dn(2,:)))-D(6,:)];
[Dn,fvaln,flagn] = fsolve(fun0,Dn0,options1);

Xraw=[D;Dn]'+1*[0.4 0.005 5.3 0.566 5.3 0.1061 0.00234 6 5.9 0.003].*rand(N,10);
Xrawtr=Xraw(1:300,:);      % Training data
Xrawts=Xraw(301:500,:);    % Testing data

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
alpha=0.01;       % Hyperparameter 1
beta=0.5;         % Hyperparameter 2
w=3;              % Hyperparameter 3
Q=diag([w,w^2,w^3,w^4,w^5,w^6,w^7,w^8,w^9,w^10]);  
h=11; 
m=n;
A0=rand(h,2*(n+m)); % Initializing A   % A contain the weights (and biases) of encoder and decoder: A=[A1 A2 A3 A4]

%% Training RAEO  
% Defining the loss function for RAEO
fun = @(A)trace((Xtr-(A(:,n+2*m+1:2*n+2*m)'*tanh(A(:,n+m+1:n+2*m)*(Xtr+(A(:,n+1:n+m)'*tanh(A(:,1:n)*Xtr))))))'*(Xtr-(A(:,n+2*m+1:2*n+2*m)'*tanh(A(:,n+m+1:n+2*m)*(Xtr+(A(:,n+1:n+m)'*tanh(A(:,1:n)*Xtr)))))))+alpha*trace((Xtr+(A(:,n+1:n+m)'*tanh(A(:,1:n)*Xtr)))'*Q*(Xtr+(A(:,n+1:n+m)'*tanh(A(:,1:n)*Xtr))))+beta*trace(A'*A);
% Training the autoencoder RAEO
options = optimoptions('fminunc','MaxIterations',1e6,'MaxFunctionEvaluations',1e6,'OptimalityTolerance',1e-5);
[A,fval,flag]=fminunc(fun,A0,options);   
Ytr= (Xtr+(A(:,n+1:n+m)'*tanh(A(:,1:n)*Xtr))); 
p=6;        % Number of independent variables in the input data
Eytr=mean(Ytr(p+1:m,:)')';     % Computing mean values of residual latent variables for the training data
Vytr=[var(Ytr(1,:));var(Ytr(2,:));var(Ytr(3,:)),;var(Ytr(4,:));var(Ytr(5,:));var(Ytr(6,:));var(Ytr(7,:));var(Ytr(8,:));var(Ytr(9,:));var(Ytr(10,:))];
% In Xhattr computation, we will be replacing the residual latent variables with its mean Eytr
Xhattr=(A(:,n+2*m+1:2*n+2*m)'*tanh(A(:,n+m+1:n+m+p)*Ytr(1:p,:)+A(:,n+m+p+1:n+2*m)*Eytr));

%% Prediction
% Prediction on training data
% Solving the nonlinear equation f(Xp,Xr)=0 
Xrtr0=Xtr(p+1:n,:)+0.1*rand(n-p,Ntr);
fun0=@(Xrtr)Xrtr+A(:,n+p+1:n+m)'*tanh(A(:,1:p)*Xtr(1:p,:)+A(:,p+1:n)*Xrtr);
[Xrtr,fval0,flag0] = fsolve(fun0,Xrtr0,options1);
MSEprtr= mse(Xtr(p+1:n,:),Xrtr);
MSEretr=mse(Xtr,Xhattr);

% Prediction on testing data
Yts= (Xts+(A(:,n+1:n+m)'*tanh(A(:,1:n)*Xts)));
Xhatts=(A(:,n+2*m+1:2*n+2*m)'*tanh(A(:,n+m+1:n+m+p)*Yts(1:p,:)+A(:,n+m+p+1:n+2*m)*Eytr));
Xrts0=Xts(p+1:n,:)+0.1*rand(n-p,Nts);
fun0s=@(Xrts)Xrts+A(:,n+p+1:n+m)'*tanh(A(:,1:p)*Xts(1:p,:)+A(:,p+1:n)*Xrts);
[Xrts,fval0s,flag0s] = fsolve(fun0s,Xrts0,options1);
MSEprts= mse(Xts(p+1:n,:),Xrts);
MSErets=mse(Xts,Xhatts);

%% Solving RTO problem for the CSTR system using the RAEO model
fun = @(z)z(7);   % z is the decision vector which contains the operating points to be optimized.
z0=rand(10,1);    % Initial values of the operating points
%z0=[40 0.5 530 56.626 530 10.6137 0.2345 600 590 0.32];     % Initial values as steady-state operating points
Ez=Etr';        
Vz=Vtr';
F=[];g=[];Feq=[];geq=[]; nonlcon=@nlcon;
W=1;
lb=[40-W*4 0.5-W*0.05 530-W*53 56.626-W*5.6626 530-W*53 10.6137-W*1.06 0.2345-W*0.02345 600-W*60 590-W*59 0.32-W*0.03];
ub=[40+W*4 0.5+W*0.05 530+W*53 56.626+W*5.6626 530+W*53 10.6137+W*1.06 0.2345+W*0.02345 600+W*60 590+W*59 0.32+W*0.03];
options3 = optimoptions('fmincon','MaxIterations',1e7,'MaxFunctionEvaluations',1e7,'OptimalityTolerance',1e-3,'FunctionTolerance',1e-3,'Algorithm','interior-point');
z=fmincon(fun,z0,F,g,Feq,geq,lb,ub,nonlcon,options3);
z        % Optimal values of the operating points

%% Comparing the model extraction performance with PCA
p=6;    % Number of independent variables in the input data
cov_mat=cov(Xtr');
[vec,val]=eig(cov_mat);
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
  
Xr0=Xtr(p+1:n,:)+0.1*randn(n-p,Ntr);
options1 = optimoptions('fsolve','MaxIterations',1e7,'MaxFunctionEvaluations',1e6)
fun0=@(XrPCA)Pr'*[Xtr(1:p,:);XrPCA];
[XrPCA,fval0,flag0] = fsolve(fun0,Xr0,options1);
msep= mse(Xtr(p+1:n,:),XrPCA);
mser=mse(Xtr,Xhat);

Xhats=Pp*Pp'*Xts;
Xr0s=Xts(p+1:n,:)+0.1*randn(n-p,Nts);
options1 = optimoptions('fsolve','MaxIterations',1e7,'MaxFunctionEvaluations',1e6)
fun0=@(XrPCAs)Pr'*[Xts(1:p,:);XrPCAs];
[XrPCAs,fval0,flag0] = fsolve(fun0,Xr0s,options1);
mseps= mse(Xts(p+1:n,:),XrPCAs);
msers=mse(Xts,Xhats);

%% Plotting results
figure(1)
plot(Xtr(2,:),Xtr(9,:),'r.','LineWidth',.7)
hold on
plot(Xtr(2,:),Xrtr(3,:),'b.','LineWidth',.7)
hold on
plot(Xtr(2,:),XrPCA(3,:),'g.','LineWidth',.7)
hold on
xlabel('$x_{2}$','Interpreter','latex');ylabel('$x_{9}$','Interpreter','latex');
legend('$x_{2}\hspace{.1cm} \mbox{vs} \hspace{.1cm} x_{9}$','$x_{2} \hspace{.1cm} \mbox{vs} \hspace{.1cm} x_{pr_9}(RAEO)$','$x_{2} \hspace{.1cm} \mbox{vs} \hspace{.1cm} x_{pr_9}(PCA)$', 'Interpreter','latex');
grid on

function [c,ceq] =nlcon(z)
global A n m p Ez Vz
c=[];
ceq = (z(7:10,1)-Ez(7:10,1))./Vz(7:10,1)+(A(:,n+p+1:n+m)'*tanh(A(:,1:n)*((z-Ez)./Vz)));
end