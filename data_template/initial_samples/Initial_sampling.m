clear;
clc;
addpath(genpath('Public'));

%Problem = 'DTLZ';
Problem = 'DDMOPP';
Runs=35;
%Design='LHS';
%Design='Random';
Design='MVNORM';
sample_size = 1000;

no_vars = 10
%[~,Boundary,Coding] = P_objective('init',Problem,2,100);
Boundary(1,:) = ones(1,no_vars);
Boundary(2,:) = -1*ones(1,no_vars);
Boundary
no_var = size(Boundary,2);
%sample_size = 11*no_var - 1;

for i=1:Runs
    %no_var=10;
    Boundary=Boundary(:,1:no_var);

    ub = Boundary(1,:);
    lb = Boundary(2,:);

    if Design == "LHS"
        Xn = lhsdesign(sample_size,no_var);
        Population = bsxfun(@plus,lb,bsxfun(@times,Xn,(ub-lb)));
    elseif Design == "Random"
        lb_mat = repmat(lb,sample_size,1);
        ub_mat = repmat(ub,sample_size,1);
        Population = lb_mat + (ub_mat - lb_mat).*rand(sample_size,no_var);
    else
       mean_pnt = (Boundary(1,1) + Boundary(2,1))/2; 
       lb = Boundary(2,1)-mean_pnt;
       ub = Boundary(1,1)-mean_pnt;
       %mu = mean_pnt*ones(1,nvars);
       sigma = 0.0*ones(no_vars,no_vars)+0.1*eye(no_vars); 
       Population = mvrandn(lb*ones(1,no_vars),ub*ones(1,no_vars),sigma,sample_size)+mean_pnt;
       Population = Population';
    end
    
    if Problem == "DDMOPP"
        Initial_Population_DDMOPP(i).c = Population;
    else
        Initial_Population_DTLZ(i).c = Population;
    end
end

%save('Initial_Population_DTLZ_Random_AM_500.mat','Initial_Population_DTLZ');

%save('Initial_Population_WFG_Random_AM_new.mat','Initial_Population_WFG');

%save('Initial_Population_DTLZ_LHS_AM_1000.mat','Initial_Population_DTLZ');

if Problem == "DDMOPP"
    save(['Initial_Population_' Problem '_' Design '_AM_' num2str(no_vars) '_' num2str(sample_size) '.mat'],'Initial_Population_DDMOPP');
else
    save(['Initial_Population_' Problem '_' Design '_AM_' num2str(no_vars) '_' num2str(sample_size) '.mat'],'Initial_Population_DTLZ');
end
