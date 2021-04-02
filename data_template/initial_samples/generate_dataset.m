clear all;


Mobj=[2,3,4,5,6,8,10]; %,5];
num_vars = [10] %,8,10];
%managements = {'1','7','8'}; %,'Offline_m5_ibea','Offline_ei2_ibea'}; %'Offline_m3','Offline_m5','Offline_m6','Offline_m3_ei','Offline_m3_ei2'}; %,'Offline_m3','Offline_m4'}; %'Offline_m2','Offline_m1','Offline_m3'}; %,
Strategies = {'MVNORM', 'LHS'};
Problems = {'P2'};
Runs=35;
%Design='LHS';
%Design='Random';
sample_size = 2000;

%load Initial_Population_DDMOPP_LHS_AM_109.mat
nsamples = sample_size;



for m = 1:length(Mobj)
    M=Mobj(m);
    for nv = 1:length(num_vars)
    nvars = num_vars(nv);
        for Prob = 1:length(Problems)
            Problem = Problems{Prob};

              for strat = 1:length(Strategies)
                Strategy=Strategies{strat};
                load(['Initial_Population_DDMOPP_' Strategy '_AM_' num2str(nvars) '_' num2str(sample_size) '.mat'])
                num_objectives=M
                num_dimensions=nvars
                curvature=false
                number_of_disconnected_set_regions=1
                number_of_local_fronts=0
                number_of_dominance_resistance_regions=0
                number_of_discontinuous_regions=0
                varying_density=false
                non_identical_pareto_sets=false
                varying_objective_ranges=false
                fill_space=false
                plot_wanted=false
                random_seed=1
                problem_parameters = distance_problem_generator(num_objectives,num_dimensions,...
                    curvature, number_of_disconnected_set_regions,...
                    number_of_local_fronts,number_of_dominance_resistance_regions, ...
                    number_of_discontinuous_regions,...
                    varying_density,non_identical_pareto_sets,varying_objective_ranges, ...
                    fill_space,plot_wanted,random_seed);
                obj_vals = zeros(nsamples,M);
                    for Run = 1:35

                        Population = Initial_Population_DDMOPP(Run).c;
                        for samp = 1:nsamples
                            obj_vals(samp,:) = distance_points_problem(Population(samp,:),problem_parameters);        
                        end
                        %obj_vals
                        Obj_vals_DDMOPP(Run).c = obj_vals;

                    end
                    Obj_vals_DDMOPP
                    save(strcat('DDMOPP_Params_',Strategy,'_',Problem,'_',num2str(M),'_',num2str(nvars),'_',num2str(sample_size),'.mat'), 'problem_parameters')
                    save(strcat('Obj_vals_DDMOPP_',Strategy,'_AM_',Problem,'_', num2str(M), '_', num2str(nvars),'_',num2str(sample_size),'.mat'),'Obj_vals_DDMOPP');
              end
        end
    end
end

