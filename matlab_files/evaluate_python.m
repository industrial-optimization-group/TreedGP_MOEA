function obj_vals = evaluate_python(population, init_folder, approaches, Strategy, Problem, M, nvars, sample_size, run, is_plot, plot_init)
    load(fullfile(init_folder,['DDMOPP_Params_' Strategy '_' Problem '_' num2str(M) '_' num2str(nvars) '_' num2str(sample_size) '.mat']));
    load(fullfile(init_folder,['Initial_Population_DDMOPP_' Strategy '_AM_' num2str(nvars) '_' num2str(sample_size) '.mat']));
    M=double(M);
    run = double(run)
    obj_vals = zeros(size(population,1),M);
    if plot_init == 1
        population = Initial_Population_DDMOPP(Run+1).c;
        obj_vals = zeros(size(population,1),M);
    end
    for samp = 1:size(population,1)
        obj_vals(samp,:) = distance_points_problem(population(samp,:),problem_parameters);        
    end
 
    if plot_init == 0
        X = population;
    else
        X = Initial_Population_DDMOPP(Run+1).c;
    end
    
    if is_plot == 1
        figure;
        if nvars > 2
            X_project = zeros(size(X,1),2);
            for i=1:size(X,1)
                X_project(i,:) = project_nD_point_to_2D(X(i,:),problem_parameters.projection_vectors(1,:),problem_parameters.projection_vectors(2,:));
            end
            X= X_project;
        end
        %figure;
        plot_dbmopp_2D_regions(problem_parameters,0,M,0, ...
            0);
        
        %hold on;
        box on;
        set(gca,'xTick',[]);
        set(gca,'yTick',[]);
        xlabel('');
        ylabel('');
        set(gcf, 'PaperSize', [5 5])
        saveas(gcf,fullfile('/home/amrzr/Work/Codes/data/test_runs/Test_DR_CSC_ARDMatern4/Plots_solutions',['Instance_' approaches '_' Strategy '_' Problem '_' num2str(M) '_' num2str(nvars) '_' num2str(sample_size) '_' num2str(run) '.pdf']))
        sz=30;
        figure;
        scatter(X(:,1),X(:,2),sz,'green','filled');
        hold on;
        scatter(X(:,1),X(:,2));
        non = P_sort(obj_vals,'first')==1;
        non_dom_pop = X(non,:);
        PF=obj_vals(non,:);
        scatter(non_dom_pop(:,1), non_dom_pop(:,2),'*','red')%,'linewidth',2)
        axis([-1,1,-1,1]);
        box on;
        axis square;
        set(gca,'xtick',[],'ytick',[]);
        set(gcf, 'PaperSize', [5 5])
        hold off;
        saveas(gcf,fullfile('/home/amrzr/Work/Codes/data/test_runs/Test_DR_CSC_ARDMatern4/Plots_solutions',['Solutions_' approaches '_' Strategy '_' Problem '_' num2str(M) '_' num2str(nvars) '_' num2str(sample_size) '_' num2str(run) '.pdf']))
        close all;
        %clear all;
        
    end
    %if write_initsamples == 1
    %    non = P_sort(obj_vals,'first')==1;
    %    non
    %    non_dom_pop = X(non,:);
    %    PF=obj_vals(non,:);
    %end
    %filename_solns = strcat(folder,'/','Run_', num2str(Run),'_soln');
    %dlmwrite(filename_solns,obj_vals);
    