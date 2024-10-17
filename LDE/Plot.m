clear;
Color = [237,177,32;
    217,83,25;
    255,153,200;
    77,190,238;
    162,20,47;
    125,46,143;
    119,172,48;]./255;
node_shape =['+-';'x-';'<-';'o-';'s-';'p-';'*-';];
%% ORL
% load("cv5_Eigen_ORL");
% acc_eig = mean(Acc_cv5,1); % 1*len_d
% load("cv5_Fish_ORL");
% acc_fish = mean(Acc_cv5,1);
% load("cv5_LDE_ORL");
% acc_LDE = mean(Acc_cv5,1);
% load("cv5_kLDE_ORL");
% acc_kLDE = mean(Acc_cv5,1);
% load("cv5_LPP_ORL");
% acc_LPP = mean(Acc_cv5,1);
% load("cv5_kLPP_ORL");
% acc_kLPP = mean(Acc_cv5,1);
% load("cv5_exNPE_ORL");
% acc_exNPE = mean(Acc_cv5,1);
% figure;
% plot(lb_d:ub_d, acc_eig,  node_shape(1,:), 'MarkerFaceColor','w','Linewidth', 1.5, 'Color', Color(1,:)); hold on;
% plot(lb_d:ub_d, acc_fish, node_shape(2,:), 'MarkerFaceColor','w','Linewidth', 1.5, 'Color', Color(2,:)); hold on;
% plot(lb_d:ub_d, acc_LDE,  node_shape(3,:), 'MarkerFaceColor','w','Linewidth', 1.5, 'Color', Color(3,:)); hold on;
% plot(lb_d:ub_d, acc_kLDE, node_shape(4,:), 'MarkerFaceColor','w','Linewidth', 1.5, 'Color', Color(4,:)); hold on;
% plot(lb_d:ub_d, acc_LPP,  node_shape(5,:), 'MarkerFaceColor','w','Linewidth', 1.5, 'Color', Color(5,:)); hold on;
% plot(lb_d:ub_d, acc_kLPP, node_shape(6,:), 'MarkerFaceColor','w','Linewidth', 1.5, 'Color', Color(6,:)); hold on;
% plot(lb_d:ub_d, acc_exNPE,node_shape(7,:), 'MarkerFaceColor','w','Linewidth', 1.5, 'Color', Color(7,:)); hold on;
% xlabel('Dims', 'Fontsize', 16);
% ylabel('Classification accuracy (%)', 'Fontsize', 16);
% legend('Eigenfaces', 'Fisherfaces', 'LDE', 'kLDE','LPP','kLPP','exNPE',...
%     'Location', 'southeast', 'Fontsize', 16);
% hold off;


% load("cv5_2dLDE_ORL");
% acc_2dLDE = mean(Acc_cv5,3); % len_d1*len_d2
% load("cv5_2dLPP_ORL");
% acc_2dLPP = mean(Acc_cv5,3); % len_d1*len_d2


%% Yale
load("cv5_Eigen_Yale");
acc_eig = mean(Acc_cv5,1); % 1*len_d
load("cv5_Fish_Yale");
acc_fish = mean(Acc_cv5,1);
load("cv5_LDE_Yale");
acc_LDE = mean(Acc_cv5,1);
load("cv5_kLDE_Yale");
acc_kLDE = mean(Acc_cv5,1);
load("cv5_LPP_Yale");
acc_LPP = mean(Acc_cv5,1);
load("cv5_kLPP_Yale");
acc_kLPP = mean(Acc_cv5,1);
load("cv5_exNPE_Yale");
acc_exNPE = mean(Acc_cv5,1);
figure;
plot(lb_d:ub_d, acc_eig,  node_shape(1,:), 'MarkerFaceColor','w','Linewidth', 1.5, 'Color', Color(1,:)); hold on;
plot(lb_d:ub_d, acc_fish, node_shape(2,:), 'MarkerFaceColor','w','Linewidth', 1.5, 'Color', Color(2,:)); hold on;
plot(lb_d:ub_d, acc_LDE,  node_shape(3,:), 'MarkerFaceColor','w','Linewidth', 1.5, 'Color', Color(3,:)); hold on;
plot(lb_d:ub_d, acc_kLDE, node_shape(4,:), 'MarkerFaceColor','w','Linewidth', 1.5, 'Color', Color(4,:)); hold on;
plot(lb_d:ub_d, acc_LPP,  node_shape(5,:), 'MarkerFaceColor','w','Linewidth', 1.5, 'Color', Color(5,:)); hold on;
plot(lb_d:ub_d, acc_kLPP, node_shape(6,:), 'MarkerFaceColor','w','Linewidth', 1.5, 'Color', Color(6,:)); hold on;
plot(lb_d:ub_d, acc_exNPE,node_shape(7,:), 'MarkerFaceColor','w','Linewidth', 1.5, 'Color', Color(7,:)); hold on;
xlabel('Dims', 'Fontsize', 16);
ylabel('Classification accuracy (%)', 'Fontsize', 16);
legend('Eigenfaces', 'Fisherfaces', 'LDE', 'kLDE','LPP','kLPP','exNPE',...
    'Location', 'southeast', 'Fontsize', 16);
hold off;




