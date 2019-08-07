%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
%&%                                                   %&%
%&%             �.07 - ������� ���������              %&%
%&%                2� ����� ��������                  %&%
%&%                                                   %&%
%&%                   ���������:                      %&%
%&%                                                   %&%
%&%             ������ �����     �.�.:372             %&%
%&%         ����������� �������� �.�.:359             %&%
%&%                                                   %&%
%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
%%
clear all;
clc;

% ������� ��� ������� ���������/������������
load ('BatsiDimitriadisData.mat');

%% ������ 1: ���������� �� �� ������ Bagging ��� �� random forest:
% ----------------------------------------------------------------

% �) ������� Bagging ��� Ensemble Learning:
% 
% ->M������ ��� ���������� ����������:
% �������� ����� ��� cross-validation
% ->������ ����������� ��� ������:
% 25,50,75 ��� 100 ����������
Bagging25Model = fitcensemble(X,classes,'Method','Bag','NumLearningCycles',25,'CrossVal','on');

Bagging50Model = fitcensemble(X,classes,'Method','Bag','NumLearningCycles',50,'CrossVal','on');

Bagging75Model = fitcensemble(X,classes,'Method','Bag','NumLearningCycles',75,'CrossVal','on');

Bagging100Model = fitcensemble(X,classes,'Method','Bag','NumLearningCycles',100,'CrossVal','on');

kfL25 = kfoldLoss(Bagging25Model,'Mode','cumulative');
figure;
plot(kfL25);
ylabel('10-fold Misclassification rate');
xlabel('Learning cycle');
disp(min(kfL25));

kfL50 = kfoldLoss(Bagging50Model,'Mode','cumulative');
figure;
plot(kfL50);
ylabel('10-fold Misclassification rate');
xlabel('Learning cycle');
disp(min(kfL50));

kfL75 = kfoldLoss(Bagging75Model,'Mode','cumulative');
figure;
plot(kfL75);
ylabel('10-fold Misclassification rate');
xlabel('Learning cycle');
disp(min(kfL75));

kfL100 = kfoldLoss(Bagging100Model,'Mode','cumulative');
figure;
plot(kfL100);
ylabel('10-fold Misclassification rate');
xlabel('Learning cycle');
disp(min(kfL100));

% ��� ����������� k, �� ���������� k-fold
% cross-validation ������ �� ��������������� 
% ����������� ��� ����������� ��� ���� �������:
% cvp = cvpartition(PartitionSize,'KFold',k)
% fitcensemble(...,'CVPartition',cvp,...)
% ---------------------------------------


%  �) ������� Random Forest ��� Ensemble Learning:
% 
% ->M������ ��� ���������� ����������:
% �������� ����� ��� out-of-bag error
% ->������ ������� ����������� ��� ������:
% 25,50,75 ��� 100 ����������
RF25Model = TreeBagger(25,X,classes,'OOBPrediction','On','MinLeafSize',5,'Method','classification');

RF50Model = TreeBagger(50,X,classes,'OOBPrediction','On','MinLeafSize',5,'Method','classification');

RF75Model = TreeBagger(75,X,classes,'OOBPrediction','On','MinLeafSize',5,'Method','classification');

RF100Model = TreeBagger(100,X,classes,'OOBPrediction','On','MinLeafSize',5,'Method','classification');

% � ������� 'OOBPrediction' �� 'on' , �������� ���
% ���������� ��� �� ����� ������������ �����
% out-of-bag ���� ������ ��������. 
% ���� � ���������� ������ �� �������������� ���
% ��� ��������� oobPrediction ��� �� ������������
% �� ����������� ��� ������������� �������� ���� 
% ������� ��� Ensemble ��������.

% ������� ����������� ��������� ��������������
% ��� ����������� ��������
view(RF25Model.Trees{14},'Mode','graph')
view(RF50Model.Trees{30},'Mode','graph')
view(RF75Model.Trees{55},'Mode','graph')
view(RF100Model.Trees{80},'Mode','graph')

figure;
oobError25 = oobError(RF25Model);
plot(oobError25)
xlabel '������ �������';
ylabel 'Out-of-bag error �����������';
min(oobError25)
 
figure;
oobError50 = oobError(RF50Model);
plot(oobError50)
xlabel '������ �������';
ylabel 'Out-of-bag error �����������';
min(oobError50)

figure;
oobError75 = oobError(RF75Model);
plot(oobError75)
xlabel '������ �������';
ylabel 'Out-of-bag error �����������';
min(oobError75)

figure;
oobError100 = oobError(RF100Model);
plot(oobError100)
xlabel '������ �������';
ylabel 'Out-of-bag error �����������';
min(oobError100)

% � �������� �� ���� ����������� ���
% ������������ ���� 1� ����� ��������
% ���� ����� ���� ������ �������

%% ������ 2: �����������:
% -----------------------

% ������� ��� ������� ���������/������������
% ���� ��� �������������
X1 = load('3rings.mat');
X1 = X1.X;

X2 = load('3wings.mat');
X2 = X2.X;

X3 = load('4rectangles.mat');
X3 = X3.X;

X4 = load('5clusters.mat');
X4 = X4.X;

X5 = load('5Gaussians.mat');
X5 = X5.X;

X6 = load('5rings.mat');
X6 = X6.X;

X7 = load('7clusters.mat');
X7 = X7.X;

GR = load('gaussian_rings.mat');
GR = GR.X;

clear X;

% �������� ��� ������� ������������ ��� �������� 
% ������ ���������, �� ������ ������ �� ����������:
% 1) k-means

% �� k = 3 ��� �� ������ '3rings'
% ��� '3wings' ���������

[K_3M_X1,C1] = kmeans(X1,3);

plot_max10_clusters(X1,K_3M_X1);

[K_3M_X2,C2] = kmeans(X2,3);

plot_max10_clusters(X2,K_3M_X2);

% �� k = 4 ��� �� ������ '4rectangles'

[K_4M_X3,C3] = kmeans(X3,4);

plot_max10_clusters(X3,K_4M_X3);

% �� k = 5 ��� �� ������ '5clusters',
% '5Gaussians' ��� '5rings' ����������

[K_5M_X4,C4] = kmeans(X4,5);

plot_max10_clusters(X4,K_5M_X4);

[K_5M_X5,C5] = kmeans(X5,5);

plot_max10_clusters(X5,K_5M_X5);

[K_5M_X6,C6] = kmeans(X6,5);

plot_max10_clusters(X6,K_5M_X6);

% �� k = 7 ��� �� ������ '7clusters'

[K_7M_X7,C7] = kmeans(X7,7);

plot_max10_clusters(X7,K_7M_X7);
% ------------------------------
% 2) Agglomerative Clustering (single/average link)

% ���������� �� ������� � ����������
% (�������� ����� �� 'chebychev','mahalanobis',
% 'hamming' �.�.), ��� �� ������ ���� ������������:

% �� 3 clusters �� ������ '3rings' ��� '3wings' ����������
Z1S = linkage(X1,'single','euclidean');
Z1A = linkage(X1,'average','euclidean');

c1S = cluster(Z1S,'maxclust',3);
c1A = cluster(Z1A,'maxclust',3);

figure;
scatter(X1(:,1),X1(:,2),5,c1S)
figure;
scatter(X1(:,1),X1(:,2),5,c1A)

Z2S = linkage(X2,'single','euclidean');
Z2A = linkage(X2,'average','euclidean');
c2S = cluster(Z2S,'maxclust',3);
c2A = cluster(Z2A,'maxclust',3);

figure;
scatter(X2(:,1),X2(:,2),5,c2S)
figure;
scatter(X2(:,1),X2(:,2),5,c2A)

% �� 4 clusters �� ������ '4rectangles'
Z3S = linkage(X3,'single','euclidean');
Z3A = linkage(X3,'average','euclidean');
c3S = cluster(Z3S,'maxclust',4);
c3A = cluster(Z3A,'maxclust',4);

figure;
scatter(X3(:,1),X3(:,2),5,c3S)
figure;
scatter(X3(:,1),X3(:,2),5,c3A)

% �� 5 clusters �� ������ '5clusters',
% '5Gaussians' ��� '5rings' ����������
Z4S = linkage(X4,'single','euclidean');
Z4A = linkage(X4,'average','euclidean');
c4S = cluster(Z4S,'maxclust',5);
c4A = cluster(Z4A,'maxclust',5);

figure;
scatter(X4(:,1),X4(:,2),5,c4S)
figure;
scatter(X4(:,1),X4(:,2),5,c4A)

Z5S = linkage(X5,'single','euclidean');
Z5A = linkage(X5,'average','euclidean');
c5S = cluster(Z5S,'maxclust',5);
c5A = cluster(Z5A,'maxclust',5);

figure;
scatter(X5(:,1),X5(:,2),5,c5S)
figure;
scatter(X5(:,1),X5(:,2),5,c5A)

Z6S = linkage(X6,'single','euclidean');
Z6A = linkage(X6,'average','euclidean');
c6S = cluster(Z6S,'Maxclust',5);
c6A = cluster(Z6A,'Maxclust',5);

figure;
scatter(X6(:,1),X6(:,2),5,c6S)
figure;
scatter(X6(:,1),X6(:,2),5,c6A)

% �� 7 clusters �� ������ '7clusters'
Z7S = linkage(X7,'single','euclidean');
Z7A = linkage(X7,'average','euclidean');
c7S = cluster(Z7S,'Maxclust',7);
c7A = cluster(Z7A,'Maxclust',7);

figure;
scatter(X7(:,1),X7(:,2),5,c7S)
figure;
scatter(X7(:,1),X7(:,2),5,c7A)
% ----------------------------
% 3) Spectral Clustering (sigma=0.1, 0.5 ��� 1)

% �� 3 clusters �� ������ '3rings' ��� '3wings' ����������
SpecX11 = spectral(X1,3,0.1);
SpecX12 = spectral(X1,3,0.5);
SpecX13 = spectral(X1,3,1);

SpecX21 = spectral(X2,3,0.1);
SpecX22 = spectral(X2,3,0.5);
SpecX23 = spectral(X2,3,1);

% �� 4 clusters �� ������ '4rectangles'
SpecX31 = spectral(X3,4,0.1);
SpecX32 = spectral(X3,4,0.5);
SpecX33 = spectral(X3,4,1);

% �� 5 clusters �� ������ '5clusters',
% '5Gaussians' ��� '5rings' ����������
SpecX41 = spectral(X4,5,0.1);
SpecX42 = spectral(X4,5,0.5);
SpecX43 = spectral(X4,5,1);

SpecX51 = spectral(X5,5,0.1);
SpecX52 = spectral(X5,5,0.5);
SpecX53 = spectral(X5,5,1);

SpecX61 = spectral(X6,5,0.1);
SpecX62 = spectral(X6,5,0.5);
SpecX63 = spectral(X6,5,1);

% �� 7 clusters �� ������ '7clusters'
SpecX71 = spectral(X7,7,0.1);
SpecX72 = spectral(X7,7,0.5);
SpecX73 = spectral(X7,7,1);
% -------------------------
% ��� �� ������ 'gaussian_rings', ����������
% �� ��������� sigma ����� ��� �����������
% [0.1, 0.4] ��� �� ����� � ���������� spectral
% ����� �� ����� ���� ������������. ����, ��
% ������������� �� ����:
for i=0.1:0.05:0.4
    
    SpecGR = spectral(GR,5,i);
    % ���������� �� 5, �� 6 � 7, �������� �� �����
    % �� ��� ������� �� ��������� sigma ��� �������
    % ��������, �������� �� ��������� �� ����
    % ��� ���������� ��� 0.01 ��� �� ������
    % �� ��� ����� �����������, ���������� ��������
end
% -------------------------------------------------
% ��� �� ������ ��������� ��� ��� ��������� ����������,
% ��������� ���� ��� evalclusters �� ������ ��� ������
% ���� �������. ���������� ������������ ����� � k-means
% ��� �������� ����������� �� silhouette.
evaX2 = evalclusters(X2,'kmeans','silhouette','KList',[1:7]);
evaX3 = evalclusters(X3,'kmeans','silhouette','KList',[2:9]);
evaX4 = evalclusters(X4,'kmeans','silhouette','KList',[2:8]);
evaX5 = evalclusters(X5,'kmeans','silhouette','KList',[3:10]);
evaX7 = evalclusters(X7,'kmeans','silhouette','KList',[4:12]);

disp(evaX2.OptimalK);
disp(evaX3.OptimalK);
disp(evaX4.OptimalK);
disp(evaX5.OptimalK);
disp(evaX7.OptimalK);
