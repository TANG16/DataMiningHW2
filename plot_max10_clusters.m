function plot_max10_clusters( data, IDX )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
figure,
hold on;

for i=1:size(IDX,1)
    if IDX(i,1) == 1
        plot(data(i,1),data(i,2),'m+');
    elseif IDX(i,1) == 2
        plot(data(i,1),data(i,2),'g+');
    elseif IDX(i,1) == 3
        plot(data(i,1),data(i,2),'r+');
    elseif IDX(i,1) == 4
        plot(data(i,1),data(i,2),'y+');
    elseif IDX(i,1) == 5
        plot(data(i,1),data(i,2),'b+');
     elseif IDX(i,1) == 6
        plot(data(i,1),data(i,2),'k+');
    elseif IDX(i,1) == 7
        plot(data(i,1),data(i,2),'c+');
    elseif IDX(i,1) == 8
        plot(data(i,1),data(i,2),'y*');
    elseif IDX(i,1) == 9
        plot(data(i,1),data(i,2),'b*');
    elseif IDX(i,1) == 10
        plot(data(i,1),data(i,2),'m*');
    else
        plot(data(i,1),data(i,2),'b-');        
    end
end

hold off;
end

