%{ 
*******************************************************************************************
Evaluation Code
Automatic Cephalometric X-Ray Landmark Detection Challenge, ISBI 2014.
In this challenge, there are 19 landmarks to detect in total. 
Two main criteria will be considered to evaluate the performance of all the proposed methods.
1. Mean radial errors with standard deviation 
2. Successful detection rate / Percentage of landmarks detected with various ranges of accuracy


Detailed information:
http://www-o.ntust.edu.tw/~cweiwang/celph/  > Evaluation Metrics
Copyright (c) Prof Ching-Wei Wang
*******************************************************************************************
%}

auto_path = ['']; % give your result path
list_auto_path = [auto_path '*.csv'];
list_auto = dir(list_auto_path); % Set auto-csv file path
R = zeros(length(list_auto),19);
MRE = zeros(19);
SD = zeros(19);
file_path = [auto_path 'StatisticResult.txt'];
fid2 = fopen(file_path,'w');
fprintf(fid2,'Landmark ID, Distance(MRE, SD), Successful detection rates with accuracy of less than 2.0mm, 2.5mm, 3.0mm, and 4.0mm\n');
hit = zeros(19);
miss = zeros(19);
auto_t = zeros(length(list_auto)*8,1);
manual_t = zeros(length(list_auto)*8,1);
for y = 1:(length(list_auto))
    fprintf('Load patient %d \n',y);
    path_manualtxt = ['' num2str(y+300, '%03d') '.txt'] % give the manual result path
    path_autotxt = [auto_path num2str(y+300, '%03d') '.csv']
    manualtxt = csvread(path_manualtxt);
    autotxt = csvread(path_autotxt);
    %autotxt = load(path_autotxt);
   
    for x = 1:19
        R(y,x) = sqrt((manualtxt(x,1)-autotxt(x,1))^2 + (manualtxt(x,2)-autotxt(x,2))^2);
    end
    for x = 20:27
        auto_t(((y-1)*8)+x-19,1) = autotxt(x,1);
        manual_t(((y-1)*8)+x-19,1) = manualtxt(x,1);
    end
%     if (y == 21)
%         manualtxt(x,1)
%         autotxt(x,1)
%         manualtxt(x,2)
%         autotxt(x,2)
%         R(y,:)
%         pause(5000000);
%     end
   
end

%Mean radial errors with standard deviation

for x = 1:19
    sumR = 0;
    MRE(x) = sum(R(:,x))/length(list_auto);
    for y = 1:length(list_auto)
        sumR = sumR + (power((R(y,x)-MRE(x)),2));
    end
    SD(x) = sqrt (sumR / (length(list_auto)));
end

for x = 1:19
    for i = 1:4
        if  i == 1
            accur_mm = 2;
        elseif i == 2
            accur_mm = 2.5;
        elseif i == 3
            accur_mm = 3;
        else
            accur_mm = 4;
        end
        numhit(x,i) = sum(sign(find(R(:,x) <= accur_mm*10)));
    end
   fprintf(fid2,'L%d, %.3f, %.3f, %.2f%%, %.2f%%, %.2f%%, %.2f%%\n',x,MRE(x),SD(x),numhit(x,1)/length(list_auto)*100,numhit(x,2)/length(list_auto)*100,numhit(x,3)/length(list_auto)*100,numhit(x,4)/length(list_auto)*100);
end
fprintf(fid2,'AVERAGE, %.3f, %.3f, %.2f%%, %.2f%%, %.2f%%, %.2f%%\n',sum(MRE(1:19,1))/19,sum(SD(1:19,1))/19,(sum(numhit(1:19,1))/19)/length(list_auto)*100,(sum(numhit(1:19,2))/19)/length(list_auto)*100,(sum(numhit(1:19,3))/19)/length(list_auto)*100,(sum(numhit(1:19,4))/19)/length(list_auto)*100);
fclose(fid2);


