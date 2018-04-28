sizeofPatient = 100;
a1=[];a2=[];a3=[];a4=[];a5=[];a6=[];a7=[];a8=[];
m1=[];m2=[];m3=[];m4=[];m5=[];m6=[];m7=[];m8=[];
for i = 1:sizeofPatient*8
    if mod(i,8)==1
        a1(length(a1)+1)=auto_t(i);
        m1(length(m1)+1)=manual_t(i);
    end
    if mod(i,8)==2
        a2(length(a2)+1)=auto_t(i);
        m2(length(m2)+1)=manual_t(i);
    end
    if mod(i,8)==3
        a3(length(a3)+1)=auto_t(i);
        m3(length(m3)+1)=manual_t(i);
    end
    if mod(i,8)==4
        a4(length(a4)+1)=auto_t(i);
        m4(length(m4)+1)=manual_t(i);
    end
    if mod(i,8)==5
        a5(length(a5)+1)=auto_t(i);
        m5(length(m5)+1)=manual_t(i);
    end
    if mod(i,8)==6
        a6(length(a6)+1)=auto_t(i);
        m6(length(m6)+1)=manual_t(i);
    end
    if mod(i,8)==7
        a7(length(a7)+1)=auto_t(i);
        m7(length(m7)+1)=manual_t(i);
    end
    if mod(i,8)==0
        a8(length(a8)+1)=auto_t(i);
        m8(length(m8)+1)=manual_t(i);
    end
    
end
fid1 = fopen('Lindner-result.txt','w');
fprintf(fid1,'1. ANB\n');
 CM2=confusionmat(m1,a1)
fprintf(fid1,'Diagonal Average: %3.2f %% \n',  (CM2(1,1)/sum(CM2(1,:))+CM2(2,2)/sum(CM2(2,:))+CM2(3,3)/sum(CM2(3,:)))/3 * 100)
for i=1:3       %¶]¦C
   for j=1:3       %¶]¦æ
      fprintf(fid1,'%03.2f%% \t',CM2(i,j)/sum(CM2(i,:))*100);
   end
   fprintf(fid1,'\r\n');
end

 fprintf(fid1,'\n2. SNB\n');
 CM2=confusionmat(m2,a2)
fprintf(fid1,'Diagonal Average: %3.2f %% \n', (CM2(1,1)/sum(CM2(1,:))+CM2(2,2)/sum(CM2(2,:))+CM2(3,3)/sum(CM2(3,:)))/3 * 100)
for i=1:3       %¶]¦C
   for j=1:3       %¶]¦æ
      fprintf(fid1,'%03.2f%% \t',CM2(i,j)/sum(CM2(i,:))*100);
   end
   fprintf(fid1,'\r\n');
end
 
 
 fprintf(fid1,'\n3. SNA\n');
 CM3=confusionmat(m3,a3)
 fprintf(fid1,'Diagonal Average: %3.2f %% \n', (CM3(1,1)/sum(CM3(1,:))+CM3(2,2)/sum(CM3(2,:))+CM3(3,3)/sum(CM3(3,:)))/3 * 100)
for i=1:3       %¶]¦C
   for j=1:3       %¶]¦æ
      fprintf(fid1,'%03.2f%% \t',CM3(i,j)/sum(CM3(i,:))*100);
   end
   fprintf(fid1,'\r\n');
end
 
 fprintf(fid1,'\n4. ODI\n');
 CM4=confusionmat(m4,a4)
 fprintf(fid1,'Diagonal Average: %3.2f %% \n', (CM4(1,1)/sum(CM4(1,:))+CM4(2,2)/sum(CM4(2,:))+CM4(3,3)/sum(CM4(3,:)))/3 * 100)
for i=1:3       %¶]¦C
   for j=1:3       %¶]¦æ
      fprintf(fid1,'%03.2f%% \t',CM4(i,j)/sum(CM4(i,:))*100);
   end
   fprintf(fid1,'\r\n');
end
  
 fprintf(fid1,'\n5. APDI\n');
 CM5=confusionmat(m5,a5)
 fprintf(fid1,'Diagonal Average: %3.2f %% \n', (CM5(1,1)/sum(CM5(1,:))+CM5(2,2)/sum(CM5(2,:))+CM5(3,3)/sum(CM5(3,:)))/3 * 100)
for i=1:3       %¶]¦C
   for j=1:3       %¶]¦æ
      fprintf(fid1,'%03.2f%% \t', CM5(i,j)/sum(CM5(i,:))*100 );
   end
   fprintf(fid1,'\r\n');
end
 
 fprintf(fid1,'\n6. FHI\n');
 CM6=confusionmat(a6,m6)
  fprintf(fid1,'Diagonal Average: %3.2f %% \n', (CM6(1,1)/sum(CM6(1,:))+CM6(2,2)/sum(CM6(2,:))+CM6(3,3)/sum(CM6(3,:)))/3 * 100)
for i=1:3       %¶]¦C
   for j=1:3       %¶]¦æ
      fprintf(fid1,'%03.2f%% \t',CM6(i,j)/sum(CM6(i,:))*100);
   end
   fprintf(fid1,'\r\n');
end
 
 
fprintf(fid1, '\n7. FMA\n');
 CM7=confusionmat(a7,m7)
  fprintf(fid1,'Diagonal Average: %3.2f %% \n', (CM7(1,1)/sum(CM7(1,:))+CM7(2,2)/sum(CM7(2,:))+CM7(3,3)/sum(CM7(3,:)))/3 * 100)
for i=1:3       %¶]¦C
   for j=1:3       %¶]¦æ
      fprintf(fid1,'%03.2f%% \t',CM7(i,j)/sum(CM7(i,:))*100);
   end
   fprintf(fid1,'\r\n');
end
 
 
 
fprintf(fid1,'\n8. MW \n');
 CM8=confusionmat(m8,a8)
 fprintf(fid1,'Diagonal Average: %3.2f %% \n',(CM8(1,1)/sum(CM8(1,:))+CM8(2,2)/sum(CM8(2,:))+CM8(3,3)/sum(CM8(3,:)))/3 * 100)
for i=1:3       %¶]¦C
   if (i == 2)
           fprintf(fid1, 'null \tnull \t null \t null \t \n', 0);
   end
    for j=1:3       %¶]¦æ
       if (j == 2)
           fprintf(fid1, 'null \t', 0);
       end
      fprintf(fid1,'%03.2f%% \t',CM8(i,j)/sum(CM8(i,:))*100);
    end
   fprintf(fid1,'\r\n');
end
 fclose(fid1);