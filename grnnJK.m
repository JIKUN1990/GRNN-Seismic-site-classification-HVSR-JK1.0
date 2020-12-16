 function [S,D,ya] = grnnJK(traindata,trainresult,x,spread)
% 
% spread=0.8236
% x=testdata(4,:)';
[N1,N2]=size(traindata);  % traindata N1£ºÎ¬Êı N2£º¸öÊı
N4=length(trainresult)  ;
for i=1:N4
v(i)=sum((x-traindata(:,i)).^2);
end
for i=1:N4
   yr(i) = exp(-v(i)/(2*spread.^2));
end
D=sum(yr)
for i=1:N4
ya(i,1)=trainresult(i)*yr(i)/(D);
end
S=sum(ya);

% end