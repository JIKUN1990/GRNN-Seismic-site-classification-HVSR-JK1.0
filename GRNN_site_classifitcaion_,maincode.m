%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% written by Dr.Ji Kun %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Using GRNN for site classifitcaion 
% Modified on 2020/12/16;
% Ref: General regression neural network (GRNN)-based seismic site classification scheme for Chinese seismic code using HVSR curves 
% Authors: Ji Kun ; Ren Yefei*; Ruizhi Wen; Zhu ChuanBin; Liu Ye; 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc
clear
% In our code, the testdata(target HV curves) and the traindata(pattern
% curves) have same number of periods (0.02s to 5.0s, 94 points)
% Period=[0.0200000000000000,0.0220000000000000,0.0250000000000000,0.0290000000000000,0.0300000000000000,0.0320000000000000,0.0350000000000000,0.0360000000000000,0.0400000000000000,0.0420000000000000,0.0440000000000000,0.0450000000000000,0.0460000000000000,0.0480000000000000,0.0500000000000000,0.0550000000000000,0.0600000000000000,0.0650000000000000,0.0670000000000000,0.0700000000000000,0.0750000000000000,0.0800000000000000,0.0850000000000000,0.0900000000000000,0.0950000000000000,0.100000000000000,0.110000000000000,0.120000000000000,0.130000000000000,0.133000000000000,0.140000000000000,0.150000000000000,0.160000000000000,0.170000000000000,0.180000000000000,0.190000000000000,0.200000000000000,0.220000000000000,0.240000000000000,0.250000000000000,0.260000000000000,0.280000000000000,0.290000000000000,0.300000000000000,0.320000000000000,0.340000000000000,0.350000000000000,0.360000000000000,0.380000000000000,0.400000000000000,0.420000000000000,0.440000000000000,0.450000000000000,0.460000000000000,0.480000000000000,0.500000000000000,0.550000000000000,0.600000000000000,0.650000000000000,0.667000000000000,0.700000000000000,0.750000000000000,0.800000000000000,0.850000000000000,0.900000000000000,0.950000000000000,1,1.10000000000000,1.20000000000000,1.30000000000000,1.40000000000000,1.50000000000000,1.60000000000000,1.70000000000000,1.80000000000000,1.90000000000000,2,2.20000000000000,2.40000000000000,2.50000000000000,2.60000000000000,2.80000000000000,3,3.20000000000000,3.40000000000000,3.50000000000000,3.60000000000000,3.80000000000000,4,4.20000000000000,4.40000000000000,4.60000000000000,4.80000000000000,5]
load HVresult-KiKnet
testdata=IHV_raw % Cl-site class HVSR curves in KiKnet using as testdata for example
load Four_reference_curves % reference curevs for pattern layers in GRNN 
curve11=Four_reference_curves(1:94,1)'
curve22=Four_reference_curves(1:94,2)'
curve33=Four_reference_curves(1:94,3)'
curve44=Four_reference_curves(1:94,4)'

spread=1.0   % spread factor
threshold=0.5  %threshold

traindata=[curve11;curve22;curve33;curve44];
trainresult=[1;2;3;4] % site class CL-I, CL-IIa,CL-IIb,CL-III
t=ind2vec(trainresult');% class 1-->[1,0,0,0]; class IIa-->[0,1,0,0]; class IIb-->[0,0,1,0]; class III-->[0,0,0,1];  
ss=sqrt(1/(2*spread));
net=newgrnn(traindata',t,ss);		% GRNN is builit using the MATLAB Nerual Network Toolbox
%net=newpnn(traindata',t,spread);		% PNN

for i=1:length(testdata(:,1))
targetHV=testdata(i,:)
y=sim(net,targetHV');% Simulation using the GRNN
% y refers to the probability results of different site classes using GRNN
a(i,:)=[y(1),y(2)+y(3),y(4)] % the probability of the CL-II is the summation of the CL-IIa and CL-IIb

% Judgement using the threshold value
    if a(i,1)>a(i,2) && a(i,1)>a(i,3) && a(i,1)>=threshold
       result(i,1)=1;
    elseif a(i,2)>a(i,1) && a(i,2)>a(i,3) && a(i,2)>=threshold
        result(i,1)=2;
    elseif a(i,3)>a(i,1) && a(i,3)>a(i,2) && a(i,3)>=threshold
       result(i,1)=3;
    else
       result(i,1)=-999;
    end

% yc=vec2ind(y)'		% output layer
% result1(i,1)=yc
end
