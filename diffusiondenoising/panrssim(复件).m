clear all;
close all;
clc

%�������Ƶ�txt�ļ����������ַ���ֻ������
data=load('C:\Users\�����\Desktop\����ʵ��ԭͼ\3.26\ps1.txt');
x=1:999;
y=data(:,1);
plot(x,y,'b');
xlabel('Iterations')
ylabel('PSNR')


data1=load('C:\Users\�����\Desktop\����ʵ��ԭͼ\3.26\result_3.25_ssim.txt');
x1=1:999;
y1=data1(:,1);
plot(x1,y1,'r');
xlabel('Iterations')
ylabel('SSIM')
