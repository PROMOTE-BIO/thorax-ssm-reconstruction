close all; clear;
addpath('..');
%% input files
x   =sprintf('%s/../../win/point_cloud_0.txt',pwd);
y   =sprintf('%s/../../win/point_cloud_2.txt',pwd);
fnm =sprintf('%s/../../bcpd',                pwd);
fnw =sprintf('%s/../../win/bcpd.exe',        pwd);
if(ispc) bcpd=fnw; else bcpd=fnm; end;
%% parameters and execution 1
omg ='0.0';
bet ='0.7';
lmd ='100.0';
gma ='3';
zeta='1e-3';
K   ='200';
J   ='300';
c   ='1e-6';
n   ='50';
modo='-G geodesic,0.5,10,0.15';
down='B,10000,0.08'; %B-both by voxel, number of points wanted, size of voxel 

prm1=sprintf('-w%s -b%s -l%s -g%s -z%s',omg,bet,lmd,gma,zeta);
prm2=sprintf('-J%s -K%s -p',J,K);
% prm3=sprintf('-c%s -n%s -h -r1 ',c,n);
prm3=sprintf('%s', modo);
downsample=sprintf('-D%s',down);
cmd =sprintf('%s -x%s -y%s %s %s %s -sY %s',bcpd,x,y,prm1,prm2,prm3,downsample);
system(cmd); optpath3;

%% parameters  and execution 2
omg ='0.0';
bet ='0.7';
lmd ='100.0';
gma ='3';
K   ='200';
J   ='300';
c   ='1e-6';
n   ='50';
modo='-DB,3000,0.15';

prm1=sprintf('-w%s -b%s -l%s -g%s',omg,bet,lmd,gma);
prm2=sprintf('-J%s -K%s -p',J,K);
prm4=sprintf('-c%s -n%s',c,n);
prm3=sprintf('%s', modo);
cmd =sprintf('%s -x%s -y%s %s %s %s %s -sY',bcpd,x,y,prm1,prm2,prm3, prm4);
system(cmd); optpath3;

%% parameters  and execution 3
omg ='0.1';
bet ='2';
lmd ='2.0';
K   ='80';
J   ='300';
c   ='1e-6';
n   ='100';
modo='-DB,10000,0.08';

prm1=sprintf('-w%s -b%s -l%s',omg,bet,lmd);
prm2=sprintf('-J%s -K%s -p',J,K);
prm4=sprintf('-c%s -n%s',c,n);
prm3=sprintf('%s', modo);
cmd =sprintf('%s -x%s -y%s %s %s %s %s -svYP',bcpd,x,y,prm1,prm2,prm3, prm4);
system(cmd); optpath3;

%% parameters  and execution 4
omg ='0.0';
bet ='100';
lmd ='0.7';
gma='0.1';
K   ='100';
J   ='300';
c   ='1e-6';
n   ='100';
modo1='-G geodesic,0.5,10,0.15';
modo2='-DB,15000,0.15';


prm1=sprintf('-w%s -b%s -l%s -g%s',omg,bet,lmd,gma);
prm2=sprintf('-J%s -K%s -p',J,K);
prm4=sprintf('-c%s -n%s',c,n);
prm3=sprintf('%s %s', modo1, modo2);
cmd =sprintf('%s -x%s -y%s %s %s %s %s -sveYP',bcpd,x,y,prm1,prm2,prm3, prm4);
system(cmd); optpath3;

%% parameters  and execution 5
omg ='0.0';
bet ='100';
lmd ='0.7';
gma='0.1';
K   ='200';
J   ='300';
c   ='1e-6';
n   ='500';
modo2='-DB,3000,0.15';


prm1=sprintf('-w%s -b%s -l%s -g%s',omg,bet,lmd,gma);
prm2=sprintf('-J%s -K%s -p',J,K);
prm4=sprintf('-c%s -n%s',c,n);
prm3=sprintf('%s', modo2);
cmd =sprintf('%s -x%s -y%s %s %s %s %s -sveYP',bcpd,x,y,prm1,prm2,prm3, prm4);
system(cmd); optpath3;