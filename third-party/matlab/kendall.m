display('GSE3526.GPL570.txt')
A = importdata('GSE3526.GPL570.txt');
B=transpose(A);
clearvars A
tic
corr(B, B, 'type', 'Kendall');
toc
clearvars B

display('GSE13070.GPL570.txt')
A = importdata('GSE13070.GPL570.txt');
B=transpose(A);
clearvars A
tic
corr(B, B, 'type', 'Kendall');
toc
clearvars B

display('GSE19784.GPL570.txt')
A = importdata('GSE19784.GPL570.txt');
B=transpose(A);
clearvars A
tic
corr(B, B, 'type', 'Kendall');
toc
clearvars B

display('GSE21050.GPL570.txt')
A = importdata('GSE21050.GPL570.txt');
B=transpose(A);
clearvars A
tic
corr(B, B, 'type', 'Kendall');
toc
clearvars B

display('to exit')
exit
