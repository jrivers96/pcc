clist <- c("GSE21050.GPL570.txt", "GSE19784.GPL570.txt", "GSE13070.GPL570.txt", "GSE3526.GPL570.txt");

for (file in clist) {
	message(sprintf("file: %s", file));
	my_data <- as.matrix(read.table(file, header=FALSE, sep=" "));
	#my_data <- matrix( rnorm(100 * 10, mean = 0, sd=1), 10, 100);
	message(sprintf("dimensions: %d %d", nrow(my_data), ncol(my_data)));
	my_data_t <- t(my_data);	#transpose the matrix
	remove(my_data);

	#start the timer
	stime <- proc.time()
	cor_matrix <- cor(my_data_t, method="kendall");

	#stop the timer
	etime = proc.time() - stime
	print(etime);

	remove(cor_matrix);
	remove(my_data_t);
}

