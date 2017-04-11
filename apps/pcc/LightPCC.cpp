/*
 * LightPCC.cpp
 *
 *  Created on: Mar 17, 2016
 *      Author: Yongchao Liu
 *		  School of Computational Science & Engineering
 *		  Georgia Institute of Technology, USA
 *		  URL: www.liuyc.org
 */

#include "LightPCC.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

static void printUsage() {
	fprintf(stderr, "LightPCC: parallel pairwise correlation computation using Intel Xeon Phi clusters\n");
	fprintf(stderr,
			"Usage: LightPCC (v1.0.15) command\n");
	fprintf(stderr, "Commands:\n");
	fprintf(stderr, "    pearson:      Pearson's correleation coefficient\n");
	fprintf(stderr, "    pearsonmkl:   Pearson's correlation coefficient using GEMM in Intel MKL\n");
	fprintf(stderr, "    spearman:     Spearman's rank correlation coefficient\n");
	fprintf(stderr, "    kendall:      Kendall rank correlation coefficient\n");
	fprintf(stderr, "    distance:     Distance correlation\n");
	fprintf(stderr, "    miadaptive:   Mututal information using adaptive partitioning estimator\n");
	fprintf(stderr, "\n");
}

int main(int argc, char* argv[]) {
	int ret;

	/*intialize MPI*/
#ifdef WITH_MPI
	MPI_Init(&argc, &argv);
#endif
	if(argc < 2){
		printUsage();
		return -1;
	}

	/*parse the arguments*/
	if(strcasecmp(argv[1], "pearson") == 0){
//		ret = lightPearsonR(argc - 1, argv + 1);
	}else if(strcasecmp(argv[1], "pearsonmkl") == 0){
		ret = lightPearsonRMKL(argc - 1, argv + 1);
	}else if(strcasecmp(argv[1], "spearman") == 0){
//		ret = lightSpearmanR(argc - 1, argv + 1);
	}else if(strcasecmp(argv[1], "kendall") == 0){
//		ret = lightKendallTau(argc - 1, argv + 1);
	}else if(strcasecmp(argv[1], "distance") == 0){
//		ret = lightDistanceCorr(argc - 1, argv + 1);
	}else if(strcasecmp(argv[1], "miadaptive") == 0){
//		ret = lightMIAdaptive(argc - 1, argv + 1);
	}else{
		fprintf(stderr, "Unsupported command: %s\n", argv[1]);
	}
	
	/*finalize MPI*/
#ifdef WITH_MPI
	MPI_Finalize();
#endif

	return ret;
}
