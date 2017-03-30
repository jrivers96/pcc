/*
 * NetFusion.cpp
 *
 *  Created on: Sep 07, 2016
 *      Author: Yongchao Liu
 *		  School of Computational Science & Engineering
 *		  Georgia Institute of Technology, USA
 *		  URL: www.liuyc.org
 */

#include "NetFusion.h"
#include <EXPMatrixReader.hpp>
#include <PearsonRMKL.hpp>
#include <KendallTau.hpp>
#include <SpearmanR.hpp>
#include <DistanceCorr.hpp>
#include <MIAdaptive.hpp>
#include <Normalization.hpp>
#include <Transformation.hpp>

static Options option;
static void printUsage() {
	fprintf(stderr,
			"NetFusion [options]\n");
	fprintf(stderr, "Options:\n");
	fprintf(stderr, "\t-i <str> (input EXP formatted file [required]\n");
	fprintf(stderr,
			"\t-t <int> (number of CPU threads, default = %d [0 means auto])\n",
			option._numCPUThreads);
	fprintf(stderr, "\t-h (print out options)\n");
	fprintf(stderr, "\n");
}

static bool parseArgs(int argc, char* argv[]) {
	int opt;
	if (argc < 2) {
		printUsage();
		return false;
	}
	while ((opt = getopt(argc, argv, "i:t:h")) != -1) {
		switch (opt) {
		case 'i':
			option._input = optarg;
			break;
		case 't':
			option._numCPUThreads = atoi(optarg);
			break;
		case 'h':
			printUsage();
			return false;
		default:
			fprintf(stderr, "Unknown option: %s\n", optarg);
			return false;
		}
	}

	/*check the input files*/
	if(option._input.length() == 0){
		fprintf(stderr, "Input file name must be specified\n");
		return false;
	}

	fprintf(stderr, "Get number of vectors and vector size\n");
	if (EXPMatrixReader<double>::getMatrixSize(option._input,
			option._numVectors, option._vectorSize) == false) {
		return false;
	}

	return true;
}

int main(int argc, char* argv[]) {

	/*parse the arguments*/
	if (!parseArgs(argc, argv)) {
		return -1;
	}

	/*print out command line*/
	fprintf(stderr, "command line: ");
	for (int i = 0; i < argc; ++i) {
		fprintf(stderr, "%s ", argv[i]);
	}
	fprintf(stderr, "\n");

	/*statistics*/
	size_t numPairs = (size_t) (option._numVectors + 1) * option._numVectors / 2;	/*including self-vs-self*/
	fprintf(stderr, "Double precision: %d\n", option._useDouble ? 1 : 0);
	fprintf(stderr, "Vector size: %d\n", option._vectorSize);
	fprintf(stderr, "Number of vectors: %d\n", option._numVectors);
	fprintf(stderr, "Number of vector pairs: %ld\n", numPairs);
	fprintf(stderr, "Execution mode: %d\n", option._mode);

  PearsonRMKL<double> pr(option._numVectors, option._vectorSize,
        option._numCPUThreads, option._numMICThreads, option._micIndex,
        option._rank, option._numProcs);
  KendallTau<double> kendall(option._numVectors, option._vectorSize,
        option._numCPUThreads, option._numMICThreads, option._micIndex,
        option._rank, option._numProcs);
  SpearmanR<double> spearman(option._numVectors, option._vectorSize,
        option._numCPUThreads, option._numMICThreads, option._micIndex,
        option._rank, option._numProcs);
  DistanceCorr<double> distance(option._numVectors, option._vectorSize,
        option._numCPUThreads, option._numMICThreads, option._micIndex,
        option._rank, option._numProcs);
  MIAdaptive<double> mi(option._numVectors, option._vectorSize,
        option._numCPUThreads, option._numMICThreads, option._micIndex,
        option._rank, option._numProcs);

	/*load the matrix data*/
	EXPMatrixReader<double>::loadMatrixData(option._input, option._genes,
					option._samples, kendall.getVectors(), kendall.getNumVectors(),
					kendall.getVectorSize(), kendall.getVectorSizeAligned());

	EXPMatrixReader<double>::loadMatrixData(option._input, option._genes,
					option._samples, kendall.getVectors(), kendall.getNumVectors(),
					kendall.getVectorSize(), kendall.getVectorSizeAligned());

	EXPMatrixReader<double>::loadMatrixData(option._input, option._genes,
					option._samples, spearman.getVectors(), spearman.getNumVectors(),
					spearman.getVectorSize(), spearman.getVectorSizeAligned());

	EXPMatrixReader<double>::loadMatrixData(option._input, option._genes,
					option._samples, distance.getVectors(), distance.getNumVectors(),
					distance.getVectorSize(), distance.getVectorSizeAligned());

	EXPMatrixReader<double>::loadMatrixData(option._input, option._genes,
					option._samples, mi.getVectors(), mi.getNumVectors(),
					mi.getVectorSize(), mi.getVectorSizeAligned());


	/*run the kernel*/
	pr.runMultiThreaded();
	kendall.runMultiThreaded();
	spearman.runMultiThreaded();
	distance.runMultiThreaded();
	mi.runMultiThreaded();

	return 0;
}
