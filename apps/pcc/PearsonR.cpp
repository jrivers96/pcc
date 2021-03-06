/*
 * PearsonR.cpp
 *
 */

#include <PearsonR.hpp>
#include <EXPMatrixReader.hpp>
#include "PCC.h"

/*execution mode*/
#define SINGLE_THREADED		0
#define MULTI_THREADED		1
#ifdef WITH_PHI
#define XEON_PHI_ASYNC		2
#endif

#ifdef WITH_MPI
#define CPU_MPI						3
#ifdef WITH_PHI
#define XEON_PHI_MPI			4
#endif
#endif

static Options option;
static void printUsage() {
	fprintf(stderr,
			"LightPCC pearson [options] -m exe_mode\n");
	fprintf(stderr, "Options:\n");
	fprintf(stderr, "\t-i <str> (input EXP formatted file [random data if not given]\n");
	fprintf(stderr, "\t-d <int> (use double precision, default = %d)\n",
			option._useDouble);
	fprintf(stderr, "\t-n <int> (number of vectors, default = %d [random data])\n",
			option._numVectors);
	fprintf(stderr, "\t-l <int> (vector size, default = %d [random data])\n",
			option._vectorSize);
	fprintf(stderr,
			"\t-t <int> (number of CPU threads, default = %d [0 means auto])\n",
			option._numCPUThreads);
	fprintf(stderr,
			"\t-p <int> (number of Xeon Phi threads, default = %d [0 means auto])\n",
			option._numMICThreads);

	fprintf(stderr, "\t-m <int> (execution mode, default = %d [-1 invaid])\n",
			option._mode);
#ifndef WITH_MPI	/*without mpi*/
	fprintf(stderr, "\t    %d: singled-threaded on the CPU\n", SINGLE_THREADED);
	fprintf(stderr, "\t    %d: multi-threaded on the CPU\n", MULTI_THREADED);
#ifdef WITH_PHI
	fprintf(stderr, "\t    %d: single Xeon Phi\n", XEON_PHI_ASYNC);
#endif	/*WITH_PHI*/

#else	/*WITH_MPI*/
	fprintf(stderr, "\t    %d: MPI for CPU clusters\n", CPU_MPI);
#ifdef WITH_PHI
	fprintf(stderr, "\t    %d: MPI for Xeon Phi clusters\n", XEON_PHI_MPI);
#endif	/*WITH_PHI*/
#endif	/*WITH_MPI*/

#ifdef WITH_PHI
	fprintf(stderr, "\t-x <int> (Xeon Phi index [single Xeon Phi mode], default = %d)\n",
			option._micIndex);
#endif
	fprintf(stderr, "\t-h (print out options)\n");
	fprintf(stderr, "\n");
}

static bool parseArgs(int argc, char* argv[]) {
	int opt;
	if (argc < 2) {
		printUsage();
		return false;
	}
	while ((opt = getopt(argc, argv, "i:d:n:l:t:p:m:hx:")) != -1) {
		switch (opt) {
		case 'i':
			option._input = optarg;
			break;
		case 'd':
			option._useDouble = atoi(optarg);
			break;
		case 'n':
			option._numVectors = atoi(optarg);
			if (option._numVectors < 0) {
				option._numVectors = 0;
			}
			break;
		case 'l':
			option._vectorSize = atoi(optarg);
			if (option._vectorSize < 0) {
				option._vectorSize = 0;
			}
			break;
		case 't':
			option._numCPUThreads = atoi(optarg);
			break;
		case 'p':
			option._numMICThreads = atoi(optarg);
			break;
		case 'm':
			option._mode = atoi(optarg);
#ifdef WITH_PHI
			if (option._mode == XEON_PHI_ASYNC) {
				/*enable the use of Xeon Phi*/
				option._enableMic = 1;
			}
#endif

#if	defined(WITH_MPI) && defined(WITH_PHI)
			/*enable the use of MPI*/
			if (option._mode == XEON_PHI_MPI) {
				option._enableMic = 1;
			}
#endif
			break;
		case 'x':
			option._micIndex = atoi(optarg);
			if (option._micIndex < 0) {
				option._micIndex = 0;
			}
			break;
		case 'h':
			printUsage();
			return false;
		default:
			fprintf(stderr, "Unknown option: %s\n", optarg);
			return false;
		}
	}
	if (option._mode < 0) {
		fprintf(stderr, "Must specify the execution mode using paramter: -m\n");
		return false;
	}

	if (option._input.length() == 0) {
		if (option._numVectors == 0) {
			fprintf(stderr,
					"Must specifiy the number of vectors using paramter: -n\n");
			return false;
		}
		if (option._vectorSize == 0) {
			fprintf(stderr,
					"Must specify the vector size using paramter: -l\n");
			return false;
		}
	} else {
		/*read the file*/
		fprintf(stderr, "Get number of vectors and vector size\n");
		if (!option._useDouble) {
			if (EXPMatrixReader<float>::getMatrixSize(option._input,
					option._numVectors, option._vectorSize) == false) {
				return false;
			}
		} else {
			if (EXPMatrixReader<double>::getMatrixSize(option._input,
					option._numVectors, option._vectorSize) == false) {
				return false;
			}
		}
	}

	return true;
}

int lightPearsonR(int argc, char* argv[]) {

	/*parse the arguments*/
	if (!parseArgs(argc, argv)) {
		return -1;
	}

#ifdef WITH_MPI
	MPI_Comm_rank(MPI_COMM_WORLD, &option._rank);
	MPI_Comm_size(MPI_COMM_WORLD, &option._numProcs);

	if(option._enableMic) {
		option._assignXeonPhis();
	}
#endif /*WITH_MPI*/

	/*print out command line*/
	if (option._rank == 0) {
		fprintf(stderr, "command line: ");
		for (int i = 0; i < argc; ++i) {
			fprintf(stderr, "%s ", argv[i]);
		}
		fprintf(stderr, "\n");

		/*statistics*/
		size_t numPairs = (size_t) (option._numVectors + 1) * option._numVectors
				/ 2;	/*including self-vs-self*/
		fprintf(stderr, "Double precision: %d\n", option._useDouble ? 1 : 0);
		fprintf(stderr, "Vector size: %d\n", option._vectorSize);
		fprintf(stderr, "Number of vectors: %d\n", option._numVectors);
		fprintf(stderr, "Number of vector pairs: %ld\n", numPairs);
		fprintf(stderr, "Execution mode: %d\n", option._mode);
#ifdef WITH_PHI_ASSEMBLY_FLOAT
		fprintf(stderr, "Xeon Phi with assemblies for single precision\n");
#endif
#ifdef WITH_PHI_ASSEMBLY_DOUBLE
		fprintf(stderr, "Xeon Phi with assemblies for double precision\n");
#endif
	}
	/*create object and simulate data if applicable*/
	if (!option._useDouble) {
		PearsonR<float> pr(option._numVectors, option._vectorSize,
				option._numCPUThreads, option._numMICThreads, option._micIndex,
				option._rank, option._numProcs);

		if (option._input.length()) {
			EXPMatrixReader<float>::loadMatrixData(option._input, option._genes,
					option._samples, pr.getVectors(), pr.getNumVectors(),
					pr.getVectorSize(), pr.getVectorSizeAligned());
		} else {
			pr.generateRandomData();
		}

		/*run the kernel*/
		switch (option._mode) {
#ifndef WITH_MPI
		case SINGLE_THREADED:
			pr.runSingleThreaded();
			break;
		case MULTI_THREADED:
			pr.runMultiThreaded();
			break;
#ifdef WITH_PHI
			case XEON_PHI_ASYNC:
			pr.runSingleXeonPhi();
			break;
#endif

#else
			case CPU_MPI:
			pr.runMPICPU();
			break;
#ifdef WITH_PHI
			case XEON_PHI_MPI:
			pr.runMPIXeonPhi();
			break;
#endif
#endif
		default:
			fprintf(stderr, "Not supported mode: %d\n", option._mode);
			return -1;
		}
	} else {
		PearsonR<double> pr(option._numVectors, option._vectorSize,
				option._numCPUThreads, option._numMICThreads, option._micIndex,
				option._rank, option._numProcs);
		if (option._input.length()) {
			EXPMatrixReader<double>::loadMatrixData(option._input, option._genes,
					option._samples, pr.getVectors(), pr.getNumVectors(),
					pr.getVectorSize(), pr.getVectorSizeAligned());
		} else {
			pr.generateRandomData();
		}
#if 0
		FILE* fp = fopen("matlab.txt", "w");
		for(int row = 0; row < pr.getVectorSize(); ++row){
			double* data;
			for(int col = 0; col < pr.getNumVectors() - 1; ++col){
					data = pr.getVectors() + col * pr.getVectorSizeAligned() + row;
					fprintf(fp, "%f ", *data);
			}
			data = pr.getVectors() + (pr.getNumVectors() - 1) * pr.getVectorSizeAligned() + row;
			fprintf(fp, "%f\n", *data);
		}
		fclose(fp);
#endif

		/*run the kernel*/
		switch (option._mode) {
#ifndef WITH_MPI
		case SINGLE_THREADED:
			pr.runSingleThreaded();
			break;
		case MULTI_THREADED:
			pr.runMultiThreaded();
			break;
#ifdef WITH_PHI
			case XEON_PHI_ASYNC:
			pr.runSingleXeonPhi();
			break;
#endif

#else
			case CPU_MPI:
			pr.runMPICPU();
			break;
#ifdef WITH_PHI
			case XEON_PHI_MPI:
			pr.runMPIXeonPhi();
			break;
#endif
#endif
		default:
			fprintf(stderr, "Not supported mode: %d\n", option._mode);
			return -1;
		}

	}
	return 0;
}
