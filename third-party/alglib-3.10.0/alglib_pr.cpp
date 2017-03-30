#include <stdio.h>
#include <time.h>
#include "ap.h"
#include "linalg.h"
#include "statistics.h"
#include <omp.h>
#include <vector>
#include <list>
#include <stdlib.h>
#include <CustomFileReader.hpp>
using namespace alglib;
using namespace alglib_impl;

bool getMatrixSize(string& fileName, int& numVectors,
		int& vectorSize) {
	char* buffer = NULL, *tok;
	size_t bufferSize = 0;
	int numChars;
	const char delim[] = "\t";
	CustomFileReader fileReader;

	/*open the file*/
	if (!fileReader.open(fileName.c_str(), "rb")) {
		fprintf(stderr, "Failed to open file %s\n", fileName.c_str());
		return false;
	}

	numVectors = vectorSize = 0;
	/*read the header to get the number of samples*/
	numChars = fileReader.getline(&buffer, &bufferSize);
	if (numChars <= 0) {
		fprintf(stderr, "The file is incomplete\n");
		fileReader.close();
		return false;
	}

	/*analyze the header on the first row*/
	for (tok = strtok(buffer, delim); tok != NULL; tok = strtok(NULL, delim)) {
		vectorSize++;
	}
	vectorSize -= 2; /*exclude the first columns of the header: prob id and locus id*/
	fprintf(stderr, "Number of samples: %d\n", vectorSize);

	/*skip the second and the third rows*/
	if(fileReader.getline(&buffer, &bufferSize) <= 0){
		fprintf(stderr, "EXP file is incomplete at the second row\n");
		fileReader.close();
		return false;
	}
	if(fileReader.getline(&buffer, &bufferSize) <= 0){
		fprintf(stderr, "EXP file is incomplete at the third row\n");
		fileReader.close();
		return false;
	}

	/*get gene expression profiles*/
	while ((numChars = fileReader.getline(&buffer, &bufferSize)) != -1) {
		/*empty line*/
		if (numChars == 0) {
			continue;
		}
		++numVectors;
	}
	fprintf(stderr, "Number of gene expression profiles: %d\n", numVectors);

	/*close the file*/
	fileReader.close();

	return true;
}

bool loadMatrixData(string& fileName, const int numVectors, const int vectorSize, real_2d_array& vectors)
{
	char* buffer = NULL, *tok;
	size_t bufferSize = 0;
	int numChars, index;
	bool firstEntry;
	const char delim[] = "\t";
	CustomFileReader fileReader;

	/*open the file*/
	if (!fileReader.open(fileName.c_str(), "rb")) {
		fprintf(stderr, "Failed to open file %s\n", fileName.c_str());
		return false;
	}

	int numGenes = 0;
	int numSamples = 0;
	/*read the header to get the number of samples*/
	numChars = fileReader.getline(&buffer, &bufferSize);
	if (numChars <= 0) {
		fprintf(stderr, "The file is incomplete\n");
		fileReader.close();
		return false;
	}

	/*analyze the header*/
	tok = strtok(buffer, delim);
	if(tok == NULL){
		fprintf(stderr, "Incomplete header at line %d\n", __LINE__);
		fileReader.close();
		return false;
	}
	tok = strtok(NULL, delim);
	if(tok == NULL){
    fprintf(stderr, "Incomplete header at line %d\n", __LINE__);
    fileReader.close();
    return false;
	}

	/*skip the second and third rows*/
	if(fileReader.getline(&buffer, &bufferSize) <= 0){
		fprintf(stderr, "Incomplete file at line %d\n", __LINE__);
		fileReader.close();
		return false;
	}
	if(fileReader.getline(&buffer, &bufferSize) <= 0){
		fprintf(stderr, "Incomplete file at line %d\n", __LINE__);
		fileReader.close();
		return false;
	}

	/*get gene expression profiles*/
	numGenes = 0;
	while ((numChars = fileReader.getline(&buffer, &bufferSize)) != -1) {
		/*empty line*/
		if (numChars == 0) {
			continue;
		}
		/*consistency check*/
		if (numGenes >= numVectors) {
			fprintf(stderr,
					"Eorror: number of genes (%d) is not equal to (%d)\n", numGenes, numVectors);
			fileReader.close();
			return false;
		}

		/*skip the first two columns*/
		tok = strtok(buffer, delim);
		if(tok == NULL){
			fprintf(stderr, "incomplete file at line %d\n", __LINE__);
			fileReader.close();
			return false;
		}
		tok = strtok(NULL, delim);
    if(tok == NULL){
      fprintf(stderr, "incomplete file at line %d\n", __LINE__);
      fileReader.close();
      return false;
    }

		/*extract gene expression values*/
		index = 0;
		for (tok = strtok(NULL, delim); tok != NULL;
				tok = strtok(NULL, delim)) {
			if (index >= vectorSize) {
				fprintf(stderr,
						"Error: number of gene expression values is inconsistent with others\n");
				fileReader.close();
				return false;
			}
	
			/*save the value*/
			vectors[index][numGenes] = atof(tok);

			/*increase the index*/
			++index;
		}

		/*increase the gene index*/
		++numGenes;
	}
	if (numGenes != numVectors) {
		fprintf(stderr,
				"Eorror: number of genes (%d) is inconsistent with number of vectors (%d)\n", numGenes, numVectors);
		fileReader.close();
		return false;
	}

	/*close the file*/
	fileReader.close();

	return true;
}


#include <unistd.h>
#include <sys/time.h>

static double getSysTime() {
    double dtime;
    struct timeval tv;

    gettimeofday(&tv, NULL);

    dtime = (double) tv.tv_sec;
    dtime += (double) (tv.tv_usec) / 1000000.0;

    return dtime;
  }

int main(int argc, char* argv[])
{
	int n, m;
	string infile;
	if(argc < 2){
		printf("parameters: vector_size num_vectors\n");
		return 0;
	}
	if(argc == 3){
		n = atoi(argv[1]);
		m = atoi(argv[2]);
	}else if(argc == 2){
		    /*read the file*/
    fprintf(stderr, "Get number of vectors and vector size\n");
		infile = argv[1];
    if (getMatrixSize(infile, m, n) == false) {
			return -1;
    }
	}
	printf("number of vectors: %d\n", m);
	printf("vector size: %d\n", n);

	real_2d_array x, y;
	x.setlength(n, m);
	y.setlength(m, m);

	if(argc == 3){
		/*generate random data*/
		srand48(11);
		for(int i = 0; i < m; ++i){
			for(int j = 0; j < n; ++j){
				x[j][i] = drand48();
				//printf("%f ", x[j][i]);
			}
			//printf("\n");
		}
	}else if(argc == 2){
		infile = argv[1];
		loadMatrixData(infile, m, n, x);
	}

  /*mean of each vector*/
	printf("start computing\n");
	double stime = getSysTime();
	pearsoncorrm(x, y);
	double etime = getSysTime();
	printf("Overall time: %f seconds\n", etime - stime);
#if 0
	for(int i = 0; i < m; ++i){
		for(int j = 0; j < m; ++j){
			printf("%f ", y[i][j]);
		}
		printf("\n");
	}
#endif
	return -1;
}
