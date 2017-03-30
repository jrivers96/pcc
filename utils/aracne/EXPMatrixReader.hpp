/*
 * EXPMatrixReader.hpp
 *
 *  Created on: Mar 25, 2016
 *      Author: yongchao
 */

#ifndef INCLUDE_EXPMATRIXREADER_HPP_
#define INCLUDE_EXPMATRIXREADER_HPP_
#include <CustomFileReader.hpp>
#include <string>
#include <vector>
using namespace std;

template<typename FloatType>
class EXPMatrixReader {
public:
	/*get gene expression matrix size*/
	static bool getMatrixSize(string& fileName, int& numVectors, int& vectorSize, const bool skip = false);

	/*get the matrix data*/
	static bool loadMatrixData(string& fileName, vector<string>& genes,
			vector<string>& samples, FloatType* vectors, const int numVectors,
			const int vectorSize, const int vectorSizeAligned, const bool skip = false);
};

template<typename FloatType>
bool EXPMatrixReader<FloatType>::getMatrixSize(string& fileName, int& numVectors,
		int& vectorSize, const bool skip) {
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
	vectorSize -= 2; /*exclude the first two columns of the header: prob id and locus id*/
	fprintf(stderr, "Number of samples: %d\n", vectorSize);

	if(skip){
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

template<typename FloatType>
bool EXPMatrixReader<FloatType>::loadMatrixData(string& fileName,
		vector<string>& genes, vector<string>& samples, FloatType* vectors,
		const int numVectors, const int vectorSize,
		const int vectorSizeAligned, const bool skip) {
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

	/*analyze the header and skip the first two values*/
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

	/*save sample names*/
	for (tok = strtok(NULL, delim); tok != NULL; tok = strtok(NULL, delim)) {
		samples.push_back(string(tok));
		numSamples++;
	}

	/*check consistency*/
	if (numSamples != vectorSize) {
		fprintf(stderr,
				"The number of samples (%d) not equal to vector size (%d)\n",
				numSamples, vectorSize);
		fileReader.close();
		return false;
	}

	if(skip){
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

		/*skip the first column*/
		tok = strtok(buffer, delim);
		if(tok == NULL){
			fprintf(stderr, "incomplete file at line %d\n", __LINE__);
			fileReader.close();
			return false;
		}
		/*save the locus id*/
		genes.push_back(string(tok));

#if 0
		/*from SEEK data, do not need to skip the second value*/
		tok = strtok(NULL, delim);
    if(tok == NULL){
      fprintf(stderr, "incomplete file at line %d\n", __LINE__);
      fileReader.close();
      return false;
    }
#endif

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
			*(vectors + numGenes * vectorSizeAligned + index) = atof(tok);

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

#endif /* INCLUDE_EXPMATRIXREADER_HPP_ */
