/*
 * GEMatrixReader.hpp
 *
 *  Created on: Mar 25, 2016
 *  Author: Liu, Yongchao
 *  Affiliation: School of Computational Science & Engineering
 *  						Georgia Institute of Technology, Atlanta, GA 30332
 *  URL: www.liuyc.org
 */

#ifndef INCLUDE_GEMATRIXREADER_HPP_
#define INCLUDE_GEMATRIXREADER_HPP_
#include <CustomFileReader.hpp>
#include <string>
using namespace std;

template<typename FloatType>
class GEMatrixReader {
public:
	/*get gene expression matrix size*/
	static bool getMatrixSize(string& fileName, int& numVectors, int& vectorSize);

	/*get the matrix data*/
	static bool loadMatrixData(string& fileName, vector<string>& genes,
			vector<string>& samples, FloatType* vectors, const int numVectors,
			const int vectorSize, const int vectorSizeAligned);
};

template<typename FloatType>
bool GEMatrixReader<FloatType>::getMatrixSize(string& fileName, int& numVectors,
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


	/*analyze the header*/
	for (tok = strtok(buffer, delim); tok != NULL; tok = strtok(NULL, delim)) {
		vectorSize++;
	}
	--vectorSize; /*delete the first header*/

	/*get the data*/
	while ((numChars = fileReader.getline(&buffer, &bufferSize)) != -1) {
		/*empty line*/
		if (numChars == 0) {
			continue;
		}
		++numVectors;
	}

	/*close the file*/
	fileReader.close();

	return true;
}

template<typename FloatType>
bool GEMatrixReader<FloatType>::loadMatrixData(string& fileName,
		vector<string>& genes, vector<string>& samples, FloatType* vectors,
		const int numVectors, const int vectorSize,
		const int vectorSizeAligned) {
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
	firstEntry = true;
	for (tok = strtok(buffer, delim); tok != NULL; tok = strtok(NULL, delim)) {
		if (!firstEntry) {
			/*save the sample name*/
			samples.push_back(string(tok));
			numSamples++;
		} else {
			firstEntry = false;
		}
	}

	/*check consistency*/
	if (numSamples != vectorSize) {
		fprintf(stderr,
				"The number of samples (%d) not equal to vector size (%d)\n",
				numSamples, vectorSize);
		fileReader.close();
		return false;
	}

	/*get the data*/
	numGenes = 0;
	while ((numChars = fileReader.getline(&buffer, &bufferSize)) != -1) {
		/*empty line*/
		if (numChars == 0) {
			continue;
		}
		if (numGenes >= numVectors) {
			fprintf(stderr,
					"Eorror: number of genes is inconsistent with number of vectors\n");

			fileReader.close();
			return false;
		}

		/*extract the data*/
		firstEntry = true;
		index = 0;
		for (tok = strtok(buffer, delim); tok != NULL;
				tok = strtok(NULL, delim)) {
			if (!firstEntry) {
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
			} else {
				/*save the gene name*/
				firstEntry = false;
				genes.push_back(string(tok));
			}
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

#endif /* INCLUDE_GEMATRIXREADER_HPP_ */
