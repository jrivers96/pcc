#include <EXPMatrixReader.hpp>
#include <string>
#include <map>
#include <vector>
using namespace std;

int main(int argc, char* argv[])
{
	int numVectors;
	int vectorSize;
	double* matrix;

  vector<string> genes;
  vector<string> samples;

	string infile, outfile;
	if(argc < 3){
		fprintf(stderr, "convert2aracne inlist outfile\n");
		return -1;
	}

	infile = argv[1];
	outfile = argv[2];

	/*get matrix data*/
	if(EXPMatrixReader<double>::getMatrixSize(infile, numVectors, vectorSize) == false) {
		fprintf(stderr, "The input matrix file is invalid\n");
		return -1;
	}

	/*allocate data*/
	matrix = new double [numVectors * vectorSize];
	if(!matrix){
		fprintf(stderr, "memory allocation failed\n");
		return -1;
	}
	
	/*load data*/
	EXPMatrixReader<double>::loadMatrixData(infile, genes, samples, matrix, numVectors, vectorSize, vectorSize);

	/*write data to the file*/
	FILE* file = fopen(outfile.c_str(), "w");
	if(!file){
		fprintf(stderr, "Failed to open file %s\n", outfile.c_str());
		return -1;
	}

	/*write the data*/
	ssize_t i, j;
	fprintf(file, "Gene\tProbID\t");
	for(i = 0; i + 1 < samples.size(); ++i){
		fprintf(file, "%s\t", samples[i].c_str());
	}
	fprintf(file, "%s\n", samples[i].c_str());

	for(i = 0; i < genes.size(); ++i){
		fprintf(file, "%s\t%s\t", genes[i].c_str(), genes[i].c_str());
		for(j = 0; j < vectorSize - 1; ++j){
			fprintf(file, "%f\t", matrix[i * vectorSize + j]);
		}
		fprintf(file, "%f\n", matrix[i * vectorSize + j]);
	}
	
	fclose(file);

	return 0;
}
