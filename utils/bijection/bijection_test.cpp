#include<stdio.h>
#include <math.h>

static ssize_t compCoordinate(int matrixDimSize, int row, int col)
{
	ssize_t globalIndex = 0;
	double p, q;
	p = static_cast<double>(matrixDimSize);

	globalIndex = (ssize_t)((p - 0.5 * (row - 1) ) * row + col - row);
	if(globalIndex < 0){
		printf("wrong computation: %ld\n", globalIndex);
	}

	return globalIndex;
}
static void getCoordinate(ssize_t globalIndex, ssize_t matrixDimSize,
    int& row, int& col) {
  double p, q;
  p = static_cast<double>(matrixDimSize);
  q = static_cast<double>(globalIndex);

	row = (int) ceil(p - 0.5 - sqrt(p * p + p - 2 * (q + 1) + 0.25));
	col = row + (globalIndex - (2 * matrixDimSize - (row - 1)) * row / 2);

}
int main(int argc, char* argv[])
{
	ssize_t jobId;
	int numVectors = 128000;
	ssize_t numPairs = (numVectors + 1) * numVectors / 2;

	jobId = 0;
	for(int row = 0; row < numVectors; ++row){
		for(int col = row; col < numVectors; ++col){
			int r, c;
			ssize_t globalIndex = compCoordinate(numVectors, row, col);
			if(globalIndex != jobId){
				printf("wrong: job id is not correct %ld != %ld\n", globalIndex, jobId);
				break;
			}

			/*reversely compute the coordinate*/
			getCoordinate(globalIndex, numVectors, r, c);
			if(r != row || c != col){
				printf("wrong: globalIndex: %ld r %d c %d row %d col %d\n", globalIndex, r, c, row, col);
				break;
			}
			/*increase the job identifier*/
			++jobId;
		}
	}
	return 0;
}
