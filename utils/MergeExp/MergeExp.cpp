#include <CustomFileReader.hpp>
#include <string>
#include <map>
#include <vector>
using namespace std;

template<typename FloatType>
bool loadMatrixData(char* infile, map<int, vector<FloatType> >& exps, int& accNumSamples, const bool reference)
{
	char* buffer = NULL, *tok;
	size_t bufferSize = 0;
	int numChars, numSamples;
	const char delim[] = "\t";
	CustomFileReader fileReader;
	typedef typename map<int, vector<FloatType> >::iterator iterator_type;

	/*open the file*/
	if (!fileReader.open(infile, "rb")) {
		fprintf(stderr, "Failed to open file %s\n", infile);
		return false;
	}

	/*skip the header*/
	numChars = fileReader.getline(&buffer, &bufferSize);
	if (numChars <= 0) {
		fprintf(stderr, "The file is incomplete\n");
		fileReader.close();
		return false;
	}

	/*get gene expression profiles*/
	numSamples = 0;
	while ((numChars = fileReader.getline(&buffer, &bufferSize)) != -1) {
		/*empty line*/
		if (numChars == 0) {
			continue;
		}
		/*get the gene ID*/
		tok = strtok(buffer, delim);
		if(tok == NULL){
			fprintf(stderr, "incomplete file at line %d\n", __LINE__);
			fileReader.close();
			return false;
		}
		int id = atoi(tok);
		iterator_type iter = exps.find(id);
		if(iter == exps.end() && reference == true){
			pair<iterator_type, bool> ret = exps.insert(make_pair(id, vector<FloatType>()));
			iter = ret.first;
		}

		/*extract gene expression values*/
		if(iter != exps.end()){
			int index = 0;
			for (tok = strtok(NULL, delim); tok != NULL;
				tok = strtok(NULL, delim)) {
				iter->second.push_back(atof(tok));
				++index;
			}
			if(numSamples == 0){
				numSamples = index;
			}
		}
	}
	/*close the file*/
	fileReader.close();

	accNumSamples += numSamples;
	fprintf(stderr, "number of samples: %d / %d\n", numSamples, accNumSamples);

	return true;
}

template<typename FloatType>
void mergeFiles(char* infile, char* outfile, const int top)
{
  char* buffer = NULL;
  size_t bufferSize = 0;
	int numFiles = 0, numSamples = 0, numChars;
  CustomFileReader fileReader;
  typedef typename map<int, vector<FloatType> >::iterator iterator_type;

  /*open the file*/
  if (!fileReader.open(infile, "rb")) {
    fprintf(stderr, "Failed to open file %s\n", infile);
		exit(-1);
  }

	map<int, vector<FloatType> > exps;
  while((numChars = fileReader.getline(&buffer, &bufferSize)) != -1){
		if(numChars == 0) continue;
		
		/*analyze the file*/
		fprintf(stderr, "ID (%d): %s\n", numFiles + 1, buffer);
		loadMatrixData<FloatType>(buffer, exps, numSamples, numFiles ? false: true);
		++numFiles;
		if(numFiles >= top) break;
	}

	int numGenes = 0;
	FILE* fout = fopen(outfile, "wb");
	if(!fout){
		fprintf(stderr, "failed to open output file %s\n", outfile);
		exit(-1);
	}
	/*write header*/
	fprintf(fout, "gene\tprob\t");
	for(int i = 0; i < numSamples - 1; ++i){
		fprintf(fout, "gsm%d\t", i);
	}
	if(numSamples > 0){
		fprintf(fout, "gsm%d\n", numSamples - 1);
	}
	fprintf(fout, "#\n#\n");
	for(iterator_type iter = exps.begin(); iter != exps.end(); ++iter){
		if((int)iter->second.size() != numSamples){
			fprintf(stderr, "this gene is missing in some dataset and must be discarded\n");
			continue;
		}
		++numGenes;

		/*write expression data*/
		fprintf(fout, "%d\t0\t", iter->first);
		for(int i = 0; i < numSamples - 1; ++i){
			fprintf(fout, "%f\t", iter->second[i]);
		}
		if(numSamples > 0){
			fprintf(fout, "%f\n", iter->second[numSamples - 1]);
		}
	}
	fclose(fout);
}
int main(int argc, char* argv[])
{
	int top = 0x7FFFFFFF;
	char* infile, *outfile;
	if(argc < 3){
		fprintf(stderr, "MergeExp infile_list outfile [top #files]\n");
		return -1;
	}

	infile = argv[1];
	outfile = argv[2];
	if(argc > 3){
		top = atoi(argv[3]);
	}
	
	mergeFiles<double>(infile, outfile, top);
	return 0;
}
