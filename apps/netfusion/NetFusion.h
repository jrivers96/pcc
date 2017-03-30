#ifndef __NET_FUNSION_H
#define __NET_FUNSION_H
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <map>
#include <set>
#include<vector>
using namespace std;

#include <KendallTau.hpp>

struct Options {
	Options() {
		_useDouble = 0;
		_numVectors = 0; /*20k*/
		_vectorSize = 0; /*5k*/
		_numCPUThreads = 0; /*number of CPU threads*/
		_numMICThreads = 0; /*number of MIC threads*/
		_micIndex = 0; /*Xeon Phi index*/
		_numMics = 0;
		_enableMic = 0;
		_mode = -1; /*invalid mode*/
		_rank = 0;
		_numProcs = 1;
		_kendallVariant = KT_MERGE_SORT_TAU_B;
	}
	string _input;
	vector<string> _genes;
	vector<string> _samples;
	int _useDouble; /*use single precision*/
	int _numVectors; /*number of vectors*/
	int _vectorSize; /*vector size*/
	int _numCPUThreads; /*number of CPU threads*/
	int _numMICThreads; /*number of MIC threads*/
	int _micIndex; /*MIC index*/
	int _numMics; /*number of MICs*/
	int _enableMic; /*enable the use of MIC*/
	int _mode; /*execution mode*/
	int _rank; /*process rank*/
	int _numProcs; /*number of processes*/
	int _kendallVariant;	/*variant of Kendall rank correlation coefficient*/
};
#endif
