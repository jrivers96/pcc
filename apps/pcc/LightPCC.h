#ifndef __LIGHT_PCC_H
#define __LIGHT_PCC_H
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <map>
#include <set>
#include<vector>
using namespace std;

#include <KendallTau.hpp>

#ifdef WITH_MPI
#include <mpi.h>
#endif

/*
extern "C"
{
#include <Utilities.h>
}
*/

/*external functions*/
//extern int lightPearsonR(int argc, char* argv[]);
extern int lightPearsonRMKL(int argc, char* argv[]);
//extern int lightSpearmanR(int argc, char* argv[]);
//extern int lightKendallTau(int argc, char* argv[]);
//extern int lightDistanceCorr(int argc, char* argv[]);
//extern int lightMIAdaptive(int argc, char* argv[]);

#ifdef WITH_MPI
class RegistrationInfo
{
public:
	RegistrationInfo(int proc, int32_t num) {
		_procs.insert(proc);
		_numXeonPhis = num;
	}
	void add(int proc, int32_t numXeonPhis) {
		if(_procs.size() > 0 && _numXeonPhis != numXeonPhis) {
			fprintf(stderr, "Xeon Phi devices are inconsitent between processes\n");
			exit(-1);
		}
		if(_procs.size() == 0) {
			_numXeonPhis = numXeonPhis;
		}
		_procs.insert(proc);
	}
	bool isValid() {
		return _procs.size() <= _numXeonPhis;
	}
	set<int>& getProcs() {return _procs;}
	int getNumXeonPhis() {return _numXeonPhis;}
private:
	set<int> _procs;
	int _numXeonPhis;
};
#endif

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
		_transpose = 0;	/* do not transpose the input matrix by default*/
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
	int _transpose;	/*whether to transpose the input matrix*/
#ifdef WITH_MPI
	int _assignXeonPhis()
	{
		const int hostNameLength = 4095;
		const int maxBuffer = 4095;
		char hostName[4096];
		char buffer[4096];
		int numXeonPhis;
		int totalXeonPhis = 0;
		int numProcsSharingHost;

		/*check the number of Xeon Phis*/
#ifdef __INTEL_OFFLOAD
		_numMics = _Offload_number_of_devices();
#endif
		if (_numMics < 1) {
			fprintf(stderr, "Xeon Phi coprocessors are unavailable\n");
			return 0;
		}

		//get the host name
		if(gethostname(hostName, hostNameLength)) {
			fprintf(stderr, "Get host name failed");
		}

		if(_rank == 0) {
			map<string, RegistrationInfo*> hostMap;

			/*insert the Xeon Phi information on its own node*/
			string host((const char*)hostName);
			map<string, RegistrationInfo*>::iterator iter = hostMap.find(host);
			if(iter != hostMap.end()) {
				iter->second->add(_rank, _numMics);
			} else {
				RegistrationInfo* regInfo = new RegistrationInfo(_rank, _numMics);
				hostMap.insert(pair<string, RegistrationInfo*>(host, regInfo));
			}
			/*receive the Xeon Phi information from any other process*/
			for(int rank = 1; rank < _numProcs; ++rank) {
				MPI_Status status;
				MPI_Recv(buffer, maxBuffer, MPI_BYTE, rank, 0, MPI_COMM_WORLD, &status);

				/*forming a string*/
				int bufferLength;
				MPI_Get_count(&status, MPI_BYTE, &bufferLength);
				buffer[bufferLength] = '\0';

				/*get the hostname and number of devices*/
				sscanf(buffer, "%s %d", hostName, &numXeonPhis);

				/*insert the Xeon Phi information*/
				string host((const char*)hostName);
				map<string, RegistrationInfo*>::iterator iter = hostMap.find(host);
				if(iter != hostMap.end()) {
					iter->second->add(rank, numXeonPhis);
				} else {
					RegistrationInfo* regInfo = new RegistrationInfo(rank, numXeonPhis);
					hostMap.insert(pair<string, RegistrationInfo*>(host, regInfo));
				}
			}
			//check the validity of the each host, and assign Xeon Phis to each process
			for(map<string, RegistrationInfo*>::iterator iter = hostMap.begin(); iter != hostMap.end(); ++iter) {
				if(!iter->second->isValid()) {
					fprintf(stderr,"The number of processes exceed the number of Xeon Phis in the node %s\n", iter->first.c_str());
					fprintf(stderr,"Processes: %ld devices: %d\n", iter->second->getProcs().size(), iter->second->getNumXeonPhis());
					exit(-1);
				}
				int phiIndex = 0;
				string host = iter->first;
				set<int> procs = iter->second->getProcs();
				for(set<int>::iterator piter = procs.begin(); piter != procs.end(); ++piter) {
					if(*piter == 0) {
						//set the device ID
						_micIndex = phiIndex;
						numProcsSharingHost = procs.size();
					} else {
						//send the device ID
						MPI_Send(&phiIndex, 1, MPI_INTEGER, *piter, 0, MPI_COMM_WORLD);

						//send the number of procs in the host
						int procNum = procs.size();
						MPI_Send(&procNum, 1, MPI_INTEGER, *piter, 0, MPI_COMM_WORLD);
					}
					fprintf(stderr,"Process %d uses the Xeon Phi with ID %d in the host \"%s\"\n", *piter, phiIndex, host.c_str());
					++phiIndex;
				}
				totalXeonPhis += procs.size();
			}

			//release the host map
			for(map<string, RegistrationInfo*>::iterator iter = hostMap.begin(); iter != hostMap.end(); ) {
				delete iter->second;
				hostMap.erase(iter++);
			}
		} else {
			int phiIndex;
			//send the Xeon Phi information in this host
			sprintf(buffer, "%s %d", hostName, _numMics);
			MPI_Send(buffer, strlen(buffer), MPI_BYTE, 0, 0, MPI_COMM_WORLD);

			//receive the assigned device ID
			MPI_Status status;
			MPI_Recv(&phiIndex, 1, MPI_INTEGER, 0, 0, MPI_COMM_WORLD, &status);
			//set the GPU device number
			_micIndex = phiIndex;

			//receive the number of procs in the host
			MPI_Recv(&numProcsSharingHost, 1, MPI_INTEGER, 0, 0, MPI_COMM_WORLD, &status);
			totalXeonPhis = 1;
		}

		//broadcast the number of GPUs in the distribued system
		MPI_Bcast(&totalXeonPhis, 1, MPI_INTEGER, 0, MPI_COMM_WORLD);

		return totalXeonPhis;
	}
#endif
};
#endif
