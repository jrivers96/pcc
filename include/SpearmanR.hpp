/*
 * SpearmanR.hpp
 *
 *  Created on: Mar 22, 2016
 *  Author: Liu, Yongchao
 *  Affiliation: School of Computational Science & Engineering
 *  						Georgia Institute of Technology, Atlanta, GA 30332
 *  URL: www.liuyc.org
 */

#ifndef SPEARMANR_HPP_
#define SPEARMANR_HPP_
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/time.h>
#include <omp.h>
#include <vector>
#include <list>
#include <typeinfo>
using namespace std;
#pragma once

#include <DeviceUtils.hpp>

#ifdef WITH_MPI
#include <mpi.h>
#endif	/*with mpi*/

#ifdef WITH_PHI
#include <immintrin.h>
/*for device memroy allocation*/
#define SR_MIC_REUSE alloc_if(0) free_if(0)
#define SR_MIC_ALLOC alloc_if(1) free_if(0)
#define SR_MIC_FREE  length(0) alloc_if(0) free_if(1)
#define SR_MIC_ALLOC_FREE alloc_if(1) free_if(1)
#endif	/*with phi*/

/*tile size*/
#define SR_PHI_TILE_DIM	4		/*must be divided by 236*/
#define SR_PHI_TILE_SIZE	(SR_PHI_TILE_DIM * SR_PHI_TILE_DIM)
#define SR_MT_TILE_DIM		8
#define SR_MPI_TILE_DIM	8
#define SR_MPI_TILE_SIZE (SR_MPI_TILE_DIM * SR_MPI_TILE_DIM)

/*maximum dimension size*/
#ifndef SR_MAX_MAT_DIM
#define SR_MAX_MAT_DIM			(1<<25)
#endif

/*maximum Xeon Phi buffer size*/
#define SR_PHI_BUFFER_SIZE (1 << 29)

/*software barrier for hardware threads per core*/
//#define SOFT_BARRIER	1	

/*template class*/
template<typename FloatType>
class SpearmanR {
public:
	SpearmanR(int numVectors, int vectorSize, int numCPUThreads,
			int numMICThreads, int micIndex, int rank, int numProcs);
	~SpearmanR();

	inline FloatType* getVectors() {
		return _vectors;
	}
	inline int getNumVectors() {
		return _numVectors;
	}
	inline int getVectorSize() {
		return _vectorSize;
	}
	inline int getVectorSizeAligned() {
		return _vectorSizeAligned;
	}

	/*generate random data*/
	void generateRandomData(const int seed = 11);

	/*single threaded implementation*/
	void runSingleThreaded();

	/*multiple threads with optimized*/
	void runMultiThreaded();

#ifdef WITH_PHI
	/*exon phi*/
	void runSingleXeonPhi();
#endif	/*with phi*/

	/*MPI*/
#ifdef WITH_MPI
	void runMPICPU();
#ifdef WITH_PHI
	void runMPIXeonPhi();
#endif	/*with phi*/
#endif	/*with mpi*/

	/*transpose the matrix*/
	void transpose();
private:
	FloatType* _vectors; /*vector data. Stored consecutively and each vector contains an aligned number of elements*/
	int _numVectors; /*number of vectors in the data*/
	int _vectorSize; /*effective vector size, i.e. the real number of elements per vector*/
	int _vectorSizeAligned; /*aligned to vector size to 16 so that the address is aligned to 64 byte boundary*/
	//int _maxPartitionSize; /*maximum partition size for asynchronous mode*/
	int _numCPUThreads; /*the number of CPU threads*/
	int _numMICThreads; /*the number of MIC threads*/
	int _micIndex;	/*index of MIC used by this processs*/
	int _rank;			/*process rank*/
	int _numProcs;	/*number of MPI processes*/
	int _tiledPrMatrix; /*how the pearson correlation matrix is stored*/
	FloatType* _spearmanCorr; /*pearson correlation matrix*/

	/*used for software barrier*/
	int* _barriers;
	int* _counts;

	/*for Xeon Phi*/
#ifdef WITH_PHI
	__attribute__((target(mic)))
#endif
	void _softBarrierInit(const int numSlots, const int numThreads)
	{
		_barriers = (int*)mm_malloc(numSlots * sizeof(int), 64);
		_counts = (int*)mm_malloc(numSlots * sizeof(int), 64);
		for(int i = 0; i < numSlots; ++i){
			_counts[i] = numThreads;
			_barriers[i] = 0;
		}
	}
#ifdef WITH_PHI
  __attribute__((target(mic)))
#endif
	void _softBarrierFinalize()
	{
		mm_free(_barriers);
		mm_free(_counts);
	}
#ifdef WITH_PHI
  __attribute__((target(mic)))
#endif
	void _softBarrierSync(const int barrierIndex, const int numThreads, int& localSense)
	{
		localSense = !localSense;	/*inverse local variable*/
		if(__sync_fetch_and_sub(_counts + barrierIndex, 1) == 1){
			__sync_fetch_and_add(_counts + barrierIndex, numThreads);
			__sync_bool_compare_and_swap(_barriers + barrierIndex, !localSense, localSense);
			//_barriers[barrierIndex] = localSense;
			//_counts[barrierIndex] = numThreads;
		}else{
			//while(_barriers[barrierIndex] != localSense);
			while(__sync_fetch_and_sub(_barriers + barrierIndex, 0) != localSense);
		}
	}

#ifdef WITH_PHI
  __attribute__((target(mic)))
#endif
	void _runSingleXeonPhiCore(
			FloatType* __restrict__ vectors, FloatType* __restrict__ prValues, const ssize_t indexRangeStart, const ssize_t indexRangeClose);

	/*static function members*/
public:
	/*when matrix dimension size is large than 2^21, we should use long double instead of double*/
#ifdef WITH_PHI
	__attribute__((target(mic)))
#endif
	inline void getCoordinate(const ssize_t globalIndex,
			const ssize_t matrixDimSize, int& row, int& col) {
#if SR_MAX_MAT_DIM < (1 << 25)
		typedef double mydouble;
#else
		typedef long double mydouble;
#endif
		mydouble p, q;
		p = static_cast<mydouble>(matrixDimSize - 1);
		q = static_cast<mydouble>(globalIndex);
		row = (int) ceil(p - 0.5 - sqrt(p * p + p - 2 * (q + 1) + 0.25));
		col = row
				+ (globalIndex - (2 * (matrixDimSize - 1) - row + 1) * row / 2)
				+ 1;
	}

	/*conditions: row < col*/
#ifdef WITH_PHI
  __attribute__((target(mic)))
#endif
	inline ssize_t getGlobalIndex(const ssize_t matrixDimSize, const int row,
			const int col) {
#if SR_MAX_MAT_DIM < (1 << 25)
		typedef double mydouble;
#else
		typedef long double mydouble;
#endif

		mydouble p, q;
		p = static_cast<mydouble>(matrixDimSize);

		return (ssize_t) ((p - 1 - 0.5 * (row - 1)) * row + col - (row + 1));
	}

#ifdef WITH_PHI
	__attribute__((target(mic)))
#endif
	inline void getTileCoordinate(const ssize_t globalIndex,
			const ssize_t tileDim, int& row, int& col) {
#if SR_MAX_MAT_DIM < (1 << 25)
		typedef double mydouble;
#else
		typedef long double mydouble;
#endif
		mydouble p, q;
		p = static_cast<mydouble>(tileDim);
		q = static_cast<mydouble>(globalIndex);
		row = (int) ceil(p - 0.5 - sqrt(p * p + p - 2 * (q + 1) + 0.25));
		col = row + (globalIndex - (2 * tileDim - row + 1) * row / 2);
	}

	/*conditions: row <= col*/
#ifdef WITH_PHI
  __attribute__((target(mic)))
#endif
	inline ssize_t getTileGlobalIndex(const ssize_t matrixDimSize, const int row,
		const int col) {
#if SR_MAX_MAT_DIM < (1 << 25)
    typedef double mydouble;
#else
    typedef long double mydouble;
#endif

    mydouble p, q;
    p = static_cast<mydouble>(matrixDimSize);

    return (ssize_t) ((p - 0.5 * (row - 1)) * row + col - row);
	}
#ifdef WITH_PHI
__attribute__((target(mic)))
#endif
	inline double getSysTime() {
		double dtime;
		struct timeval tv;

		gettimeofday(&tv, NULL);

		dtime = (double) tv.tv_sec;
		dtime += (double) (tv.tv_usec) / 1000000.0;

		return dtime;
	}
#ifdef WITH_PHI
__attribute__((target(mic)))
#endif
	inline void* mm_malloc(ssize_t size, ssize_t alignment)
	{
		return _mm_malloc(size, alignment);
	}
#ifdef WITH_PHI
__attribute__((target(mic)))
#endif
	inline void mm_free(void* buffer)
	{
		_mm_free(buffer);
	}

};

template<typename FloatType>
SpearmanR<FloatType>::SpearmanR(int numVectors, int vectorSize, int numCPUThreads,
		int numMICThreads, int micIndex, int rank, int numProcs) {
	int alignment = 64 / sizeof(FloatType);	/*align to 64 byte boundary*/
	_numVectors = numVectors;
	_vectorSize = vectorSize;
	_vectorSizeAligned = (_vectorSize + alignment - 1) / alignment * alignment;
	_numCPUThreads = numCPUThreads;
	_numMICThreads = numMICThreads;
	_micIndex = micIndex; /*index of mic device*/
	_rank = rank;
	_numProcs = numProcs;

	/*check the number of vectors*/
	if (_numVectors > SR_MAX_MAT_DIM) {
		fprintf(stderr,
				"The number of vectors exceeds the maximum limit (%ld)\n",
				(ssize_t) SR_MAX_MAT_DIM);
		exit(-1);
	}

	/*allocate space*/
	_spearmanCorr = NULL;

	/*align each vector*/
	_vectors = (FloatType*) mm_malloc(
			(ssize_t)_numVectors * _vectorSizeAligned * sizeof(FloatType), 64);
	if (!_vectors) {
		fprintf(stderr, "Memory allocation failed at line %d in file %s\n",
		__LINE__, __FILE__);
		exit(-1);
	}
}

template<typename FloatType>
SpearmanR<FloatType>::~SpearmanR() {
	if (_vectors) {
		mm_free(_vectors);
	}
	if (_spearmanCorr) {
		mm_free(_spearmanCorr);
	}
}

template<typename FloatType>
void SpearmanR<FloatType>::generateRandomData(const int seed) {
	srand48(11);
	for (int i = 0; i < _numVectors; ++i) {
		FloatType* __restrict__ dst = _vectors + i * _vectorSizeAligned;
		for (int j = 0; j < _vectorSize; ++j) {
			dst[j] = drand48();
		}
	}
}
template<typename FloatType>
void SpearmanR<FloatType>::runSingleThreaded() {
	double stime, etime;
#ifdef VERBOSE
	fprintf(stderr, "execute function %s\n", __FUNCTION__);
#endif

	/*output matrix*/
	_spearmanCorr = (FloatType*) mm_malloc(
			(ssize_t) _numVectors * _numVectors * sizeof(FloatType), 64);
	if (!_spearmanCorr) {
		fprintf(stderr, "Memory allocation failed\n");
		exit(-1);
	}
	_tiledPrMatrix = 0;

	/*record the system time*/
	stime = getSysTime();

	/*allocate vectors for mean and variance*/
	double t1 = getSysTime();
	int rank, j;
	FloatType x, meanX, varX, prod;
	FloatType* __restrict__ vecX;
	FloatType* __restrict__ vecY;
	const FloatType avg = 1.0 / (FloatType)_vectorSize;

	vecX = _vectors;
	CustomPair<FloatType, int>* ranks = (CustomPair<FloatType, int>*)mm_malloc(_vectorSize * sizeof(CustomPair<FloatType, int>), 64);
	for (int i = 0; i < _numVectors; ++i, vecX += _vectorSizeAligned) {

		/*get the rank vector*/
		for(j = 0; j < _vectorSize; ++j){
			ranks[j]._first = vecX[j];
			ranks[j]._second = j;
		}
		qsort(ranks, _vectorSize, sizeof(CustomPair<FloatType, int>), CustomPair<FloatType, int>::ascendFirst);
		for(j = 0, rank = 1; j < _vectorSize - 1; ++j){
			vecX[ranks[j]._second] = rank;
			if(ranks[j]._first != ranks[j + 1]._first){
				rank++;
			}
		}
		vecX[ranks[j]._second] = rank;

		/*do we need to normalize the ranks to avoid overflow? FIXME*/
    for(j = 0; j < _vectorSize; ++j){
      vecX[j] /= rank;
    }
		
		/*get the vector data*/
		meanX = 0;
#pragma vector aligned
#pragma simd reduction(+:meanX)
		for (j = 0; j < _vectorSize; ++j) {
			meanX += vecX[j] * avg;
		}

		/*compute the variance*/
		varX = 0;
#pragma vector aligned
#pragma simd reduction(+:varX)
		for (j = 0; j < _vectorSize; ++j) {
			x = vecX[j] - meanX;
			varX += x * x;
		}
		varX = 1 / sqrt(varX);

		/*normalize the data*/
#pragma vector aligned
#pragma simd
		for (j = 0; j < _vectorSize; ++j) {
			x = vecX[j] - meanX;
			vecX[j] = x * varX;
		}
	}
	mm_free(ranks);
	double t2 = getSysTime();
	fprintf(stderr, "time for ranking and transformation: %f seconds\n", t2 - t1);

	/*compute pairwise correlation coefficient*/
	vecX = _vectors;
	for (int row = 0; row < _numVectors; ++row, vecX += _vectorSizeAligned) {
		vecY = _vectors + row * _vectorSizeAligned;
		for (int col = row; col < _numVectors; ++col, vecY +=
				_vectorSizeAligned) {
			prod = 0;
#pragma vector aligned
#pragma simd reduction(+:prod)
			for (int j = 0; j < _vectorSize; ++j) {
				prod += vecX[j] * vecY[j];
			}
			_spearmanCorr[(ssize_t)row * _numVectors + col] = prod;
			_spearmanCorr[(ssize_t)col * _numVectors + row] = prod;
		}
	}

	/*recored the system time*/
	etime = getSysTime();
	fprintf(stderr, "Overall time: %f seconds\n",
			etime - stime);

#if 0
  for(int i = 0; i < _numVectors; ++i){
    for(int j = i; j < _numVectors; ++j){
      printf("%f ", _spearmanCorr[i *_numVectors + j]);
    }
    printf("\n");
  }
#endif
}

template<typename FloatType>
void SpearmanR<FloatType>::runMultiThreaded() {
	double stime, etime;
	const int tileDim = (_numVectors + SR_MT_TILE_DIM - 1) / SR_MT_TILE_DIM;
	const ssize_t numTiles = (ssize_t) (tileDim + 1) * tileDim / 2;
	ssize_t totalNumPairs = 0;
	const FloatType avg = 1.0 / (FloatType)_vectorSize;

#ifdef VERBOSE
  fprintf(stderr, "execute function %s\n", __FUNCTION__);
#endif

	/*record system time*/
	stime = getSysTime();

	/*allocate space*/
	_spearmanCorr = (FloatType*) mm_malloc(
			(ssize_t) _numVectors * _numVectors * sizeof(FloatType), 64);
	if (!_spearmanCorr) {
		fprintf(stderr, "Memory allocation failed\n");
		exit(-1);
	}
	_tiledPrMatrix = 0;

	/*enter the core computation*/
	if (_numCPUThreads < 1) {
		_numCPUThreads = omp_get_num_procs();
	}
	omp_set_num_threads(_numCPUThreads);

	/*entering the core loop*/
#pragma omp parallel reduction(+:totalNumPairs)
	{
		ssize_t chunkSize;
		int row, col, rank, j;
		int loRowRange, hiRowRange;
		int loColRange, hiColRange;
		int startColPerRow, endColPerRow;
		int rowStart, rowEnd, colStart, colEnd;
		ssize_t numPairsProcessed = 0;
		FloatType meanX, varX, x, prod;
		FloatType* __restrict__ vecX;
		FloatType* __restrict__ vecY;
		const int tid = omp_get_thread_num();
		const int nthreads = omp_get_num_threads();
		typedef CustomPair<FloatType, int> MyPair;
		MyPair* ranks = (MyPair*)mm_malloc(_vectorSize * sizeof(MyPair), 64);

		/*compute mean and variance*/
		chunkSize = (_numVectors + nthreads - 1) / nthreads;
		loRowRange = tid * chunkSize;
		hiRowRange = min(_numVectors, (tid + 1) * (int) chunkSize) - 1;

		vecX = _vectors + loRowRange * _vectorSizeAligned;
		for (row = loRowRange; row <= hiRowRange; ++row, vecX +=
				_vectorSizeAligned) {

    	/*get the rank vector*/
    	for(j = 0; j < _vectorSize; ++j){
      	ranks[j]._first = vecX[j];
      	ranks[j]._second = j;
    	}
    	qsort(ranks, _vectorSize, sizeof(CustomPair<FloatType, int>), CustomPair<FloatType, int>::ascendFirst);
	    for(j = 0, rank = 1; j < _vectorSize - 1; ++j){
     		vecX[ranks[j]._second] = rank;
    	  if(ranks[j]._first != ranks[j + 1]._first){
        	rank++;
  	   	}
 	   	}
			vecX[ranks[j]._second] = rank;

			/*do we need to normalize the ranks to avoid overflow? FIXME*/
    	for(j = 0; j < _vectorSize; ++j){
      	vecX[j] /= rank;
    	}

			/*compute the mean*/
			meanX = 0;
#pragma vector aligned
#pragma simd reduction(+:meanX)
			for (j = 0; j < _vectorSize; ++j) {
				meanX += vecX[j] * avg;
			}

			/*compute the variance*/
			varX = 0;
#pragma vector aligned
#pragma simd reduction(+:varX)
			for (j = 0; j < _vectorSize; ++j) {
				x = vecX[j] - meanX;
				varX += x * x;
			}
			varX = 1 / sqrt(varX);

			/*normalize the data*/
#pragma vector aligned
#pragma simd
			for (j = 0; j < _vectorSize; ++j) {
				x = vecX[j] - meanX;
				vecX[j] = x * varX;
			}
		}
		/*synchronize all threads*/
#pragma omp barrier

		/*compute pairwise correlation coefficient*/
		chunkSize = (numTiles + nthreads - 1) / nthreads;
		loRowRange = loColRange = tileDim + 1;
		if (tid * chunkSize < numTiles) {
			getTileCoordinate(tid * chunkSize, tileDim, loRowRange, loColRange);
		}
		hiRowRange = hiColRange = tileDim;
		if ((tid + 1) * chunkSize <= numTiles) {
			getTileCoordinate((tid + 1) * chunkSize - 1, tileDim, hiRowRange,
					hiColRange);
		}
		//fprintf(stderr, "tid (%d): %d %d %d %d\n", tid, loRowRange, loColRange, hiRowRange, hiColRange);
		for (int tileRow = loRowRange; tileRow <= hiRowRange; tileRow++) {
			/*compute the effective range per row*/
			startColPerRow = (tileRow == loRowRange) ? loColRange : tileRow;
			endColPerRow = (tileRow == hiRowRange) ? hiColRange : tileDim - 1;

			rowStart = tileRow * SR_MT_TILE_DIM;
			rowEnd = min(_numVectors, rowStart + SR_MT_TILE_DIM);
			for (int tileCol = startColPerRow; tileCol <= endColPerRow;
					++tileCol) {
				colStart = tileCol * SR_MT_TILE_DIM;
				colEnd = min(_numVectors, colStart + SR_MT_TILE_DIM);
				/*compute the tile*/
				vecX = _vectors + rowStart * _vectorSizeAligned;
				for (row = rowStart; row < rowEnd; ++row, vecX +=
						_vectorSizeAligned) {
					vecY = _vectors + colStart * _vectorSizeAligned;
					for (col = colStart; col < colEnd; ++col, vecY +=
							_vectorSizeAligned) {
						if(row > col){
							continue;
						}
						/*statistics*/
						++numPairsProcessed;

						prod = 0;
#pragma vector aligned
#pragma simd reduction(+:prod)
						for (int j = 0; j < _vectorSize; ++j) {
							prod += vecX[j] * vecY[j];
						}
						_spearmanCorr[(ssize_t)row * _numVectors + col] = prod;
						_spearmanCorr[(ssize_t)col * _numVectors + row] = prod;
					}
				}
			}
		}
		mm_free(ranks);

		/*reduction*/
		totalNumPairs += numPairsProcessed;
	} /*#pragma omp paralle*/
	
	/*recored the system time*/
	etime = getSysTime();
	fprintf(stderr, "Overall time (%ld pairs): %f seconds\n", totalNumPairs,
			etime - stime);

#if 0
  for(int i = 0; i < _numVectors; ++i){
    for(int j = i; j < _numVectors; ++j){
      printf("%f ", _spearmanCorr[i *_numVectors + j]);
    }
    printf("\n");
  }
#endif

}

#ifdef WITH_PHI
template<typename FloatType>
void SpearmanR<FloatType>::runSingleXeonPhi() {
	const int tileDim = (_numVectors + SR_PHI_TILE_DIM - 1) / SR_PHI_TILE_DIM;
	const ssize_t numTiles = (ssize_t)(tileDim + 1) * tileDim / 2;
	ssize_t indexRangeStart, indexRangeClose;
	ssize_t indexRangeStartPrev, indexRangeClosePrev;
	double stime, etime;
	FloatType* prValues[2];

#ifdef VERBOSE
  fprintf(stderr, "execute function %s\n", __FUNCTION__);
#endif

	/*set the number of threads on Xeon Phi*/
#pragma offload target(mic:_micIndex) inout(_numMICThreads)
	{
		/*set the number of threads*/
		if (_numMICThreads < SR_PHI_TILE_DIM) {
			_numMICThreads = omp_get_num_procs();
		}
		_numMICThreads = _numMICThreads / SR_PHI_TILE_DIM * SR_PHI_TILE_DIM;
		omp_set_num_threads (_numMICThreads);
	}
	fprintf(stderr, "number of threads: %d\n", _numMICThreads);

	/*allocate intermediate memory*/
	ssize_t maxPrValuesSize = SR_PHI_BUFFER_SIZE / sizeof(FloatType); /*a total of 512 MB*/
	ssize_t alignment = _numMICThreads * SR_PHI_TILE_SIZE;
	maxPrValuesSize = (maxPrValuesSize + alignment - 1) / alignment * alignment;
	for (int i = 0; i < 2; ++i) {
		prValues[i] = (FloatType*) mm_malloc(
				maxPrValuesSize * sizeof(FloatType), 64);
		if (!prValues[i]) {
			fprintf(stderr, "Memory allocation failed at line %d in file %s\n",
					__LINE__, __FILE__);
			exit(-1);
		}
	}

	/*align to the tile size*/
	_spearmanCorr = (FloatType*)mm_malloc((ssize_t)_numVectors * _numVectors * sizeof(FloatType), 64);
	if(!_spearmanCorr) {
		fprintf(stderr, "Memory allocation failed\n");
		exit(-1);
	}
	_tiledPrMatrix = 0;

	/*set the variables*/
  int signalVar, tileRowStart, tileRowEnd;
  int tileColStart, tileColEnd;
  ssize_t tileOffset, numElems, maxNumTilesPerPass;
  FloatType value;
	FloatType* __restrict__ inValues = prValues[0];	/*virtual memory address only on the host*/
	FloatType* __restrict__ outValues = prValues[1];
	FloatType* __restrict__ vectors = _vectors;
	FloatType* __restrict__ hostPrValues = prValues[0];	/*this virtual memory will be phisically allocated on the host*/

	fprintf(stderr, "Start transfering data\n");
#pragma offload_transfer target(mic: _micIndex) \
		in(vectors: length(_numVectors * _vectorSizeAligned) SR_MIC_ALLOC) \
		nocopy(inValues: length(maxPrValuesSize) SR_MIC_ALLOC) \
		nocopy(outValues: length(maxPrValuesSize) SR_MIC_ALLOC)

	fprintf(stderr, "Start computing\n");
	/*record system time*/
	stime = getSysTime();

	/*compute mean and variance*/
	double t1 = getSysTime();
#pragma offload target(mic:_micIndex) nocopy(vectors: SR_MIC_REUSE)
	{
#pragma omp parallel
		{
			int row, col, rank, j;
			int chunkSize, loRowRange, hiRowRange;
			FloatType x, y, r, meanX, varX;
			FloatType* __restrict__ vecX;
			const int tid = omp_get_thread_num();
			const int nthreads = omp_get_num_threads();
			const FloatType avg = 1.0 / (FloatType)_vectorSize;
			typedef CustomPair<FloatType, int> MyPair;
			MyPair* ranks = (MyPair*)mm_malloc(_vectorSize * sizeof(MyPair), 64);

			/*compute mean and variance*/
			chunkSize = (_numVectors + nthreads - 1) / nthreads;
			loRowRange = tid * chunkSize;
			hiRowRange = min(_numVectors, (tid + 1) * chunkSize);

			vecX = vectors + loRowRange * _vectorSizeAligned;
			for (row = loRowRange; row < hiRowRange; ++row, vecX +=
					_vectorSizeAligned) {

    		/*get the rank vector*/
      	for(j = 0; j < _vectorSize; ++j){
      	  ranks[j]._first = vecX[j];
        	ranks[j]._second = j;
      	}
      	qsort(ranks, _vectorSize, sizeof(CustomPair<FloatType, int>), CustomPair<FloatType, int>::ascendFirst);
	    	for(j = 0, rank = 1; j < _vectorSize - 1; ++j){
      		vecX[ranks[j]._second] = rank;
      		if(ranks[j]._first != ranks[j + 1]._first){
        		rank++;
      		}
   		 	}
				vecX[ranks[j]._second] = rank;
				/*do we need to normalize the ranks to avoid overflow? FIXME*/
    		for(j = 0; j < _vectorSize; ++j){
      		vecX[j] /= rank;
    		}

				/*compute the mean*/
				meanX = 0;
#pragma vector aligned
#pragma simd reduction(+:meanX)
				for (j = 0; j < _vectorSize; ++j) {
					meanX += vecX[j] * avg;
				}

				/*compute the variance*/
				varX = 0;
#pragma vector aligned
#pragma simd reduction(+:varX)
				for(j = 0; j < _vectorSize; ++j) {
					x = vecX[j] - meanX;
					varX += x * x;
				}
				varX = 1 / sqrt(varX);

				/*normalize the data*/
#pragma vector aligned
#pragma simd
				for(j = 0; j < _vectorSize; ++j) {
					x = vecX[j] - meanX;
					vecX[j] = x * varX;
				}
			}
			mm_free(ranks);
		}
	}
	double t2 = getSysTime();
	fprintf(stderr, "time for ranking and transformation: %f seconds\n", t2 - t1);

	/*the first round*/
	maxNumTilesPerPass = maxPrValuesSize / SR_PHI_TILE_SIZE;
	indexRangeStart = 0;
	indexRangeClose = min(numTiles, maxNumTilesPerPass);
#pragma offload target(mic:_micIndex) \
		nocopy(vectors: SR_MIC_REUSE)	\
		nocopy(inValues: SR_MIC_REUSE)	\
		in(indexRangeStart) \
		in(indexRangeClose) \
		signal(&signalVar)
	{
		/*compute*/
		_runSingleXeonPhiCore(vectors, inValues, indexRangeStart,
				indexRangeClose);
	}
	/*enter the core loop*/
	int round = 1;
	while (1) {
		/*wait the completion of the previous computation and swap the buffers*/
#pragma offload target(mic:_micIndex) \
    nocopy(inValues: SR_MIC_REUSE) \
    nocopy(outValues: SR_MIC_REUSE) \
    wait(&signalVar)
		{
				swap(inValues, outValues);
		}
		swap(inValues, outValues);

		/*save the finished round*/
		fprintf(stderr, "Round %d: %ld %ld\n", round++, indexRangeStart, indexRangeClose);
		indexRangeStartPrev = indexRangeStart;
		indexRangeClosePrev = indexRangeClose;
		if (indexRangeClose >= numTiles) {
			/*already finished the computation*/
			break;
		}

		/*compute the next range*/
		indexRangeStart += maxNumTilesPerPass;
		indexRangeClose = min(numTiles, maxNumTilesPerPass + indexRangeStart);
#pragma offload target(mic:_micIndex) \
		nocopy(vectors : SR_MIC_REUSE)	\
		nocopy(inValues: SR_MIC_REUSE)	\
		in(indexRangeStart)	\
		in(indexRangeClose)	\
		signal(&signalVar)
		{
			_runSingleXeonPhiCore(vectors, inValues, indexRangeStart,
					indexRangeClose);
		}

		/*device-to-host data transfer and save*/
		numElems = (indexRangeClosePrev - indexRangeStartPrev) * SR_PHI_TILE_SIZE;
		if(numElems > maxPrValuesSize){
			fprintf(stderr, "Error: %ld > %ld\n", numElems, maxPrValuesSize);
		}
#pragma offload_transfer target(mic:_micIndex) out(outValues[0:numElems]:into(hostPrValues[0:numElems]) SR_MIC_REUSE)

    /*save the data to the correlation matrix*/
		tileOffset = 0;
    for(ssize_t idx = indexRangeStartPrev; idx < indexRangeClosePrev; ++idx, tileOffset += SR_PHI_TILE_SIZE){
      /*get the row and column index of the tile*/
      getTileCoordinate(idx, tileDim, tileRowStart, tileColStart);

      /*compute the row range*/
      tileRowStart *= SR_PHI_TILE_DIM;
      tileColStart *= SR_PHI_TILE_DIM;
      tileRowEnd = min(_numVectors, tileRowStart + SR_PHI_TILE_DIM);
      tileColEnd = min(_numVectors, tileColStart + SR_PHI_TILE_DIM);

      /*compute each tile*/
			ssize_t rowOffset = tileOffset;
      for(int row = tileRowStart; row < tileRowEnd; ++row, rowOffset += SR_PHI_TILE_DIM){
        for(int col = tileColStart, colOffset = 0; col < tileColEnd; ++col, ++colOffset){
					if(row <= col){
            _spearmanCorr[(ssize_t)row * _numVectors + col] = hostPrValues[rowOffset + colOffset];
            _spearmanCorr[(ssize_t)col * _numVectors + row] = hostPrValues[rowOffset + colOffset];
          }
        }
      }
    }
	}
	/*transfer the remaining data*/
	numElems = (indexRangeClosePrev - indexRangeStartPrev) * SR_PHI_TILE_SIZE;
	if(numElems > 0) {
		if(numElems > maxPrValuesSize){
			fprintf(stderr, "Error: %ld > %ld\n", numElems, maxPrValuesSize);
		}
#pragma offload_transfer target(mic:_micIndex) out(outValues[0:numElems]:into(hostPrValues[0:numElems]) SR_MIC_REUSE)
    /*save the data to the correlation matrix*/
		tileOffset = 0;
    for(ssize_t idx = indexRangeStartPrev; idx < indexRangeClosePrev; ++idx, tileOffset += SR_PHI_TILE_SIZE){
      /*get the row and column index of the tile*/
      getTileCoordinate(idx, tileDim, tileRowStart, tileColStart);

      /*compute the row range*/
      tileRowStart *= SR_PHI_TILE_DIM;
      tileColStart *= SR_PHI_TILE_DIM;
      tileRowEnd = min(_numVectors, tileRowStart + SR_PHI_TILE_DIM);
      tileColEnd = min(_numVectors, tileColStart + SR_PHI_TILE_DIM);

      /*compute each tile*/
			ssize_t rowOffset = tileOffset;
      for(int row = tileRowStart; row < tileRowEnd; ++row, rowOffset += SR_PHI_TILE_DIM){
        for(int col = tileColStart, colOffset = 0; col < tileColEnd; ++col, ++colOffset){
					if(row <= col){
            _spearmanCorr[(ssize_t)row * _numVectors + col] = hostPrValues[rowOffset + colOffset];
            _spearmanCorr[(ssize_t)col * _numVectors + row] = hostPrValues[rowOffset + colOffset];
          }
        }
      }
    }
	}

	/*release memory on the Xeon Phi*/
#pragma offload_transfer target(mic: _micIndex) \
		nocopy(vectors : SR_MIC_FREE) \
		nocopy(inValues: SR_MIC_FREE) \
		nocopy(outValues: SR_MIC_FREE)

	/*release memory on the host*/
	for (int i = 0; i < 2; ++i) {
		mm_free(prValues[i]);
	}

	/*recored the system time*/
	etime = getSysTime();
	fprintf(stderr, "Overall time: %f seconds\n", etime - stime);

#if 0
  for(int i = 0; i < _numVectors; ++i){
    for(int j = i; j < _numVectors; ++j){
      printf("%f ", _spearmanCorr[i *_numVectors + j]);
    }
    printf("\n");
  }
#endif

}

template<typename FloatType>
__attribute__((target(mic))) void SpearmanR<FloatType>::_runSingleXeonPhiCore(
		FloatType* __restrict__ vectors, FloatType* __restrict__ prValues,
		const ssize_t indexRangeStart, const ssize_t indexRangeClose) {

#ifdef __MIC__
	const int tileDim = (_numVectors + SR_PHI_TILE_DIM -1) / SR_PHI_TILE_DIM;

#ifdef SOFT_BARRIER
	/*soft barrier*/
	_softBarrierInit(_numMICThreads / SR_PHI_TILE_DIM, SR_PHI_TILE_DIM);
#endif	/*soft barrier*/

#pragma omp parallel
	{
		const int numGroups = omp_get_num_threads() / SR_PHI_TILE_DIM;
		const int tid = omp_get_thread_num() % SR_PHI_TILE_DIM;
		const int gid = omp_get_thread_num() / SR_PHI_TILE_DIM;
		int localSense = 0;	/*must be zero here*/
		ssize_t offset, idx;
		int row, col, tileRowStart, tileRowEnd, tileColStart, tileColEnd;
		FloatType x, y, prod;
		FloatType* __restrict__ vecX;
		FloatType* __restrict__ vecY;

		/*one group processes on tile*/
		for (idx = indexRangeStart + gid; idx < indexRangeClose; idx += numGroups) {

			/*get the row and column index of the tile*/
			getTileCoordinate(idx, tileDim, tileRowStart, tileColStart);

			/*compute row range*/
			tileRowStart *= SR_PHI_TILE_DIM;
			tileRowEnd = tileRowStart + SR_PHI_TILE_DIM;
			if(tileRowEnd > _numVectors) {
				tileRowEnd = _numVectors;
			}

			/*each work thread computes its own column index*/
			offset = (idx - indexRangeStart) * SR_PHI_TILE_SIZE + tid;
			col = tileColStart * SR_PHI_TILE_DIM + tid;
			if(col < _numVectors) {
				vecY = vectors + col * _vectorSizeAligned;
				vecX = vectors + tileRowStart * _vectorSizeAligned;
				for(row = tileRowStart; row < tileRowEnd; ++row, vecX += _vectorSizeAligned, offset += SR_PHI_TILE_DIM) {
					if(row > col) {
						continue;
					}

					/*compute the dot product*/
					prod = 0;
#pragma vector aligned
#pragma simd reduction(+:prod)
					for (int j = 0; j < _vectorSize; ++j) {
						prod += vecX[j] * vecY[j];
					}

					/*save the data to its own buffer*/
					prValues[offset] = prod;
				}
			}
#ifdef SOFT_BARRIER
			_softBarrierSync(gid, SR_PHI_TILE_DIM, localSense);
#endif	/*soft barrier*/
		}
	}

#ifdef SOFT_BARRIER
	_softBarrierFinalize();
#endif	/*soft barrier*/
#endif	/*__MIC__*/
}

#ifdef WITH_PHI_ASSEMBLY_FLOAT
template<>
__attribute__((target(mic))) void SpearmanR<float>::_runSingleXeonPhiCore(
		float* __restrict__ vectors, float * __restrict__ prValues,
		const ssize_t indexRangeStart, const ssize_t indexRangeClose) {

#ifdef __MIC__
	const int tileDim = (_numVectors + SR_PHI_TILE_DIM -1) / SR_PHI_TILE_DIM;
#ifdef SOFT_BARRIER
  /*soft barrier*/
  _softBarrierInit(_numMICThreads / SR_PHI_TILE_DIM, SR_PHI_TILE_DIM);
#endif	/*soft barrier*/

#pragma omp parallel
	{
		const int numGroups = omp_get_num_threads() / SR_PHI_TILE_DIM;
		const int tid = omp_get_thread_num() % SR_PHI_TILE_DIM;
		const int gid = omp_get_thread_num() / SR_PHI_TILE_DIM;
		int localSense = 0;	/*must be zero here*/
		ssize_t offset, idx;
		int row, col, tileRowStart, tileRowEnd, tileColStart, tileColEnd;
		int vectorSizeRemainer, alignedVectorSize;
		register __m512 prod, x, y;
		float* __restrict__ vecX;
		float* __restrict__ vecY;

		/*one group processes on tile*/
		for (idx = indexRangeStart + gid; idx < indexRangeClose; idx += numGroups) {

			/*get the row and column index of the tile*/
			getTileCoordinate(idx, tileDim, tileRowStart, tileColStart);

			/*compute row range*/
			tileRowStart *= SR_PHI_TILE_DIM;
			tileRowEnd = tileRowStart + SR_PHI_TILE_DIM;
			if(tileRowEnd > _numVectors) {
				tileRowEnd = _numVectors;
			}

			/*each work thread computes its own column index*/
			offset = (idx - indexRangeStart) * SR_PHI_TILE_SIZE + tid;
			col = tileColStart * SR_PHI_TILE_DIM + tid;
			if(col < _numVectors) {
				vecY = vectors + col * _vectorSizeAligned;
				vecX = vectors + tileRowStart * _vectorSizeAligned;
				for(row = tileRowStart; row < tileRowEnd; ++row, vecX += _vectorSizeAligned, offset += SR_PHI_TILE_DIM) {
					if(row > col) {
						continue;
					}

					/*compute the correlation*/
					prod = _mm512_setzero_ps();
					/*aligned to 16*/
					vectorSizeRemainer = _vectorSize & 15;
					alignedVectorSize = _vectorSize - vectorSizeRemainer;
					for (int i = 0; i < alignedVectorSize; i += 16) {
						x = _mm512_load_ps(vecX + i);
						y = _mm512_load_ps(vecY + i);
						prod = _mm512_fmadd_ps(x, y, prod);
					}
					if(vectorSizeRemainer) {
						__mmask16 mask = _mm512_int2mask((1 << vectorSizeRemainer) - 1);
						x = _mm512_load_ps(vecX + alignedVectorSize);
						y = _mm512_load_ps(vecY + alignedVectorSize);
						prod = _mm512_mask3_fmadd_ps(x, y, prod, mask);
					}

					/*save the data to its own buffer*/
					prValues[offset] = _mm512_reduce_add_ps(prod);
				}
			}
#ifdef SOFT_BARRIER
      _softBarrierSync(gid, SR_PHI_TILE_DIM, localSense);
#endif	/*soft barrier*/
		}
	}
#ifdef SOFT_BARRIER
  _softBarrierFinalize();
#endif	/*soft barrier*/
#endif	/*__MIC__*/
}	
#endif	/*WITH_PHI_ASSEMBLY_FLOAT*/

#ifdef WITH_PHI_ASSEMBLY_DOUBLE
template<>
__attribute__((target(mic))) void SpearmanR<double>::_runSingleXeonPhiCore(
		double* __restrict__ vectors, double* __restrict__ prValues,
		const ssize_t indexRangeStart, const ssize_t indexRangeClose) {

#ifdef __MIC__
	const int tileDim = (_numVectors + SR_PHI_TILE_DIM -1) / SR_PHI_TILE_DIM;
#ifdef SOFT_BARRIER
  /*soft barrier*/
  _softBarrierInit(_numMICThreads / SR_PHI_TILE_DIM, SR_PHI_TILE_DIM);
#endif	/*soft barrier*/

#pragma omp parallel
	{
		const int numGroups = omp_get_num_threads() / SR_PHI_TILE_DIM;
		const int tid = omp_get_thread_num() % SR_PHI_TILE_DIM;
		const int gid = omp_get_thread_num() / SR_PHI_TILE_DIM;
		int localSense = 0;	/*must be zero here*/
		ssize_t offset, idx;
		int row, col, tileRowStart, tileRowEnd, tileColStart, tileColEnd;
		int vectorSizeRemainer, alignedVectorSize;
		register __m512d prod, x, y;
		double* __restrict__ vecX;
		double* __restrict__ vecY;

		/*one group processes on tile*/
		for (idx = indexRangeStart + gid; idx < indexRangeClose; idx += numGroups) {

			/*get the row and column index of the tile*/
			getTileCoordinate(idx, tileDim, tileRowStart, tileColStart);

			/*compute row range*/
			tileRowStart *= SR_PHI_TILE_DIM;
			tileRowEnd = tileRowStart + SR_PHI_TILE_DIM;
			if(tileRowEnd > _numVectors) {
				tileRowEnd = _numVectors;
			}

			/*each work thread computes its own column index*/
			offset = (idx - indexRangeStart) * SR_PHI_TILE_SIZE + tid;
			col = tileColStart * SR_PHI_TILE_DIM + tid;
			if(col < _numVectors) {
				vecY = vectors + col * _vectorSizeAligned;
				vecX = vectors + tileRowStart * _vectorSizeAligned;
				for(row = tileRowStart; row < tileRowEnd; ++row, vecX += _vectorSizeAligned, offset += SR_PHI_TILE_DIM) {
					if(row > col) {
						continue;
					}

					/*compute the correlation*/
					prod = _mm512_setzero_pd();
					/*aligned to 8*/
					vectorSizeRemainer = _vectorSize & 7;
					alignedVectorSize = _vectorSize - vectorSizeRemainer;
					for (int i = 0; i < alignedVectorSize; i += 8) {
						x = _mm512_load_pd(vecX + i);
						y = _mm512_load_pd(vecY + i);
						prod = _mm512_fmadd_pd(x, y, prod);
					}
					if(vectorSizeRemainer) {
						__mmask16 mask = _mm512_int2mask((1 << vectorSizeRemainer) - 1);
						x = _mm512_load_pd(vecX + alignedVectorSize);
						y = _mm512_load_pd(vecY + alignedVectorSize);
						prod = _mm512_mask3_fmadd_pd(x, y, prod, mask);
					}

					/*save the data to its own buffer*/
					prValues[offset] = _mm512_reduce_add_pd(prod);
				}
			}
#ifdef SOFT_BARRIER
      _softBarrierSync(gid, SR_PHI_TILE_DIM, localSense);
#endif	/*soft barrier*/
		}
	}
#ifdef SOFT_BARRIER
  _softBarrierFinalize();
#endif	/*soft barrier*/
#endif	/*__MIC__*/
}
#endif	/*WITH_PHI_ASSEMBLY_DOUBLE*/

#endif	/*with phi*/


#ifdef WITH_MPI
template<typename FloatType>
void SpearmanR<FloatType>::runMPICPU() {
	double stime, etime;
	const int tileDim = (_numVectors + SR_MPI_TILE_DIM -1) / SR_MPI_TILE_DIM;
	const ssize_t numTiles = (ssize_t)(tileDim + 1) * tileDim / 2;
	const ssize_t numPairs = (ssize_t)(_numVectors + 1) * _numVectors / 2;	/*include self-vs-self*/
	ssize_t numPairsProcessed = 0, totalNumPairs;

#ifdef VERBOSE
  if(_rank == 0) fprintf(stderr, "execute function %s\n", __FUNCTION__);
#endif

	/*record the system time*/
	MPI_Barrier(MPI_COMM_WORLD);
	stime = getSysTime();

	/*compute the mean and variance*/
	int row, col;
	int loRowRange, hiRowRange;
	int loColRange, hiColRange;
	int startColPerRow, endColPerRow;
	int rowStart, rowEnd, colStart, colEnd;
	ssize_t offset = 0, rowOffset, colOffset;
	FloatType x, meanX, varX, prod;
	FloatType* __restrict__ vecX;
	FloatType* __restrict__ vecY;
	const FloatType avg = 1.0 / (FloatType)_vectorSize;

	/*allocate buffer*/
	ssize_t chunkSize = (numTiles + _numProcs - 1) / _numProcs;
	_spearmanCorr = (FloatType*)mm_malloc( chunkSize * SR_MPI_TILE_DIM * SR_MPI_TILE_DIM * sizeof(FloatType), 64);
	if(!_spearmanCorr) {
		fprintf(stderr, "Memory allocation failed\n");
		exit(-1);
	}
	_tiledPrMatrix = 1;

	/*normalize the data*/
	chunkSize = (_numVectors + _numProcs - 1) / _numProcs;
	loRowRange = _rank * chunkSize;
	hiRowRange = min(_numVectors, (_rank + 1) * (int) chunkSize);
	vecX = _vectors + loRowRange * _vectorSizeAligned;
 	
	typedef CustomPair<FloatType, int> MyPair;
	MyPair* ranks = (MyPair*)mm_malloc(_vectorSize * sizeof(MyPair), 64);
	for (row = loRowRange; row < hiRowRange; ++row, vecX +=
			_vectorSizeAligned) {
 
  	/*get the rank vector*/
		int j, rank;
   	for(j = 0; j < _vectorSize; ++j){
   		ranks[j]._first = vecX[j];
    	ranks[j]._second = j;
   	}
   	qsort(ranks, _vectorSize, sizeof(MyPair), MyPair::ascendFirst);
   	for(j = 0, rank = 1; j < _vectorSize - 1; ++j){
     	vecX[ranks[j]._second] = rank;
    	if(ranks[j]._first != ranks[j + 1]._first){
      	rank++;
     	}
    }
   	vecX[ranks[j]._second] = rank;

		/*do we need to normalize the ranks to avoid overflow? FIXME*/
		for(j = 0; j < _vectorSize; ++j){
			vecX[j] /= rank;
		}

		/*compute the mean*/
		meanX = 0;
#pragma vector aligned
#pragma simd reduction(+:meanX)
		for (j = 0; j < _vectorSize; ++j) {
			meanX += vecX[j] * avg;
		}

		/*compute the variance*/
		varX = 0;
#pragma vector aligned
#pragma simd reduction(+:varX)
		for (j = 0; j < _vectorSize; ++j) {
			x = vecX[j] - meanX;
			varX += x * x;
		}
		varX = 1 / sqrt(varX);

		/*normalize the data*/
#pragma vector aligned
#pragma simd
		for (j = 0; j < _vectorSize; ++j) {
			x = vecX[j] - meanX;
			vecX[j] = x * varX;
		}
	}
	mm_free(ranks);

	/*all gather to communicate the data*/
	int* displs = (int*)mm_malloc(_numProcs * sizeof(int), 64);
	int* recvCounts = (int*)mm_malloc(_numProcs * sizeof(int), 64);
	row = 0;
	for(int i = 0; i < _numProcs; ++i) {
		displs[i] = row;
		recvCounts[i] = min((int)chunkSize, _numVectors - row);
		row += chunkSize;
	}
	const int key = loRowRange < hiRowRange ? 1 : 0;
	MPI_Comm mycomm;
	MPI_Comm_split(MPI_COMM_WORLD, key, _rank, &mycomm);
	if(typeid(FloatType) == typeid(float)) {
		if(key) {
			MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_FLOAT, _vectors, recvCounts, displs, MPI_FLOAT, mycomm);
		}
	} else if(typeid(FloatType) == typeid(double)) {
		if(key) {
			MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DOUBLE, _vectors, recvCounts, displs, MPI_DOUBLE, mycomm);
		}
	} else {
		printf("Only support floating-point values\n");
		return;
	}
	mm_free(displs);
	mm_free(recvCounts);
	MPI_Comm_free(&mycomm);
	MPI_Barrier(MPI_COMM_WORLD);

	/*get the tile range*/
	chunkSize = (numTiles + _numProcs - 1) / _numProcs;
	loRowRange = loColRange = tileDim + 1;
	if(_rank * chunkSize < numTiles) {
		getTileCoordinate(_rank * chunkSize, tileDim, loRowRange, loColRange);
	}
	hiRowRange = hiColRange = tileDim;
	if((_rank + 1) * chunkSize <= numTiles) {
		getTileCoordinate((_rank + 1) * chunkSize - 1, tileDim,
				hiRowRange, hiColRange);
	}
	offset = 0;
	//fprintf(stderr, "proc(%d): %d %d %d %d\n", _rank, loRowRange, loColRange, hiRowRange, hiColRange);
	for (int tileRow = loRowRange; tileRow <= hiRowRange; tileRow++) {
		rowStart = tileRow * SR_MPI_TILE_DIM;
		rowEnd = min(_numVectors, rowStart + SR_MPI_TILE_DIM);
		startColPerRow = (tileRow == loRowRange) ? loColRange : tileRow;
		endColPerRow = (tileRow == hiRowRange) ? hiColRange : tileDim - 1;
		for (int tileCol = startColPerRow; tileCol <= endColPerRow;
				++tileCol) {
			colStart = tileCol * SR_MPI_TILE_DIM;
			colEnd = min(_numVectors, colStart + SR_MPI_TILE_DIM);
			/*compute each tile*/
			vecX = _vectors + rowStart * _vectorSizeAligned;
			for (row = rowStart, rowOffset = offset; row < rowEnd; row++, vecX +=
					_vectorSizeAligned, rowOffset += SR_MPI_TILE_DIM) {
				vecY = _vectors + colStart * _vectorSizeAligned;
				for (col = colStart, colOffset = rowOffset; col < colEnd; ++col, vecY +=
						_vectorSizeAligned, colOffset++) {
					if (row > col) {
						continue;
					}

					/*statistics*/
					numPairsProcessed++;

					/*compute the dot product*/
					prod = 0;
#pragma vector aligned
#pragma simd reduction(+:prod)
					for (int j = 0; j < _vectorSize; ++j) {
						prod += vecX[j] * vecY[j];
					}
					_spearmanCorr[colOffset] = prod;
				}
			}
			/*move to the next tile*/
			offset += SR_MPI_TILE_SIZE;
		}
	}

	/*compute the total number of edges*/
	MPI_Reduce(&numPairsProcessed, &totalNumPairs, 1, MPI_INT64_T, MPI_SUM, 0, MPI_COMM_WORLD);

	/*recored the system time*/
	etime = getSysTime();
	if (_rank == 0) {
		fprintf(stderr, "Overall time (%ld pairs): %f seconds\n", totalNumPairs, etime - stime);
	}
}

#ifdef WITH_PHI
template<typename FloatType>
void SpearmanR<FloatType>::runMPIXeonPhi() {
	const int tileDim = (_numVectors + SR_PHI_TILE_DIM - 1) / SR_PHI_TILE_DIM;
	const ssize_t numTiles = (ssize_t)(tileDim + 1) * tileDim / 2;
	ssize_t indexRangeStart, indexRangeClose, indexRangeEnd;
	ssize_t indexRangeStartPrev, indexRangeClosePrev;
	double stime, etime;
	FloatType* prValues[2];

#ifdef VERBOSE
  if(_rank == 0) fprintf(stderr, "execute function %s\n", __FUNCTION__);
#endif

	/*set the number of threads on the Xeon Phi*/
#pragma offload target(mic:_micIndex) inout(_numMICThreads)
	{
		/*set the number of threads*/
		if(_numMICThreads < SR_PHI_TILE_DIM) {
			_numMICThreads = omp_get_num_procs();
		}
		_numMICThreads = _numMICThreads / SR_PHI_TILE_DIM * SR_PHI_TILE_DIM;
		omp_set_num_threads(_numMICThreads);
	}

	/*allocate memory*/
	ssize_t maxPrValuesSize = SR_PHI_BUFFER_SIZE / sizeof(FloatType); /*a total of 512 MB*/
	ssize_t alignment = _numMICThreads * SR_PHI_TILE_SIZE;
	maxPrValuesSize = (maxPrValuesSize + alignment - 1) / alignment * alignment;
	for (int i = 0; i < 2; ++i) {
		prValues[i] = (FloatType*) mm_malloc(maxPrValuesSize * sizeof(FloatType), 64);
		if (!prValues[i]) {
			fprintf(stderr, "Memory allocation failed at line %d in file %s\n",
					__LINE__, __FILE__);
			exit(-1);
		}
	}
	const ssize_t chunkSize = (numTiles + _numProcs - 1) / _numProcs;
	_spearmanCorr = (FloatType*)mm_malloc( chunkSize * SR_PHI_TILE_SIZE * sizeof(FloatType), 64);
	if(!_spearmanCorr) {
		fprintf(stderr, "Memory allocation failed\n");
		exit(-1);
	}
	_tiledPrMatrix = 1;

	/*set the variables*/
	int signalVar;
	ssize_t offset = 0, numElems, maxNumTilesPerPass;
	FloatType* __restrict__ inValues = prValues[0];
	FloatType* __restrict__ outValues = prValues[1];
	FloatType* __restrict__ vectors = _vectors;

	/*transfer data*/
	if(_rank == 0) {
		fprintf(stderr, "Start transfering data\n");
	}
#pragma offload_transfer target(mic: _micIndex) \
	in(vectors: length(_numVectors * _vectorSizeAligned) SR_MIC_ALLOC) \
	nocopy(inValues: length(maxPrValuesSize) SR_MIC_ALLOC) \
	nocopy(outValues: length(maxPrValuesSize) SR_MIC_ALLOC)

	/*record system time*/
	if(_rank == 0) {
		fprintf(stderr, "Start computing\n");
	}
	MPI_Barrier(MPI_COMM_WORLD);
	stime = getSysTime();

	/*normalize the data*/
	//double t1 = getSysTime();
#pragma offload target(mic:_micIndex) nocopy(vectors: SR_MIC_REUSE)
	{
#pragma omp parallel
		{
			int row, col, rank, j;
			int chunkSize, loRowRange, hiRowRange;
			FloatType x, y, r, meanX, varX;
			FloatType* __restrict__ vecX;
			const int tid = omp_get_thread_num();
			const int nthreads = omp_get_num_threads();
			const FloatType avg = 1.0 / (FloatType)_vectorSize;
			typedef CustomPair<FloatType, int> MyPair;
			MyPair* ranks = (MyPair*)mm_malloc(_vectorSize * sizeof(MyPair), 64);

			/*compute mean and variance*/
			chunkSize = (_numVectors + nthreads - 1) / nthreads;
			loRowRange = tid * chunkSize;
			hiRowRange = (tid + 1) * chunkSize;
			if(hiRowRange > _numVectors) {
				hiRowRange = _numVectors;
			}

			vecX = vectors + loRowRange * _vectorSizeAligned;
			for (row = loRowRange; row < hiRowRange; ++row, vecX +=
					_vectorSizeAligned) {
        /*get the rank vector*/
        for(j = 0; j < _vectorSize; ++j){
          ranks[j]._first = vecX[j];
          ranks[j]._second = j;
        }
        qsort(ranks, _vectorSize, sizeof(CustomPair<FloatType, int>), CustomPair<FloatType, int>::ascendFirst);
 		   	for(j = 0, rank = 1; j < _vectorSize - 1; ++j){
    			vecX[ranks[j]._second] = rank;
      		if(ranks[j]._first != ranks[j + 1]._first){
        		rank++;
      		}
    		}
				vecX[ranks[j]._second] = rank;

				/*do we need to normalize the ranks to avoid overflow? FIXME*/
    		for(j = 0; j < _vectorSize; ++j){
      		vecX[j] /= rank;
    		}

				/*compute the mean*/
				meanX = 0;
#pragma vector aligned
#pragma simd reduction(+:meanX)
				for (j = 0; j < _vectorSize; ++j) {
					meanX += vecX[j] * avg;
				}

				/*compute the variance*/
				varX = 0;
#pragma vector aligned
#pragma simd reduction(+:varX)
				for(j = 0; j < _vectorSize; ++j) {
					x = vecX[j] - meanX;
					varX += x * x;
				}
				varX = 1 / sqrt(varX);

				/*normalize the data*/
#pragma vector aligned
#pragma simd
				for(j = 0; j < _vectorSize; ++j) {
					x = vecX[j] - meanX;
					vecX[j] = x * varX;
				}
			}
			mm_free(ranks);
		}
	}
	//double t2 = getSysTime();
	//fprintf(stderr, "time for ranking and transformation: %f seconds\n", t2 - t1);

	/*the first round*/
	maxNumTilesPerPass = maxPrValuesSize / SR_PHI_TILE_SIZE;
	indexRangeStart = min(numTiles, _rank * chunkSize);
	indexRangeEnd = min(numTiles, (_rank + 1) * chunkSize);
	indexRangeClose = min(indexRangeEnd, indexRangeStart + maxNumTilesPerPass);
#pragma offload target(mic:_micIndex) \
	nocopy(vectors: SR_MIC_REUSE)	\
	nocopy(inValues: SR_MIC_REUSE)	\
	in(indexRangeStart) \
	in(indexRangeClose) \
	signal(&signalVar)
	{
		/*compute*/
		_runSingleXeonPhiCore(vectors, inValues, indexRangeStart, indexRangeClose);
	}

	/*enter the core loop*/
	int round = 1;
	while (1) {
		/*wait the completion of the previous computation*/
#pragma offload target(mic:_micIndex) \
    nocopy(inValues: SR_MIC_REUSE) \
    nocopy(outValues: SR_MIC_REUSE) \
    wait(&signalVar)
		{
      FloatType* tmp = inValues;
      inValues = outValues;
      outValues = tmp;
		}
  	swap(inValues, outValues);

		/*statistics*/
		//fprintf(stderr, "Round %d: (proc %d) %ld %ld\n", round++, _rank, indexRangeStart, indexRangeClose);
		indexRangeStartPrev = indexRangeStart;
		indexRangeClosePrev = indexRangeClose;
		if (indexRangeClose >= indexRangeEnd) {
			break;
		}

		/*initiate another computing kernel?*/
		indexRangeStart += maxNumTilesPerPass;
		indexRangeClose = min(indexRangeEnd, indexRangeStart + maxNumTilesPerPass);
#pragma offload target(mic:_micIndex) \
	nocopy(vectors : SR_MIC_REUSE)	\
	nocopy(inValues: SR_MIC_REUSE)	\
	in(indexRangeStart)	\
	in(indexRangeClose)	\
	signal(&signalVar)
		{
			_runSingleXeonPhiCore(vectors, inValues, indexRangeStart, indexRangeClose);
		}
		numElems = (indexRangeClosePrev - indexRangeStartPrev) * SR_PHI_TILE_SIZE;
#pragma offload_transfer target(mic:_micIndex) out(outValues[0:numElems]:into(_spearmanCorr[offset:numElems]) SR_MIC_REUSE)
		offset += numElems;
	}
	/*process the data in the last round*/
	numElems = (indexRangeClosePrev - indexRangeStartPrev) * SR_PHI_TILE_SIZE;
	if(numElems > 0) {
#pragma offload_transfer target(mic:_micIndex) out(outValues[0:numElems]:into(_spearmanCorr[offset:numElems]) SR_MIC_REUSE)
	}

	/*release memory on the Xeon Phi*/
#pragma offload_transfer target(mic: _micIndex) \
	nocopy(vectors : SR_MIC_FREE) \
	nocopy(inValues: SR_MIC_FREE) \
	nocopy(outValues: SR_MIC_FREE)

	/*release memory on the host*/
	for (int i = 0; i < 2; ++i) {
		mm_free(prValues[i]);
	}

	/*recored the system time*/
	MPI_Barrier(MPI_COMM_WORLD);
	etime = getSysTime();
	if(_rank == 0) {
		fprintf(stderr, "Overall time: %f seconds\n", etime - stime);
	}
}
#endif	/*with phi*/
#endif	/*with mpi*/

template <typename FloatType>
void SpearmanR<FloatType>::transpose()
{
	int newNumVectors = _vectorSize;
	int newVectorSize = _numVectors;
	int alignmentFactor = 64 / sizeof(FloatType);
	int newVectorSizeAligned = (newVectorSize + alignmentFactor - 1) / alignmentFactor * alignmentFactor;
	FloatType *newVectors, *invectors, tmp;

	/*allocate new memory*/
	newVectors = (FloatType*)mm_malloc(newNumVectors * newVectorSizeAligned * sizeof(FloatType), 64);
#if 0
	fprintf(stderr, "Before transpose:\n");
	for(int row = 0; row < _numVectors; ++row){
		invectors = _vectors + row * _vectorSizeAligned;
		for(int col = 0; col < _vectorSize; ++col){
			tmp = *(invectors + col);
			fprintf(stderr, "%f ", tmp);
		}
		fprintf(stderr, "\n");
	}
#endif
	
	/*transpose the matrix*/
	for(int row = 0; row < _numVectors; ++row){
		invectors = _vectors + row * _vectorSizeAligned;
		for(int col = 0; col < _vectorSize; ++col){
			tmp = *(invectors + col);
			*(newVectors + col * newVectorSizeAligned + row) = tmp;
		}
	}
	mm_free(_vectors);
	_numVectors = newNumVectors;
	_vectorSize = newVectorSize;
	_vectorSizeAligned = newVectorSizeAligned;
	_vectors = newVectors;

#if 0
	fprintf(stderr, "After transposing:\n");
   for(int row = 0; row < _numVectors; ++row){
     invectors = _vectors + row * _vectorSizeAligned;
     for(int col = 0; col < _vectorSize; ++col){
       tmp = *(invectors + col);
       fprintf(stderr, "%f ", tmp);
     }
     fprintf(stderr, "\n");
   }
#endif
}


#endif /* PEARSONR_HPP_ */
