/*
 * DistanceCorr.hpp
 *
 *  Created on: Mar 22, 2016
 *  Author: Liu, Yongchao
 *  Affiliation: School of Computational Science & Engineering
 *  						Georgia Institute of Technology, Atlanta, GA 30332
 *  URL: www.liuyc.org
 */

#ifndef DISTANCE_HPP_
#define DISTANCE_HPP_
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
#define DT_MIC_REUSE alloc_if(0) free_if(0)
#define DT_MIC_ALLOC alloc_if(1) free_if(0)
#define DT_MIC_FREE  length(0) alloc_if(0) free_if(1)
#define DT_MIC_ALLOC_FREE alloc_if(1) free_if(1)
#endif	/*with phi*/

/*tile size*/
#define DT_PHI_TILE_DIM	4		/*must be divided by 236*/
#define DT_PHI_TILE_SIZE	(DT_PHI_TILE_DIM * DT_PHI_TILE_DIM)
#define DT_MT_TILE_DIM		8
#define DT_MPI_TILE_DIM	8
#define DT_MPI_TILE_SIZE	(DT_MPI_TILE_DIM * DT_MPI_TILE_DIM)

/*maximum dimension size*/
#ifndef DT_MAX_MAT_DIM
#define DT_MAX_MAT_DIM			(1<<25)
#endif

/*maximum Xeon Phi buffer size*/
#define DT_PHI_BUFFER_SIZE (1 << 29)

/*template class*/
template<typename FloatType>
class DistanceCorr {
public:
	DistanceCorr(int numVectors, int vectorSize, int numCPUThreads,
			int numMICThreads, int micIndex, int rank, int numProcs);
	~DistanceCorr();

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

	/*transpose matrix*/
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
	FloatType* _distanceCorr; /*pearson correlation matrix*/

	/*used for software barrier*/
	int* _barriers;
	int* _counts;

	/*for Xeon Phi*/
#ifdef WITH_PHI
	__attribute__((target(mic)))
#endif
	void _softBarrierInit(const int numGroups)
	{
		_barriers = (int*)mm_malloc(numGroups * sizeof(int), 64);
		_counts = (int*)mm_malloc(numGroups * sizeof(int), 64);
		for(int i = 0; i < numGroups; ++i){
			_counts[i] = 0;
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
	void _softBarrierSync(const int groupId, const int groupSize, int& localSense)
	{
		localSense = !localSense;	/*inverse local variable*/
		if(__sync_fetch_and_add(_counts + groupId, 1) == groupSize - 1){
			__sync_fetch_and_add(_counts + groupId, 0);
			__sync_bool_compare_and_swap(_barriers + groupId, !localSense, localSense);
			//_barriers[groupId] = localSense;
			//_counts[groupId] = 0;
		}else{
			//while(_barriers[groupId] != localSense);
			while(__sync_fetch_and_sub(_barriers + groupId, 0) != localSense);
		}
	}
#ifdef WITH_PHI
  __attribute__((target(mic)))
#endif
	void _runSingleXeonPhiCore(
			FloatType* __restrict__ vectors, FloatType* __restrict__ prValues, const ssize_t indexRangeStart, const ssize_t indexRangeClose);
	

#ifdef WITH_PHI
  __attribute__((target(mic)))
#endif
  FloatType _computeDistCorr(FloatType* meanX, FloatType* meanY, FloatType* vecX, FloatType* vecY, const int numElems);

	/*static function members*/
public:
	/*when matrix dimension size is large than 2^21, we should use long double instead of double*/
#ifdef WITH_PHI
	__attribute__((target(mic)))
#endif
	inline void getCoordinate(const ssize_t globalIndex,
			const ssize_t matrixDimSize, int& row, int& col) {
#if DT_MAX_MAT_DIM < (1 << 25)
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
#if DT_MAX_MAT_DIM < (1 << 25)
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
#if DT_MAX_MAT_DIM < (1 << 25)
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
#if DT_MAX_MAT_DIM < (1 << 25)
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
DistanceCorr<FloatType>::DistanceCorr(int numVectors, int vectorSize, int numCPUThreads,
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
	if (_numVectors > DT_MAX_MAT_DIM) {
		fprintf(stderr,
				"The number of vectors exceeds the maximum limit (%ld)\n",
				(ssize_t) DT_MAX_MAT_DIM);
		exit(-1);
	}

	/*allocate space*/
	_distanceCorr = NULL;

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
DistanceCorr<FloatType>::~DistanceCorr() {
	if (_vectors) {
		mm_free(_vectors);
	}
	if (_distanceCorr) {
		mm_free(_distanceCorr);
	}
}

template<typename FloatType>
#ifdef WITH_PHI
  __attribute__((target(mic)))
#endif
FloatType DistanceCorr<FloatType>::_computeDistCorr(FloatType* meanX, FloatType* meanY, FloatType* vecX, FloatType* vecY,
	const int numElems)
{
	int i, j;
	FloatType x, y, w, z, s, t;
	FloatType cov2XX, cov2YY, cov2XY, grandMeanX, grandMeanY;
	const FloatType factor = 1.0 / (FloatType)numElems;

	/*compute row (or column) mean*/
	for(i = 0; i < numElems; ++i){
		x = vecX[i];
		y = vecY[i];
		w = z = 0;
#pragma vector aligned
#pragma simd reduction(+:w, z)
		for(j = 0; j < numElems; ++j){
			w += fabs(x - vecX[j]);
			z += fabs(y - vecY[j]);
		}
		meanX[i] = w * factor;
		meanY[i] = z * factor;
	}

	/*compute marginal mean and grand mean*/
	grandMeanX = grandMeanY = 0;
#pragma vector aligned
#pragma simd reduction(+:grandMeanX, grandMeanY)
	for(i = 0; i < numElems; ++i){
		/*fmad instruction*/
		grandMeanX += meanX[i] * factor;
		grandMeanY += meanY[i] * factor;
	}

	/*compute the covariance (only upper triangle)*/
	cov2XX = cov2YY = cov2XY = 0;
	for(i = 1; i < numElems; ++i){
		x = vecX[i];
		y = vecY[i];
		s = grandMeanX - meanX[i];
		t = grandMeanY - meanY[i];

#pragma vector aligned
#pragma simd reduction(+: cov2XX, cov2XY, cov2YY)
		for(j = 0; j < i; ++j){
			w = fabs(x - vecX[j]) - meanX[j] + s;
			z = fabs(y - vecY[j]) - meanY[j] + t;
			w *= factor;
			z *= factor;

			cov2XX += w * w;
			cov2XY += w * z;
			cov2YY += z * z;
		}
	}
	cov2XX *= 2;
	cov2XY *= 2;
	cov2YY *= 2;

	/*major diagonal*/
#pragma vector aligned
#pragma simd reduction(+:cov2XX, cov2XY, cov2YY)
	for(i = 0; i < numElems; ++i){
		x = -2 * meanX[i] + grandMeanX;
		y = -2 * meanY[i] + grandMeanY;
		x *= factor;
		y *= factor;

		cov2XX += x * x;
		cov2XY += x * y;
		cov2YY += y * y;
	}

	cov2XX = sqrt(cov2XX);
	cov2XY = sqrt(cov2XY);
	cov2YY = sqrt(cov2YY);

	return cov2XY / sqrt(cov2XX * cov2YY);
}
template<typename FloatType>
void DistanceCorr<FloatType>::generateRandomData(const int seed) {
	srand48(11);
	for (int i = 0; i < _numVectors; ++i) {
		FloatType* __restrict__ dst = _vectors + i * _vectorSizeAligned;
		for (int j = 0; j < _vectorSize; ++j) {
			dst[j] = drand48();
		}
	}
}
template<typename FloatType>
void DistanceCorr<FloatType>::runSingleThreaded() {
	double stime, etime;
#ifdef VERBOSE
	fprintf(stderr, "execute function %s\n", __FUNCTION__);
#endif

	/*output matrix*/
	_distanceCorr = (FloatType*) mm_malloc(
			(ssize_t) _numVectors * _numVectors * sizeof(FloatType), 64);
	if (!_distanceCorr) {
		fprintf(stderr, "Memory allocation failed\n");
		exit(-1);
	}
	_tiledPrMatrix = 0;

	/*record the system time*/
	stime = getSysTime();

	/*allocate vectors for mean and variance*/
	double t1 = getSysTime();
	const ssize_t numRankPairs = (ssize_t)_vectorSize * (_vectorSize - 1) / 2;
	FloatType* __restrict__ vecX;
	FloatType* __restrict__ vecY;
	const FloatType avg = 1.0 / (FloatType)_vectorSize;

	vecX = _vectors;
	FloatType* meanX = (FloatType*)mm_malloc(_vectorSize * sizeof(FloatType), 64);
	FloatType* meanY = (FloatType*)mm_malloc(_vectorSize * sizeof(FloatType), 64);
	/*compare pairwise correlation*/
	for (int row = 0; row < _numVectors; ++row, vecX += _vectorSizeAligned) {
		vecY = _vectors + row * _vectorSizeAligned;
		for (int col = row; col < _numVectors; ++col, vecY +=
				_vectorSizeAligned) {

			/*save the correlation coefficient*/
			FloatType corr = _computeDistCorr(meanX, meanY, vecX, vecY, _vectorSize);
			_distanceCorr[(ssize_t)row * _numVectors + col] = corr;
			_distanceCorr[(ssize_t)col * _numVectors + row] = corr;
		}
	}
	mm_free(meanX);
	mm_free(meanY);

	/*recored the system time*/
	etime = getSysTime();
	fprintf(stderr, "Overall time: %f seconds\n",
			etime - stime);

#if 0
  for(int i = 0; i < _numVectors; ++i){
    for(int j = i; j < _numVectors; ++j){
      printf("%f\n", _distanceCorr[i *_numVectors + j]);
    }
  }
#endif
}

template<typename FloatType>
void DistanceCorr<FloatType>::runMultiThreaded() {
	double stime, etime;
	const int tileDim = (_numVectors + DT_MT_TILE_DIM - 1) / DT_MT_TILE_DIM;
	const ssize_t numTiles = (ssize_t) (tileDim + 1) * tileDim / 2;
	ssize_t totalNumPairs = 0;
	const FloatType avg = 1.0 / (FloatType)_vectorSize;

#ifdef VERBOSE
  fprintf(stderr, "execute function %s\n", __FUNCTION__);
#endif

	/*record system time*/
	stime = getSysTime();

	/*allocate space*/
	_distanceCorr = (FloatType*) mm_malloc(
			(ssize_t) _numVectors * _numVectors * sizeof(FloatType), 64);
	if (!_distanceCorr) {
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
		int row, col;
		int loRowRange, hiRowRange;
		int loColRange, hiColRange;
		int startColPerRow, endColPerRow;
		int rowStart, rowEnd, colStart, colEnd;
		ssize_t numPairsProcessed = 0;
		FloatType* __restrict__ vecX;
		FloatType* __restrict__ vecY;
		const int tid = omp_get_thread_num();
		const int nthreads = omp_get_num_threads();

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

  	FloatType* meanX = (FloatType*)mm_malloc(_vectorSize * sizeof(FloatType), 64);
  	FloatType* meanY = (FloatType*)mm_malloc(_vectorSize * sizeof(FloatType), 64);
		for (int tileRow = loRowRange; tileRow <= hiRowRange; tileRow++) {
			/*compute the effective range per row*/
			startColPerRow = (tileRow == loRowRange) ? loColRange : tileRow;
			endColPerRow = (tileRow == hiRowRange) ? hiColRange : tileDim - 1;

			rowStart = tileRow * DT_MT_TILE_DIM;
			rowEnd = min(_numVectors, rowStart + DT_MT_TILE_DIM);
			for (int tileCol = startColPerRow; tileCol <= endColPerRow;
					++tileCol) {
				colStart = tileCol * DT_MT_TILE_DIM;
				colEnd = min(_numVectors, colStart + DT_MT_TILE_DIM);
				/*compute the tile*/
				vecX = _vectors + rowStart * _vectorSizeAligned;
				for (row = rowStart; row < rowEnd; ++row, vecX +=
						_vectorSizeAligned) {
					vecY = _vectors + colStart * _vectorSizeAligned;
					for (col = colStart; col < colEnd; ++col, vecY +=
							_vectorSizeAligned) {

						if(row <= col){
							++numPairsProcessed;	/*statistics*/
							FloatType corr = _computeDistCorr(meanX, meanY, vecX, vecY, _vectorSize);
							_distanceCorr[(ssize_t)row * _numVectors + col] = corr;
							_distanceCorr[(ssize_t)col * _numVectors + row] = corr;
						}
					}
				}
			}
		}
		mm_free(meanX);
		mm_free(meanY);

		/*reduction*/
		totalNumPairs += numPairsProcessed;
	} /*#pragma omp parallel*/

	/*recored the system time*/
	etime = getSysTime();
	fprintf(stderr, "Overall time (%ld pairs): %f seconds\n", totalNumPairs,
			etime - stime);

#if 0
  for(int i = 0; i < _numVectors; ++i){
    for(int j = i; j < _numVectors; ++j){
      printf("%f\n", _distanceCorr[i *_numVectors + j]);
    }
  }
#endif

}

#ifdef WITH_PHI
template<typename FloatType>
void DistanceCorr<FloatType>::runSingleXeonPhi() {
	const int tileDim = (_numVectors + DT_PHI_TILE_DIM - 1) / DT_PHI_TILE_DIM;
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
		if (_numMICThreads < DT_PHI_TILE_DIM) {
			_numMICThreads = omp_get_num_procs();
		}
		_numMICThreads = _numMICThreads / DT_PHI_TILE_DIM * DT_PHI_TILE_DIM;
		omp_set_num_threads (_numMICThreads);
	}
	fprintf(stderr, "number of threads: %d\n", _numMICThreads);

	/*allocate intermediate memory*/
	ssize_t maxPrValuesSize = DT_PHI_BUFFER_SIZE / sizeof(FloatType); /*a total of 512 MB*/
	ssize_t alignment = _numMICThreads * DT_PHI_TILE_SIZE;
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
	_distanceCorr = (FloatType*)mm_malloc((ssize_t)_numVectors * _numVectors * sizeof(FloatType), 64);
	if(!_distanceCorr) {
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
		in(vectors: length(_numVectors * _vectorSizeAligned) DT_MIC_ALLOC) \
		nocopy(inValues: length(maxPrValuesSize) DT_MIC_ALLOC) \
		nocopy(outValues: length(maxPrValuesSize) DT_MIC_ALLOC)
	{
	}

	fprintf(stderr, "Start computing\n");
	/*record system time*/
	stime = getSysTime();

	/*the first round*/
	maxNumTilesPerPass = maxPrValuesSize / DT_PHI_TILE_SIZE;
	indexRangeStart = 0;
	indexRangeClose = min(numTiles, maxNumTilesPerPass);
#pragma offload target(mic:_micIndex) \
		nocopy(vectors: DT_MIC_REUSE)	\
		nocopy(inValues: DT_MIC_REUSE)	\
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
    nocopy(inValues: DT_MIC_REUSE) \
    nocopy(outValues: DT_MIC_REUSE) \
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
		nocopy(vectors : DT_MIC_REUSE)	\
		nocopy(inValues: DT_MIC_REUSE)	\
		in(indexRangeStart)	\
		in(indexRangeClose)	\
		signal(&signalVar)
		{
			_runSingleXeonPhiCore(vectors, inValues, indexRangeStart,
					indexRangeClose);
		}

		/*device-to-host data transfer and save*/
		numElems = (indexRangeClosePrev - indexRangeStartPrev) * DT_PHI_TILE_SIZE;
		if(numElems > maxPrValuesSize){
			fprintf(stderr, "Error: %ld > %ld\n", numElems, maxPrValuesSize);
		}
#pragma offload_transfer target(mic:_micIndex) out(outValues[0:numElems]:into(hostPrValues[0:numElems]) DT_MIC_REUSE)

    /*save the data to the correlation matrix*/
		tileOffset = 0;
    for(ssize_t idx = indexRangeStartPrev; idx < indexRangeClosePrev; ++idx, tileOffset += DT_PHI_TILE_SIZE){
      /*get the row and column index of the tile*/
      getTileCoordinate(idx, tileDim, tileRowStart, tileColStart);

      /*compute the row range*/
      tileRowStart *= DT_PHI_TILE_DIM;
      tileColStart *= DT_PHI_TILE_DIM;
      tileRowEnd = min(_numVectors, tileRowStart + DT_PHI_TILE_DIM);
      tileColEnd = min(_numVectors, tileColStart + DT_PHI_TILE_DIM);

      /*compute each tile*/
			ssize_t rowOffset = tileOffset;
      for(int row = tileRowStart; row < tileRowEnd; ++row, rowOffset += DT_PHI_TILE_DIM){
        for(int col = tileColStart, colOffset = rowOffset; col < tileColEnd; ++col, ++colOffset){
					if(row <= col){
            _distanceCorr[(ssize_t)row * _numVectors + col] = hostPrValues[colOffset];
            _distanceCorr[(ssize_t)col * _numVectors + row] = hostPrValues[colOffset];
          }
        }
      }
    }
	}
	/*transfer the remaining data*/
	numElems = (indexRangeClosePrev - indexRangeStartPrev) * DT_PHI_TILE_SIZE;
	if(numElems > 0) {
		if(numElems > maxPrValuesSize){
			fprintf(stderr, "Error: %ld > %ld\n", numElems, maxPrValuesSize);
		}
#pragma offload_transfer target(mic:_micIndex) out(outValues[0:numElems]:into(hostPrValues[0:numElems]) DT_MIC_REUSE)
    /*save the data to the correlation matrix*/
		tileOffset = 0;
    for(ssize_t idx = indexRangeStartPrev; idx < indexRangeClosePrev; ++idx, tileOffset += DT_PHI_TILE_SIZE){
      /*get the row and column index of the tile*/
      getTileCoordinate(idx, tileDim, tileRowStart, tileColStart);

      /*compute the row range*/
      tileRowStart *= DT_PHI_TILE_DIM;
      tileColStart *= DT_PHI_TILE_DIM;
      tileRowEnd = min(_numVectors, tileRowStart + DT_PHI_TILE_DIM);
      tileColEnd = min(_numVectors, tileColStart + DT_PHI_TILE_DIM);

      /*compute each tile*/
			ssize_t rowOffset = tileOffset;
      for(int row = tileRowStart; row < tileRowEnd; ++row, rowOffset += DT_PHI_TILE_DIM){
        for(int col = tileColStart, colOffset = rowOffset; col < tileColEnd; ++col, ++colOffset){
					if(row <= col){
            _distanceCorr[(ssize_t)row * _numVectors + col] = hostPrValues[colOffset];
            _distanceCorr[(ssize_t)col * _numVectors + row] = hostPrValues[colOffset];
          }
        }
      }
    }
	}

	/*release memory on the Xeon Phi*/
#pragma offload_transfer target(mic: _micIndex) \
		nocopy(vectors : DT_MIC_FREE) \
		nocopy(inValues: DT_MIC_FREE) \
		nocopy(outValues: DT_MIC_FREE)

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
      printf("%f\n", _distanceCorr[i *_numVectors + j]);
    }
  }
#endif
}

template<typename FloatType>
__attribute__((target(mic))) void DistanceCorr<FloatType>::_runSingleXeonPhiCore(
		FloatType* __restrict__ vectors, FloatType* __restrict__ prValues,
		const ssize_t indexRangeStart, const ssize_t indexRangeClose) {

#ifdef __MIC__
	const int tileDim = (_numVectors + DT_PHI_TILE_DIM -1) / DT_PHI_TILE_DIM;

#pragma omp parallel
	{
		const int numGroups = omp_get_num_threads() / DT_PHI_TILE_DIM;
		const int tid = omp_get_thread_num() % DT_PHI_TILE_DIM;
		const int gid = omp_get_thread_num() / DT_PHI_TILE_DIM;
		ssize_t offset, idx;
		int row, col, tileRowStart, tileRowEnd, tileColStart, tileColEnd;
		FloatType corr;
		FloatType* __restrict__ vecX;
		FloatType* __restrict__ vecY;
		FloatType* meanX = (FloatType*)mm_malloc(_vectorSize * sizeof(FloatType), 64);
  	FloatType* meanY = (FloatType*)mm_malloc(_vectorSize * sizeof(FloatType), 64);

		/*one group processes on tile*/
		for (idx = indexRangeStart + gid; idx < indexRangeClose; idx += numGroups) {

			/*get the row and column index of the tile*/
			getTileCoordinate(idx, tileDim, tileRowStart, tileColStart);

			/*compute row range*/
			tileRowStart *= DT_PHI_TILE_DIM;
			tileRowEnd = tileRowStart + DT_PHI_TILE_DIM;
			if(tileRowEnd > _numVectors) {
				tileRowEnd = _numVectors;
			}

			/*each work thread computes its own column index*/
			offset = (idx - indexRangeStart) * DT_PHI_TILE_SIZE + tid;
			col = tileColStart * DT_PHI_TILE_DIM + tid;
			if(col < _numVectors) {
				vecY = vectors + col * _vectorSizeAligned;
				vecX = vectors + tileRowStart * _vectorSizeAligned;
				for(row = tileRowStart; row < tileRowEnd; ++row, vecX += _vectorSizeAligned, offset += DT_PHI_TILE_DIM) {
					/*compute correlation*/
					if(row <= col){
						prValues[offset] = _computeDistCorr(meanX, meanY, vecX, vecY, _vectorSize);
					}
				}
			}
		}
		mm_free(meanX);
		mm_free(meanY);
	}
#endif	/*__MIC__*/
}
#endif	/*with phi*/


#ifdef WITH_MPI
template<typename FloatType>
void DistanceCorr<FloatType>::runMPICPU() {
	double stime, etime;
	const int tileDim = (_numVectors + DT_MPI_TILE_DIM -1) / DT_MPI_TILE_DIM;
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
	FloatType* __restrict__ vecX;
	FloatType* __restrict__ vecY;
	const FloatType avg = 1.0 / (FloatType)_vectorSize;

	/*allocate buffer*/
	ssize_t chunkSize = (numTiles + _numProcs - 1) / _numProcs;
	_distanceCorr = (FloatType*)mm_malloc( chunkSize * DT_MPI_TILE_DIM * DT_MPI_TILE_DIM * sizeof(FloatType), 64);
	if(!_distanceCorr) {
		fprintf(stderr, "Memory allocation failed\n");
		exit(-1);
	}
	_tiledPrMatrix = 1;

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

	/*initialize distance matrix*/
	FloatType* meanX = (FloatType*)mm_malloc(_vectorSize * sizeof(FloatType), 64);
	FloatType* meanY = (FloatType*)mm_malloc(_vectorSize * sizeof(FloatType), 64);
	for (int tileRow = loRowRange; tileRow <= hiRowRange; tileRow++) {
		rowStart = tileRow * DT_MPI_TILE_DIM;
		rowEnd = min(_numVectors, rowStart + DT_MPI_TILE_DIM);
		startColPerRow = (tileRow == loRowRange) ? loColRange : tileRow;
		endColPerRow = (tileRow == hiRowRange) ? hiColRange : tileDim - 1;
		for (int tileCol = startColPerRow; tileCol <= endColPerRow;
				++tileCol) {
			colStart = tileCol * DT_MPI_TILE_DIM;
			colEnd = min(_numVectors, colStart + DT_MPI_TILE_DIM);
			/*compute each tile*/
			vecX = _vectors + rowStart * _vectorSizeAligned;
			for (row = rowStart, rowOffset = offset; row < rowEnd; row++, vecX +=
					_vectorSizeAligned, rowOffset += DT_MPI_TILE_DIM) {
				vecY = _vectors + colStart * _vectorSizeAligned;
				for (col = colStart, colOffset = rowOffset; col < colEnd; ++col, vecY +=
						_vectorSizeAligned, colOffset++) {

					if(row <= col){
						numPairsProcessed++;	/*statistics*/
						_distanceCorr[colOffset] = _computeDistCorr(meanX, meanY, vecX, vecY, _vectorSize);
					}
				}
			}
			/*move to the next tile*/
			offset += DT_MPI_TILE_SIZE;
		}
	}
	mm_free(meanX);
	mm_free(meanY);

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
void DistanceCorr<FloatType>::runMPIXeonPhi() {
	const int tileDim = (_numVectors + DT_PHI_TILE_DIM - 1) / DT_PHI_TILE_DIM;
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
		if(_numMICThreads < DT_PHI_TILE_DIM) {
			_numMICThreads = omp_get_num_procs();
		}
		_numMICThreads = _numMICThreads / DT_PHI_TILE_DIM * DT_PHI_TILE_DIM;
		omp_set_num_threads(_numMICThreads);
	}

	/*allocate memory*/
	ssize_t maxPrValuesSize = DT_PHI_BUFFER_SIZE / sizeof(FloatType); /*a total of 512 MB*/
	ssize_t alignment = _numMICThreads * DT_PHI_TILE_SIZE;
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
	_distanceCorr = (FloatType*)mm_malloc( chunkSize * DT_PHI_TILE_SIZE * sizeof(FloatType), 64);
	if(!_distanceCorr) {
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
	in(vectors: length(_numVectors * _vectorSizeAligned) DT_MIC_ALLOC) \
	nocopy(inValues: length(maxPrValuesSize) DT_MIC_ALLOC) \
	nocopy(outValues: length(maxPrValuesSize) DT_MIC_ALLOC)

	/*record system time*/
	if(_rank == 0) {
		fprintf(stderr, "Start computing\n");
	}
	MPI_Barrier(MPI_COMM_WORLD);
	stime = getSysTime();

	/*the first round*/
	maxNumTilesPerPass = maxPrValuesSize / DT_PHI_TILE_SIZE;
	indexRangeStart = min(numTiles, _rank * chunkSize);
	indexRangeEnd = min(numTiles, (_rank + 1) * chunkSize);
	indexRangeClose = min(indexRangeEnd, indexRangeStart + maxNumTilesPerPass);
#pragma offload target(mic:_micIndex) \
	nocopy(vectors: DT_MIC_REUSE)	\
	nocopy(inValues: DT_MIC_REUSE)	\
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
    nocopy(inValues: DT_MIC_REUSE) \
    nocopy(outValues: DT_MIC_REUSE) \
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
	nocopy(vectors : DT_MIC_REUSE)	\
	nocopy(inValues: DT_MIC_REUSE)	\
	in(indexRangeStart)	\
	in(indexRangeClose)	\
	signal(&signalVar)
		{
			_runSingleXeonPhiCore(vectors, inValues, indexRangeStart, indexRangeClose);
		}
		numElems = (indexRangeClosePrev - indexRangeStartPrev) * DT_PHI_TILE_SIZE;
#pragma offload_transfer target(mic:_micIndex) out(outValues[0:numElems]:into(_distanceCorr[offset:numElems]) DT_MIC_REUSE)
		offset += numElems;
	}
	/*process the data in the last round*/
	numElems = (indexRangeClosePrev - indexRangeStartPrev) * DT_PHI_TILE_SIZE;
	if(numElems > 0) {
#pragma offload_transfer target(mic:_micIndex) out(outValues[0:numElems]:into(_distanceCorr[offset:numElems]) DT_MIC_REUSE)
	}

	/*release memory on the Xeon Phi*/
#pragma offload_transfer target(mic: _micIndex) \
	nocopy(vectors : DT_MIC_FREE) \
	nocopy(inValues: DT_MIC_FREE) \
	nocopy(outValues: DT_MIC_FREE)

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
void DistanceCorr<FloatType>::transpose()
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
