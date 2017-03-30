/*
 *  Transformation.hpp
 * 
 *  Created on: Nov 30, 2016
 *  Author: Liu, Yongchao
 *  Affiliation: School of Computational Science & Engineering
 *  						Georgia Institute of Technology, Atlanta, GA 30332
 *  URL: www.liuyc.org
 */

#ifndef TRANSFORMATION_HPP_
#define TRANSFORMATION_HPP_

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
#include <limits>
using namespace std;
#pragma once

template <typename FloatType, typename IntType=int>
class FisherTransform
{
public:
#ifdef WITH_PHI
  __attribute__((target(mic)))
#endif
	void operator()(FloatType* __restrict__ matrix, const IntType width, const IntType widthAligned, const IntType height);

#ifdef WITH_PHI
  __attribute__((target(mic)))
#endif
	void operator()(const FloatType* __restrict__ inMatrix, FloatType* __restrict__ outMatrix, const IntType width, const IntType widthAligned, const IntType height);
};

template<typename FloatType, typename IntType>
#ifdef WITH_PHI
  __attribute__((target(mic)))
#endif
void FisherTransform<FloatType, IntType>::operator()(FloatType* __restrict__ matrix, const IntType width, const IntType widthAligned, const IntType height)
{
	FloatType v, *mat;

	mat = matrix;
	for(IntType row = 0; row < height; ++row){
#pragma vector aligned
#pragma simd
		for(IntType col = 0;  col < width; ++col){
			v = mat[col];
			mat[col] = 0.5 * log((1 + v) / (1 - v));
		}

		/*move to the next row*/
		mat += widthAligned;
	}
}

template<typename FloatType, typename IntType>
#ifdef WITH_PHI
  __attribute__((target(mic)))
#endif
void FisherTransform<FloatType, IntType>::operator()(const FloatType* __restrict__ inMatrix, FloatType* __restrict__ outMatrix, const IntType width, const IntType widthAligned, const IntType height)
{
	FloatType v, *imat, *omat;

	imat = inMatrix;
	omat = outMatrix;
  for(IntType row = 0; row < height; ++row){
#pragma vector aligned
#pragma simd
		for(IntType col = 0; col < width; ++col){
			v = imat[col];
			omat[col] = 0.5 * log((1 + v) / (1 - v));
		}

		/*move to the next row*/
		imat += widthAligned;
		omat += widthAligned;
	}
}

#endif

