# distutils: language=c++
# -*- coding: latin-1 -*-
############################################################################
# Gnocis
# Bjørn Bredesen, 2018-2019
# bjorn.bredesen@ii.uib.no
############################################################################

from __future__ import division
from libcpp cimport bool
from .sequences cimport *


############################################################################
# Sequence features

cdef class feature:
	
	cpdef double get(self, sequence seq, bool cache=*)

# Represents feature sets.
cdef class features:
	
	cdef public str name
	cdef public list features
	cdef public list values
	cdef public int nvalues
	
	cpdef list getAll(self, sequence seq)

cdef class scaledFeature(feature):
	
	cdef public float vScale, vSub
	cdef public feature feature
	
	cpdef double get(self, sequence seq, bool cache=*)

cdef class featureScaler(features):
	
	cpdef list getAll(self, sequence seq)


############################################################################
# Motif clustering

# Sequence model feature for motif occurrence frequency.
cdef class featureMotifOccurrenceFrequency(feature):
	
	cdef public object m
	
	cpdef double get(self, sequence seq, bool cache=*)

# Sequence model feature for PREdictor-style motif pair occurrence frequency.
cdef class featurePREdictorMotifPairOccurrenceFrequency(feature):
	
	cdef public object mA, mB
	cdef public int distCut
	
	cpdef double get(self, sequence seq, bool cache=*)


############################################################################
# k-mer spectrum

# k-mer for k-spectrum feature set
cdef class kSpectrumFeature(feature):
	
	cdef public kSpectrum parent
	cdef public str kmer
	cdef public int index
	cdef public object cachedSequence
	cdef public double cachedValue
	
	cpdef double get(self, sequence seq, bool cache=*)

# Extracts k-mer spectra from sequences
cdef class kSpectrum:
	
	cdef public int nspectrum, nFeatures, bitmask
	cdef public str cacheName
	cdef public dict kmerByIndex
	cdef public list features
	
	cdef double extract(self, sequence seq, int index, cache=*)

############################################################################
# k-mer mismatch spectrum

# k-mer for k-spectrum feature set
cdef class kSpectrumMMFeature(feature):
	
	cdef public kSpectrumMM parent
	cdef public str kmer
	cdef public int index
	cdef public object cachedSequence
	cdef public double cachedValue
	
	cpdef double get(self, sequence seq, bool cache=*)

# Extracts k-mer spectra from sequences
cdef class kSpectrumMM:
	
	cdef public int nspectrum, nFeatures, bitmask
	cdef public str cacheName
	cdef public dict kmerByIndex
	cdef public list features

	cdef double extract(self, sequence seq, int index, cache=*)


############################################################################
# Positional, double-stranded k-mer spectrum

# k-mer for k-spectrum feature set
cdef class kSpectrumFeaturePDS(feature):
	
	cdef public kSpectrumPDS parent
	cdef public str kmer
	cdef public int index
	cdef public object cachedSequence
	cdef public double cachedValue
	cdef public bool section

	cpdef double get(self, sequence seq, bool cache=*)

# Extracts k-mer spectra from sequences
cdef class kSpectrumPDS:
	
	cdef public int nspectrum, nkmers, nFeatures, bitmask
	cdef public str cacheName
	cdef public dict kmerByIndex
	cdef public list features
	
	cdef double extract(self, sequence seq, int index, cache=*)


