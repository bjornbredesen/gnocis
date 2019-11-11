# distutils: language=c++
# -*- coding: latin-1 -*-
############################################################################
# Gnocis
# Bjørn Bredesen, 2018-2019
# bjorn.bredesen@ii.uib.no
############################################################################

from __future__ import division
from libcpp cimport bool
from .motifs cimport *
from .sequences cimport *
from .common import kSpectrumIndex2NT, kSpectrumNT2Index


############################################################################
# Sequence features

# General feature base class.
cdef class feature:
	"""
	The `feature` class is a base class for sequence features.
	"""
	
	def __init__(self):
		pass
	
	cpdef double get(self, sequence seq, bool cache=True):
		"""
		Extracts the feature value from a sequence.
		
		:param seq: The sequence to extract the feature value from.
		:param cache: Whether or not to cache the feature value.
		
		:type seq: sequence
		:type cache: bool
		
		:return: Feature value
		:rtype: float
		"""
		pass

# Represents feature sets.
cdef class features:
	"""
	The `features` class represents a set of features. 
	
	:param name: Name of the feature set
	:param f: Features to include. Defaults to [].
	
	:type name: str
	:type f: list, optional
	
	After constructing a `features` object, for a set of features `FS`, `len(FS)` gives the number of features, `FS[i]` gives feature with index `i`, and `[ x for x in FS ]` gives the list of `feature` objects. Two `features` objects `A` and `B` can be added together with `A+B`, yielding a concatenated set of features.
	"""
	
	def __init__(self, name, f):
		self.name = name
		self.features = f
		self.values = []
		self.nvalues = 0
	
	def __iter__(self):
		return self.features.__iter__()
	
	def __getitem__(self, i):
		return self.features[i]
	
	def __len__(self):
		return len(self.features)
	
	def __str__(self):
		return 'Feature set<%s>'%(self.name)
	
	def __repr__(self):
		return self.__str__()
	
	def __add__(self, other):
		return features('%s + %s'%(self.name, other.name), self.features + other.features)
	
	cpdef list getAll(self, sequence seq):
		"""
		Extracts the feature values from a sequence.
		
		:param seq: The sequence to extract the feature value from.
		
		:type seq: sequence
		
		:return: Feature vector
		:rtype: list
		"""
		cdef list ret
		cdef feature f
		ret = []
		for f in self.features:
			ret.append( f.get(seq) )
		return ret
		#return [ f.get(seq) for f in self.features ]
	
	@staticmethod
	def getMotifSpectrum(motifs):
		"""
		Constructs a feature set for the motif occurrence spectrum, given a set of motifs.
		
		:param motifs: The motifs to use for the feature space.
		
		:type motifs: motifs
		
		:return: Feature set
		:rtype: features
		"""
		return features('Motif spectrum: %s'%str(motifs), [ featureMotifOccurrenceFrequency(m) for m in motifs ])
	
	@staticmethod
	def getPREdictorMotifPairSpectrum(motifs, distCut):
		"""
		Constructs a feature set for the PREdictor motif pair occurrence spectrum, given a set of motifs.
		
		:param motifs: The motifs to use for the feature space.
		
		:type motifs: motifs
		
		:return: Feature set
		:rtype: features
		"""
		pairs = []
		for iA,mA in enumerate(motifs):
			for mB in motifs[iA:]:
				pairs.append( featurePREdictorMotifPairOccurrenceFrequency(mA, mB, distCut) )
		return features('PREdictor motif pair spectrum: %s within %d bp'%(str(motifs), distCut), pairs)
	
	@staticmethod
	def getKSpectrum(k):
		"""
		Constructs the k-mer spectrum kernel, the occurrence frequencies of all motifs of length k, with no ambiguous positions.
		
		:param k: k-mer length.
		
		:type k: int
		
		:return: Feature set
		:rtype: features
		"""
		return features('%d-spectrum'%(k), kSpectrum(k).features)
	
	@staticmethod
	def getKSpectrumMM(k):
		"""
		Constructs the k-mer spectrum mismatch kernel, the occurrence frequencies of all motifs of length k, with one allowed mismatch in an arbitrary position.
		
		:param k: k-mer length.
		
		:type k: int
		
		:return: Feature set
		:rtype: features
		"""
		return features('%d-spectrumMM'%(k), kSpectrumMM(k).features)
	
	@staticmethod
	def getKSpectrumMMD(k):
		"""
		Constructs the k-mer spectrum mismatch kernel with duplicates, the occurrence frequencies of all motifs of length k, with one allowed mismatch in an arbitrary position, and one duplicate registry of the core motif at every position to give higher preference.
		
		:param k: k-mer length.
		
		:type k: int
		
		:return: Feature set
		:rtype: features
		"""
		return features('%d-spectrumMMD'%(k), kSpectrumMMD(k).features)
	
	@staticmethod
	def getKSpectrumPDS(k):
		"""
		Constructs the k-mer spectrum positional double-stranded kernel, the occurrence frequencies of all motifs of length k, with position represented by grading between sequence ends.
		
		:param k: k-mer length.
		
		:type k: int
		
		:return: Feature set
		:rtype: features
		"""
		return features('%d-spectrum (PDS)'%(k), kSpectrumPDS(k).features)

cdef class scaledFeature(feature):
	"""
	The `scaledFeature` class scales and shifts an input feature. 
	
	:param _feature: The feature to scale.
	:param vScale: Value to scale by.
	:param vSub: Shifting.
	
	:type _feature: feature
	:type vScale: float
	:type vSub: float
	
	After constructing a `biomarkers` object, for a set of biomarkers `BM`, `len(BM)` gives the number of markers, `BM['FACTOR_NAME']` gives the merged set of regions for factor `FACTOR_NAME`, and `[ x for x in BM ]` gives the list of merged regions per biomarker contained in `BM`.
	"""
	
	def __init__(self, _feature, vScale, vSub):
		self.vScale = vScale
		self.vSub = vSub
		self.feature = _feature
	
	def __str__(self):
		return 'Feature scaler<%s>'%(str(self.feature))
	
	def __repr__(self): return self.__str__()
	
	cpdef double get(self, sequence seq, bool cache=True):
		return self.feature.get(seq, cache) * self.vScale - self.vSub

# Scales features to the [-1, 1] interval.
cdef class featureScaler(features):
	"""
	The `featureScaler` class scales an input feature set, based on a binary training set. 
	
	:param _features: The feature set to scale.
	:param positives: Positive training sequences.
	:param negatives: Negative training sequences.
	
	:type _features: features
	:type positives: sequences
	:type negatives: sequences
	"""
	
	def __init__(self, _features, positives, negatives):
		cdef feature f
		cdef list fv, fs
		cdef float vMin, vMax, vScale, vSub, rng
		cdef sequence seq
		cdef list fvs, cfvs
		cdef int i
		cdef float v
		#
		fvs = [ [] for f in _features ]
		for seq in positives + negatives:
			cfvs = _features.getAll(seq)
			for i, v in enumerate(cfvs):
				fvs[i].append(v)
		#
		fs = []
		for f, fv in zip(_features, fvs):
			vMin = min(fv)
			vMax = max(fv)
			rng = vMax - vMin
			if rng == 0.0:
				vScale = 0.0
			else:
				vScale = 2.0 / rng
			vSub = vMin * vScale + 1.0
			fs.append( scaledFeature( f, vScale, vSub ) )
		#
		features.__init__(self, _features.name, fs)
	
	def __str__(self):
		return 'Scaled feature set<%s>'%(self.name)
	
	def __repr__(self):
		return self.__str__()
	
	cpdef list getAll(self, sequence seq):
		cdef list ret
		cdef scaledFeature f
		ret = [
			f.feature.get(seq) * f.vScale - f.vSub
			for f in self.features
		]
		return ret


############################################################################
# Motif clustering

# Sequence model feature for motif occurrence frequency.
cdef class featureMotifOccurrenceFrequency(feature):
	"""
	Occurrence frequency feature for a motif.
	
	:param _motif: The motif that the feature is for.
	
	:type _motif: motif
	"""
	
	def __init__(self, _motif):
		self.m = _motif
	
	def __str__(self):
		return 'Feature<Motif occurrence frequency: %s>'%(self.m.name)
	
	def __repr__(self): return self.__str__()
	
	cpdef double get(self, sequence seq, bool cache=True):
		return len(self.m.findOccurrences(seq)) * 1000.0 / len(seq.seq)

# Sequence model feature for PREdictor-style motif pair occurrence frequency.
cdef class featurePREdictorMotifPairOccurrenceFrequency(feature):
	"""
	PREdictor pair occurrence frequency feature for two motifs and a distance cutoff.
	
	:param motifA: The first motif.
	:param motifB: The second motif.
	:param distanceCutoff: Pairing cutoff distance. The cutoff is for nucleotides inbetween occurrences of `motifA` and `motifB`.
	
	:type motifA: motif
	:type motifB: motif
	:type distanceCutoff: int
	"""
	
	def __init__(self, motifA, motifB, distanceCutoff):
		self.mA, self.mB, self.distCut = motifA, motifB, distanceCutoff
	
	def __str__(self):
		return 'Feature<PREdictor motif pair occurrence frequency: %s, %s (within %d nt)>'%(self.mA.name, self.mB.name, self.distCut)
	
	def __repr__(self): return self.__str__()
	
	cpdef double get(self, sequence seq, bool cache=True):
		cdef int nPairOcc, firstRelevantiB
		cdef list alloA, alloB
		cdef motifOccurrence oA, oB
		cdef int d
		nPairOcc = 0
		alloA = self.mA.findOccurrences(seq)
		alloB = self.mB.findOccurrences(seq)
		firstRelevantiB = 0
		for oA in alloA:
			for oB in alloB[firstRelevantiB:]:
				if self.mA == self.mB and oA.start <= oB.start: break
				if oB.end < oA.start - self.distCut: firstRelevantiB += 1; continue
				if oB.start > oA.end + self.distCut: break
				if oB.start-oA.end<oA.start-oB.end:
					d = oA.start-oB.end
				else:
					d = oB.start-oA.end
				if d >= 0 and d <= self.distCut:
					nPairOcc += 1
		return nPairOcc * 1000.0 / len(seq.seq)


############################################################################
# k-mer spectrum

# k-mer for k-spectrum feature set
cdef class kSpectrumFeature(feature):
	
	def __init__(self, parent, kmer, index):
		self.parent, self.kmer, self.index = parent, kmer, index
		self.cachedSequence = None
		self.cachedValue = 0.0
	
	def __str__(self):
		return 'Feature<k-mer occurrence frequency: %s>'%(self.kmer)
	
	def __repr__(self): return self.__str__()
	
	cpdef double get(self, sequence seq, bool cache=True):
		if cache:
			if self.cachedSequence == seq:
				return self.cachedValue
		return self.parent.extract(seq, self.index)

# Extracts k-mer spectra from sequences
cdef class kSpectrum:
	"""
	Feature set that extracts the occurrence frequencies of all unambiguous motifs of length k. Extraction of the entire spectrum is optimized by taking overlaps of motifs into account.
	
	:param nspectrum: Length of motifs, k, to use.
	
	:type nspectrum: int
	"""
	
	def __init__(self, nspectrum):
		self.nspectrum = nspectrum
		self.nFeatures = (1 << (2*nspectrum))
		self.bitmask = self.nFeatures - 1
		self.kmerByIndex = {}
		for ki in range(self.nFeatures):
			self.kmerByIndex[ki] = ''.join( kSpectrumIndex2NT[(ki >> ((nspectrum - 1 - x)*2)) & 3] for x in range(nspectrum) )
		self.features = [ kSpectrumFeature(self, self.kmerByIndex[ki], ki) for ki in range(self.nFeatures) ]
		self.cacheName = '%d-spectrum'%nspectrum
	
	cdef double extract(self, sequence seq, int index, cache=True):
		cdef bytes bseq
		cdef unsigned char bnt
		cdef int ki, kiRC, nnt, nspectrum, bitmask, nRCShift, slen
		cdef double nAdd, fv
		cdef list nOcc, convtable
		cdef char c
		cdef kSpectrumFeature f
		if cache:
			if self.cacheName in seq.cache.keys():
				for f, fv in zip(self.features, seq.cache[self.cacheName]):
					f.cachedSequence = seq
					f.cachedValue = fv
				return seq.cache[self.cacheName][index]
		nspectrum = self.nspectrum
		bitmask = (1 << (2*nspectrum))-1
		bseq = seq.getBytesIndexed()
		slen = len(seq)
		nRCShift = 2*(nspectrum-1)
		ki, kiRC = (0, 0)
		nAdd = 1000.0 / len(seq.seq)
		nnt = 0
		nOcc = [ 0.0 for _ in range(self.nFeatures) ]
		for i in range(slen):
			bnt = bseq[i]
			if bnt == 0xFF:
				ki = 0
				kiRC = 0
				nnt = 0
				continue
			ki = ( ( ki << 2 ) | bnt ) & bitmask
			kiRC = ( kiRC >> 2 ) | ( (bnt^1) << nRCShift )
			nnt += 1
			if nnt >= nspectrum:
				nOcc[ki] += nAdd
				nOcc[kiRC] += nAdd
		if cache:
			seq.cache[self.cacheName] = nOcc
			for f, fv in zip(self.features, nOcc):
				f.cachedSequence = seq
				f.cachedValue = fv
		return nOcc[index]


############################################################################
# k-mer mismatch spectrum

# k-mer for k-spectrum feature set
cdef class kSpectrumMMFeature(feature):
	
	def __init__(self, parent, kmer, index):
		self.parent, self.kmer, self.index = parent, kmer, index
		self.cachedSequence = None
		self.cachedValue = 0.0
	
	def __str__(self):
		return 'Feature<k-mer MM occurrence frequency: %s>'%(self.kmer)
	
	def __repr__(self): return self.__str__()
	
	cpdef double get(self, sequence seq, bool cache=True):
		if cache:
			if self.cachedSequence == seq:
				return self.cachedValue
		return self.parent.extract(seq, self.index)

# Extracts k-mer spectra from sequences
cdef class kSpectrumMM:
	"""
	Feature set that extracts the occurrence frequencies of all motifs of length k, with one mismatch allowed in an arbitrary position. Extraction of the entire spectrum is optimized by taking overlaps of motifs into account.
	
	:param nspectrum: Length of motifs, k, to use.
	
	:type nspectrum: int
	"""
	
	def __init__(self, nspectrum):
		self.nspectrum = nspectrum
		self.nFeatures = (1 << (2*nspectrum))
		self.bitmask = self.nFeatures - 1
		self.kmerByIndex = {}
		for ki in range(self.nFeatures):
			self.kmerByIndex[ki] = ''.join( kSpectrumIndex2NT[(ki >> ((nspectrum - 1 - x)*2)) & 3] for x in range(nspectrum) )
		self.features = [ kSpectrumMMFeature(self, self.kmerByIndex[ki], ki) for ki in range(self.nFeatures) ]
		self.cacheName = '%d-spectrumMM'%nspectrum
	
	cdef double extract(self, sequence seq, int index, cache=True):
		cdef bytes bseq
		cdef unsigned char bnt
		cdef int ki, kiRC, nnt, nspectrum, bitmask, nRCShift, slen
		cdef double nAdd, fv
		cdef list nOcc, convtable
		cdef char c
		cdef kSpectrumMMFeature f
		cdef int cki, ckiRC, bki, bkiRC, mutNTI, cmut, cmask, cmuts
		if cache:
			if self.cacheName in seq.cache.keys():
				for f, fv in zip(self.features, seq.cache[self.cacheName]):
					f.cachedSequence = seq
					f.cachedValue = fv
				return seq.cache[self.cacheName][index]
		nspectrum = self.nspectrum
		bitmask = (1 << (2*nspectrum))-1
		bseq = seq.getBytesIndexed()
		slen = len(seq)
		nRCShift = 2*(nspectrum-1)
		ki, kiRC = (0, 0)
		nAdd = 1000.0 / len(seq.seq)
		nnt = 0
		nOcc = [ 0.0 for _ in range(self.nFeatures) ]
		for i in range(slen):
			bnt = bseq[i]
			if bnt == 0xFF:
				ki = 0
				kiRC = 0
				nnt = 0
				continue
			ki = ( ( ki << 2 ) | bnt ) & bitmask
			kiRC = ( kiRC >> 2 ) | ( (bnt^1) << nRCShift )
			nnt += 1
			if nnt >= nspectrum:
				nOcc[ki] += nAdd
				nOcc[kiRC] += nAdd
				for mutNTI in range(nspectrum):
					# Mutate nucleotide mutNTI
					cmask = 0x7FFFFFFF ^ ( (0x3) << (mutNTI*2) )
					bki = ki & cmask
					bkiRC = kiRC & cmask
					for cmut in range(4):
						# Mutate with the cmut nucleotide
						cmuts = cmut << (mutNTI*2)
						cki = bki | cmuts
						ckiRC = bkiRC | cmuts
						if cki != ki:
							nOcc[cki] += nAdd
						if ckiRC != kiRC:
							nOcc[ckiRC] += nAdd
		if cache:
			seq.cache[self.cacheName] = nOcc
			for f, fv in zip(self.features, nOcc):
				f.cachedSequence = seq
				f.cachedValue = fv
		return nOcc[index]


############################################################################
# k-mer mismatch spectrum (duplicate)
# For each mismatch position, duplicate matches are registered, giving a
# higher weighting for the core motif

# k-mer for k-spectrum feature set
cdef class kSpectrumMMDFeature(feature):
	
	def __init__(self, parent, kmer, index):
		self.parent, self.kmer, self.index = parent, kmer, index
		self.cachedSequence = None
		self.cachedValue = 0.0
	
	def __str__(self):
		return 'Feature<k-mer MM occurrence frequency: %s>'%(self.kmer)
	
	def __repr__(self): return self.__str__()
	
	cpdef double get(self, sequence seq, bool cache=True):
		if cache:
			if self.cachedSequence == seq:
				return self.cachedValue
		return self.parent.extract(seq, self.index)

# Extracts k-mer spectra from sequences
cdef class kSpectrumMMD:
	"""
	Feature set that extracts the occurrence frequencies of all motifs of length k, with one mismatch allowed in an arbitrary position. Extraction of the entire spectrum is optimized by taking overlaps of motifs into account.
	
	:param nspectrum: Length of motifs, k, to use.
	
	:type nspectrum: int
	"""
	
	def __init__(self, nspectrum):
		self.nspectrum = nspectrum
		self.nFeatures = (1 << (2*nspectrum))
		self.bitmask = self.nFeatures - 1
		self.kmerByIndex = {}
		for ki in range(self.nFeatures):
			self.kmerByIndex[ki] = ''.join( kSpectrumIndex2NT[(ki >> ((nspectrum - 1 - x)*2)) & 3] for x in range(nspectrum) )
		self.features = [ kSpectrumMMDFeature(self, self.kmerByIndex[ki], ki) for ki in range(self.nFeatures) ]
		self.cacheName = '%d-spectrumMMD'%nspectrum
	
	cdef double extract(self, sequence seq, int index, cache=True):
		cdef bytes bseq
		cdef unsigned char bnt
		cdef int ki, kiRC, nnt, nspectrum, bitmask, nRCShift, slen
		cdef double nAdd, fv
		cdef list nOcc, convtable
		cdef char c
		cdef kSpectrumMMDFeature f
		cdef int cki, ckiRC, bki, bkiRC, mutNTI, cmut, cmask, cmuts
		if cache:
			if self.cacheName in seq.cache.keys():
				for f, fv in zip(self.features, seq.cache[self.cacheName]):
					f.cachedSequence = seq
					f.cachedValue = fv
				return seq.cache[self.cacheName][index]
		nspectrum = self.nspectrum
		bitmask = (1 << (2*nspectrum))-1
		bseq = seq.getBytesIndexed()
		slen = len(seq)
		nRCShift = 2*(nspectrum-1)
		ki, kiRC = (0, 0)
		nAdd = 1000.0 / len(seq.seq)
		nnt = 0
		nOcc = [ 0.0 for _ in range(self.nFeatures) ]
		for i in range(slen):
			bnt = bseq[i]
			if bnt == 0xFF:
				ki = 0
				kiRC = 0
				nnt = 0
				continue
			ki = ( ( ki << 2 ) | bnt ) & bitmask
			kiRC = ( kiRC >> 2 ) | ( (bnt^1) << nRCShift )
			nnt += 1
			if nnt >= nspectrum:
				for mutNTI in range(nspectrum):
					# Mutate nucleotide mutNTI
					cmask = 0x7FFFFFFF ^ ( (0x3) << (mutNTI*2) )
					bki = ki & cmask
					bkiRC = kiRC & cmask
					for cmut in range(4):
						# Mutate with the cmut nucleotide
						cmuts = cmut << (mutNTI*2)
						cki = bki | cmuts
						ckiRC = bkiRC | cmuts
						nOcc[cki] += nAdd
						nOcc[ckiRC] += nAdd
		if cache:
			seq.cache[self.cacheName] = nOcc
			for f, fv in zip(self.features, nOcc):
				f.cachedSequence = seq
				f.cachedValue = fv
		return nOcc[index]


############################################################################
# Positional, double-stranded k-mer spectrum

# k-mer for k-spectrum feature set
cdef class kSpectrumFeaturePDS(feature):
	
	def __init__(self, parent, kmer, index, section):
		self.parent, self.kmer, self.index = parent, kmer, index
		self.cachedSequence = None
		self.cachedValue = 0.0
		self.section = section
	
	def __str__(self):
		return 'Feature<k-mer occurrence frequency (PDS): %s, %s>'%(self.kmer, 'A' if self.section else 'B')
	
	def __repr__(self): return self.__str__()
	
	cpdef double get(self, sequence seq, bool cache=True):
		if cache:
			if self.cachedSequence == seq:
				return self.cachedValue
		return self.parent.extract(seq, self.index)

# Extracts k-mer spectra from sequences
cdef class kSpectrumPDS:
	"""
	Feature set that extracts the occurrence frequencies of all motifs of length k, position and strand represented. To represent position and strandedness, four features are generated per k-mer: one for each combination of strandedness and sequence end for proximity. Graded distance to each end of the sequence is used in order to represent position. Extraction of the entire spectrum is optimized by taking overlaps of motifs into account.
	
	:param nspectrum: Length of motifs, k, to use.
	
	:type nspectrum: int
	"""
	
	def __init__(self, nspectrum):
		self.nspectrum = nspectrum
		self.nkmers = 1 << (2*nspectrum)
		self.nFeatures = self.nkmers * 2
		self.bitmask = self.nkmers - 1
		self.kmerByIndex = {}
		for ki in range(self.nkmers):
			self.kmerByIndex[ki] = ''.join( kSpectrumIndex2NT[(ki >> ((nspectrum - 1 - x)*2)) & 3] for x in range(nspectrum) )
		self.features = [ f for ki in range(self.nkmers) for f in [ kSpectrumFeaturePDS(self, self.kmerByIndex[ki], ki, True), kSpectrumFeaturePDS(self, self.kmerByIndex[ki], ki, False) ] ]
		self.cacheName = '%d-spectrum'%nspectrum
	
	cdef double extract(self, sequence seq, int index, cache=True):
		cdef bytes bseq
		cdef unsigned char bnt
		cdef int ki, kiRC, nnt, nspectrum, bitmask, nRCShift, slen
		cdef double nAdd, fv
		cdef list nOcc, convtable
		cdef char c
		cdef kSpectrumFeaturePDS f
		#
		cdef double degA, degB, degD
		#
		if cache:
			if self.cacheName in seq.cache.keys():
				for f, fv in zip(self.features, seq.cache[self.cacheName]):
					f.cachedSequence = seq
					f.cachedValue = fv
				return seq.cache[self.cacheName][index]
		nspectrum = self.nspectrum
		bitmask = (1 << (2*nspectrum))-1
		bseq = seq.getBytesIndexed()
		slen = len(seq)
		nRCShift = 2*(nspectrum-1)
		ki, kiRC = (0, 0)
		nAdd = 1000.0 / len(seq.seq)
		nnt = 0
		nOcc = [ 0.0 for _ in range(self.nFeatures) ]
		#
		degA = 0.0
		degB = 1.0
		degD = 1.0 / len(seq.seq)
		#
		for i in range(slen):
			bnt = bseq[i]
			if bnt == 0xFF:
				ki = 0
				kiRC = 0
				nnt = 0
				degA += degD
				degB -= degD
				continue
			ki = ( ( ki << 2 ) | bnt ) & bitmask
			kiRC = ( kiRC >> 2 ) | ( (bnt^1) << nRCShift )
			nnt += 1
			if nnt >= nspectrum:
				nOcc[ ( ki   << 1 )     ] += nAdd * degA
				nOcc[ ( ki   << 1 ) + 1 ] += nAdd * degB
				nOcc[ ( kiRC << 1 )     ] += nAdd * degB
				nOcc[ ( kiRC << 1 ) + 1 ] += nAdd * degA
			degA += degD
			degB -= degD
		if cache:
			seq.cache[self.cacheName] = nOcc
			for f, fv in zip(self.features, nOcc):
				f.cachedSequence = seq
				f.cachedValue = fv
		return nOcc[index]


