# distutils: language=c++
# -*- coding: latin-1 -*-
############################################################################
# Gnocis
# Bjørn Bredesen, 2018-2019
# bjorn.bredesen@ii.uib.no
############################################################################

from __future__ import division
from libcpp cimport bool
from .motifs cimport motifOccurrence
from .sequences cimport sequence, sequences
from .common import kSpectrumIndex2NT, kSpectrumNT2Index, KLdiv
from .ioputil import nctable
from .models import sequenceModel
from .sequences import positive, negative
from math import log


############################################################################
# Feature network

#---------------------
# Base class

cdef class featureNetworkNode:
	"""
	The `feature` class is a base class for feature network nodes.
	"""
	
	def __init__(self):
		pass
	
	cpdef list get(self, sequences seq):
		"""
		Extracts the feature values from a sequence.
		
		:param seq: The sequence to extract the feature value from.
		
		:type seq: sequence
		
		:return: Feature values
		:rtype: list
		"""
		pass
	
	def featureNames(self):
		"""
		Returns feature names. Should be implemented by feature network nodes.
		
		:return: Feature names
		:rtype: list
		"""
		pass
	
	def weights(self):
		"""
		Returns a table of weights.
		
		:return: Weight table
		:rtype: nctable
		"""
		return None
	
	def windowSize(self):
		"""
		Returns window size.
		
		:return: Window size
		:rtype: int
		"""
		return None
	
	def windowStep(self):
		"""
		Returns window step size.
		
		:return: Window step size
		:rtype: int
		"""
		return None
	
	def train(self, trainingSet):
		"""
		Recursively trains feature nodes. Should be implemented by feature network nodes.
		
		:param trainingSet: Training set.
		
		:type trainingSet: sequences
		
		:return: Trained copy
		:rtype: featureNetworkNode
		"""
		pass
	
	def __add__(self, other):
		"""
		Concatenates feature nodes.
		
		:param other: Other node.
		
		:type other: featureNetworkNode
		
		:return: Concatenated nodes
		:rtype: featureNetworkNode
		"""
		return FNNCat([
			_i
			for i in [ self, other ]
			for _i in (i if isinstance(i, FNNCat) else [ i ])
		])
	
	def __len__(self):
		return len(self.featureNames())
	
	def __call__(self, seq):
		return self.get(seq)
	
	def __str__(self):
		"""
		String representation. Should be implemented by feature network nodes.
		
		:return: String representation
		:rtype: string
		"""
		return 'Feature network node<>'
	
	def __repr__(self):
		return self.__str__()
	
	def table(self, seqs):
		fvs = [
			self.get(sequences('', [ s ]))
			for s in seqs
		]
		_dict = {
			**{
				'Seq.': [ s.name for s, sfvs in zip(seqs, fvs) for fv in sfvs ],
			},
			**{
				fName: [ fv[fI] for s, sfvs in zip(seqs, fvs) for fv in sfvs ]
				for fI, fName in enumerate(self.featureNames())
			},
		}
		return nctable(
			'Table: ' + self.__str__() + ' applied to ' + seqs.__str__(),
			_dict,
			align = { 'Chromosome': 'l' }
		)
	
	def summary(self, seqs):
		return self.table(seqs).drop('Seq.').summary()
	
	def diffsummary(self, seqA, seqB):
		sumA = self.table(seqA).drop('Seq.').summary()
		sumB = self.table(seqB).drop('Seq.').summary()
		# http://www.allisons.org/ll/MML/KL/Normal/
		return nctable(
			'Table: ' + self.__str__() + ' applied to ' + seqA.__str__() + ' and ' + seqB.__str__(),
			{
				'Feature': sumA['Field'],
				'Mean A': sumA['Mean'],
				'Mean B': sumB['Mean'],
				'Var A': sumA['Var.'],
				'Var B': sumB['Var.'],
				'KLD(A||B)': [
					KLdiv(muA, varA, muB, varB)
					for (muA, varA), (muB, varB) in zip(
						zip(sumA['Mean'], sumA['Var.']),
						zip(sumB['Mean'], sumB['Var.'])
					)
				],
				'KLD(B||A)': [
					KLdiv(muB, varB, muA, varA)
					for (muA, varA), (muB, varB) in zip(
						zip(sumA['Mean'], sumA['Var.']),
						zip(sumB['Mean'], sumB['Var.'])
					)
				],
			}
		)
	
	#-----------------------
	# Short-hands
	
	def model(self, model):
		"""
		Returns a `sequenceModel` instance with this feature network node as features.
		
		:param name: Model name.
		:param windowSize: Window size.
		:param windowStep: Window step size.
		
		:type name: string
		:type windowSize: int
		:type windowStep: int
		
		:return: Model
		:rtype: sequenceModel
		"""
		return FNNModel(features = self, model = model)
	
	def sequenceModel(self, name, windowSize = -1, windowStep = -1):
		"""
		Returns a `sequenceModel` instance with this feature network node as features.
		
		:param name: Model name.
		:param windowSize: Window size.
		:param windowStep: Window step size.
		
		:type name: string
		:type windowSize: int
		:type windowStep: int
		
		:return: Model
		:rtype: sequenceModel
		"""
		return sequenceModelFNN(name = name, features = self, windowSize = windowSize, windowStep = windowStep)
	
	def logOdds(self, labelPositive = positive, labelNegative = negative):
		"""
		Returns a log-odds network node with this node as features.
		
		:param labelPositive: Label of positive training sequences.
		:param labelNegative: Label of negative training sequences.
		
		:type labelPositive: sequences
		:type labelNegative: sequences
		
		:return: Log-odds node
		:rtype: featureNetworkNode
		"""
		return FNNLogOdds(features = self, labelPositive = labelPositive, labelNegative = labelNegative)
	
	def sum(self):
		"""
		Returns a node yielding the sum of the features of this node.
		
		:return: Sum node
		:rtype: featureNetworkNode
		"""
		return FNNSum(self)
	
	def square(self):
		"""
		Returns a node yielding square of features of this node.
		
		:return: Squared node
		:rtype: featureNetworkNode
		"""
		return FNNSquare(self)
	
	def scale(self):
		"""
		Returns a node yielding scaled features of this node.
		
		:return: Scaled node
		:rtype: featureNetworkNode
		"""
		return FNNScaler(self)
	
	def window(self, size, step):
		"""
		Returns a node yielding a sliding window node with this as input.
		
		:param size: Window size.
		:param step: Window step size.
		
		:type size: int
		:type step: int
		
		:return: Sliding window node
		:rtype: featureNetworkNode
		"""
		return FNNWindow(self, windowSize = size, windowStep = step)

#---------------------
# Node type: Motif occurrence frequencies

cdef class FNNMotifOccurrenceFrequencies(featureNetworkNode):
	
	def __init__(self, motifs):
		super().__init__()
		self.motifs = motifs
	
	def __str__(self):
		return 'Motif occurrence frequency<%s>'%str(self.motifs)
	
	def featureNames(self):
		return [ 'occFreq(%s)'%m.name for m in self.motifs ]
	
	cpdef list getSeqVec(self, sequence seq):
		return [
			len(m.find(seq)) * 1000.0 / len(seq.seq)
			for m in self.motifs
		]
	
	cpdef list get(self, sequences seq):
		return [ self.getSeqVec(s) for s in seq ]
	
	def train(self, trainingSet):
		return FNNMotifOccurrenceFrequencies(self.motifs)

#---------------------
# Node type: Motif pair occurrence frequencies

cdef class FNNMotifPairOccurrenceFrequencies(featureNetworkNode):
	
	def __init__(self, motifs, distCut = 219):
		super().__init__()
		self.motifs = motifs
		self.distCut = distCut
	
	def __str__(self):
		return 'Motif pair occurrence frequency<%s>'%str(self.motifs)
	
	def featureNames(self):
		ret = [  ]
		for iA, mA in enumerate(self.motifs):
			for _iB, mB in enumerate(self.motifs[iA:]):
				ret.append('pairFreq(%s, %s, %d)'%(mA.name, mB.name, self.distCut))
		return ret
	
	cpdef list getSeqVec(self, sequence seq):
		cdef list ret
		cdef list occs
		cdef object m, mA, mB
		cdef int iA, _iB, iB, firstRelevantiB
		cdef list alloA, alloB
		cdef motifOccurrence oA, oB
		occs = [
			m.find(seq) for m in self.motifs
		]
		ret = [  ]
		for iA, mA in enumerate(self.motifs):
			for _iB, mB in enumerate(self.motifs[iA:]):
				iB = _iB + iA
				nPairOcc = 0
				alloA = occs[iA]
				alloB = occs[iB]
				firstRelevantiB = 0
				for oA in alloA:
					for oB in alloB[firstRelevantiB:]:
						if iA == iB and oA.start <= oB.start: break
						if oB.end < oA.start - self.distCut: firstRelevantiB += 1; continue
						if oB.start > oA.end + self.distCut: break
						if oB.start-oA.end<oA.start-oB.end:
							d = oA.start-oB.end
						else:
							d = oB.start-oA.end
						if d >= 0 and d <= self.distCut:
							nPairOcc += 1
				ret.append( nPairOcc * 1000.0 / len(seq.seq) )
		return ret
	
	cpdef list get(self, sequences seq):
		return [ self.getSeqVec(s) for s in seq ]
	
	def train(self, trainingSet):
		return FNNMotifPairOccurrenceFrequencies(motifs = self.motifs, distCut = self.distCut)

#---------------------
# Node type: k-spectrum

cdef class kSpectrum(featureNetworkNode):
	"""
	Feature set that extracts the occurrence frequencies of all unambiguous motifs of length k. Extraction of the entire spectrum is optimized by taking overlaps of motifs into account.
	
	:param nspectrum: Length of motifs, k, to use.
	
	:type nspectrum: int
	"""
	
	def __init__(self, nspectrum):
		super().__init__()
		self.nspectrum = nspectrum
		self.nFeatures = (1 << (2*nspectrum))
		self.bitmask = self.nFeatures - 1
		self.kmerByIndex = {}
		for ki in range(self.nFeatures):
			self.kmerByIndex[ki] = ''.join( kSpectrumIndex2NT[(ki >> ((nspectrum - 1 - x)*2)) & 3] for x in range(nspectrum) )
	
	def __str__(self):
		return '%d-spectrum'%self.nspectrum
	
	def featureNames(self):
		return [ self.kmerByIndex[ki] for ki in range(self.nFeatures) ]
	
	cpdef list getSeqVec(self, sequence seq):
		cdef bytes bseq
		cdef unsigned char bnt
		cdef int ki, kiRC, nnt, nspectrum, bitmask, nRCShift, slen
		cdef double nAdd, fv
		cdef list nOcc, convtable
		cdef char c
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
		return nOcc
	
	cpdef list get(self, sequences seq):
		return [ self.getSeqVec(s) for s in seq ]
	
	def train(self, trainingSet):
		return kSpectrum(self.nspectrum)

#---------------------
# Node type: k-spectrum mismatch

cdef class kSpectrumMM(featureNetworkNode):
	"""
	Feature set that extracts the occurrence frequencies of all motifs of length k, with one mismatch allowed in an arbitrary position. Extraction of the entire spectrum is optimized by taking overlaps of motifs into account.
	
	:param nspectrum: Length of motifs, k, to use.
	
	:type nspectrum: int
	"""
	
	def __init__(self, nspectrum):
		super().__init__()
		self.nspectrum = nspectrum
		self.nFeatures = (1 << (2*nspectrum))
		self.bitmask = self.nFeatures - 1
		self.kmerByIndex = {}
		for ki in range(self.nFeatures):
			self.kmerByIndex[ki] = ''.join( kSpectrumIndex2NT[(ki >> ((nspectrum - 1 - x)*2)) & 3] for x in range(nspectrum) )
	
	def __str__(self):
		return '%d-spectrum mismatch'%self.nspectrum
	
	def featureNames(self):
		return [ '%s(1xMM)'%self.kmerByIndex[ki] for ki in range(self.nFeatures) ]
	
	cpdef list getSeqVec(self, sequence seq):
		cdef bytes bseq
		cdef unsigned char bnt
		cdef int ki, kiRC, nnt, nspectrum, bitmask, nRCShift, slen
		cdef double nAdd, fv
		cdef list nOcc, convtable
		cdef char c
		cdef int cki, ckiRC, bki, bkiRC, mutNTI, cmut, cmask, cmuts
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
		return nOcc
	
	cpdef list get(self, sequences seq):
		return [ self.getSeqVec(s) for s in seq ]
	
	def train(self, trainingSet):
		return kSpectrumMM(self.nspectrum)

#---------------------
# Node type: Log-odds

cdef class FNNLogOdds(featureNetworkNode):
	
	def __init__(self, features, weights = None, labelPositive = positive, labelNegative = negative, trainingSet = None):
		super().__init__()
		self.features = features
		self.labelPositive = labelPositive
		self.labelNegative = labelNegative
		self._weights = weights
		self.trainingSet = trainingSet
	
	def __str__(self):
		return 'Log-odds<Features: %s; Positives: %s; Negatives: %s; Training set: %s>'%(str(self.features), str(self.labelPositive), str(self.labelNegative), str(self.trainingSet))
	
	def featureNames(self):
		return self.features.featureNames()
	
	cpdef list get(self, sequences seq):
		cdef float w, fv
		cdef list fvs
		return [
			[
				w * fv
				for w, fv
				in zip(self._weights, fvs)
			]
			for s, fvs in zip(seq, self.features.get(seq))
		]
	
	def weights(self):
		return nctable(
			'Weights: ' + str(self),
			{
				**{
					'Feature': self.featureNames()
				},
				**{
					'Weight': self._weights
				}
			},
			align = { 'Feature': 'l' }
		)
	
	def windowSize(self):
		return self.features.windowSize()
	
	def windowStep(self):
		return self.features.windowStep()
	
	def train(self, trainingSet):
		trainingPositives, trainingNegatives = trainingSet.withLabel([
			self.labelPositive, self.labelNegative ])
		tFeatures = self.features.train(trainingSet)
		fvPos = tFeatures.get(trainingPositives)
		fvNeg = tFeatures.get(trainingNegatives)
		assert(len(fvPos) > 0)
		assert(len(fvPos[0]) > 0)
		assert(len(fvNeg) > 0)
		assert(len(fvPos[0]) == len(fvNeg[0]))
		weights = []
		for i, _ in enumerate(fvPos[0]):
			vPos = sum( fv[i] for fv in fvPos )
			vNeg = sum( fv[i] for fv in fvNeg )
			if vPos == 0.0 or vNeg == 0.0:
				vPos += 1.0
				vNeg += 1.0
			weight = log(vPos) - log(float(len(trainingPositives)))\
				   - (log(vNeg) - log(float(len(trainingNegatives))))
			weights.append(weight)
		return FNNLogOdds(
			features = tFeatures,
			weights = weights,
			labelPositive = self.labelPositive,
			labelNegative = self.labelNegative,
			trainingSet = trainingSet)

#---------------------
# Node type: Scaler

cdef class FNNScaler(featureNetworkNode):
	
	def __init__(self, features, vScale = None, vSub = None):
		super().__init__()
		self.features = features
		self.vScale = vScale
		self.vSub = vSub
	
	def __str__(self):
		return 'Scaler<%s; Trained: %s>'%(str(self.features), 'No' if self.vScale is None else 'Yes')
	
	def featureNames(self):
		return self.features.featureNames()
	
	cpdef list get(self, sequences seq):
		cdef list sfv
		cdef float fv
		cdef int i
		return [
			[
				fv * self.vScale[i] - self.vSub[i]
				for i, fv
				in enumerate(sfv)
			]
			for sfv in self.features.get(seq)
		]
	
	def weights(self):
		return self.features.weights()
	
	def windowSize(self):
		return self.features.windowSize()
	
	def windowStep(self):
		return self.features.windowStep()
	
	def train(self, trainingSet):
		tFeatures = self.features.train(trainingSet)
		fvs = tFeatures.get(trainingSet)
		vScale = []
		vSub = []
		for i, _ in enumerate(fvs[0]):
			fv = [ fv[i] for fv in fvs ]
			cvMin = min(fv)
			cvMax = max(fv)
			rng = cvMax - cvMin
			if rng == 0.0:
				cvScale = 0.0
			else:
				cvScale = 2.0 / rng
			cvSub = cvMin * cvScale + 1.0
			vScale.append(cvScale)
			vSub.append(cvSub)
		return FNNScaler(tFeatures, vScale, vSub)

#---------------------
# Node type: Concatenation

cdef class FNNCat(featureNetworkNode):
	
	def __init__(self, inputs):
		super().__init__()
		self.inputs = inputs
	
	def __str__(self):
		return '[ ' + '; '.join(str(i) for i in self.inputs) + ' ]'
	
	def featureNames(self):
		return [ n for i in self.inputs for n in i.featureNames() ]
	
	cpdef list get(self, sequences seq):
		cdef list I
		cdef featureNetworkNode i
		cdef int sI, iI
		cdef float v
		I = [
			i.get(seq) for i in self.inputs
		]
		return [
			[
				v
				for iI in range(len(self.inputs))
				for v in I[iI][sI]
			]
			for sI in range(len(seq))
		]
	
	def train(self, trainingSet):
		return FNNCat([ i.train(trainingSet) for i in self.inputs ])

#---------------------
# Node type: Sum

class FNNSum(featureNetworkNode):
	
	def __init__(self, inputs):
		super().__init__()
		self.inputs = inputs
	
	def __str__(self):
		return 'Sum<%s>'%(str(self.inputs))
	
	def featureNames(self):
		return self.inputs.featureNames()
	
	def get(self, seq):
		fvs = self.inputs.get(seq)
		return [
			[ sum(sfv) ]
			for sfv in fvs
		]
	
	def weights(self):
		return self.inputs.weights()
	
	def windowSize(self):
		return self.inputs.windowSize()
	
	def windowStep(self):
		return self.inputs.windowStep()
	
	def train(self, trainingSet):
		return FNNSum(self.inputs.train(trainingSet))

#---------------------
# Node type: Square

class FNNSquare(featureNetworkNode):
	
	def __init__(self, inputs):
		super().__init__()
		self.inputs = inputs
	
	def __str__(self):
		return 'Square<%s>'%(str(self.inputs))
	
	def featureNames(self):
		names = self.inputs.featureNames()
		return [
			'%s * %s'%(vA, vB)
			for iA, vA in enumerate(names)
			for vB in names[:iA+1]
		]
	
	def get(self, seq):
		fvs = self.inputs.get(seq)
		return [
			[
				vA * vB
				for iA, vA in enumerate(sfv)
				for vB in sfv[:iA+1]
			]
			for sfv in fvs
		]
	
	def weights(self):
		return self.inputs.weights()
	
	def windowSize(self):
		return self.inputs.winSize
	
	def windowStep(self):
		return self.inputs.winStep
	
	def train(self, trainingSet):
		return FNNSquare(self.inputs.train(trainingSet))

#---------------------
# Node type: Window

class FNNWindow(featureNetworkNode):
	
	def __init__(self, inputs, windowSize, windowStep):
		super().__init__()
		self.inputs = inputs
		self.winSize = windowSize
		self.winStep = windowStep
	
	def __str__(self):
		return 'Sliding window<%s; Window size: %d; Window step size: %d>'%(str(self.inputs), self.winSize, self.winStep)
	
	def featureNames(self):
		return self.inputs.featureNames()
	
	def get(self, seq):
		win = sequences(seq.name, [ w for s in seq for w in s.windows(self.winSize, self.winStep) ])
		return self.inputs.get(win)
	
	def weights(self):
		return self.inputs.weights()
	
	def windowSize(self):
		return self.winSize
	
	def windowStep(self):
		return self.winStep
	
	def train(self, trainingSet):
		twin = sequences(trainingSet.name, [ w for s in trainingSet for w in s.windows(self.winSize, self.winStep) ])
		return FNNWindow(self.inputs.train(twin), windowSize = self.winSize, windowStep = self.winStep)

#---------------------
# Base model

class baseModel:
	
	def __init__(self):
		pass
	
	def __str__(self): return 'Base model'
	
	def __repr__(self): return self.__str__()
	
	def score(self, featureVectors):
		pass
	
	def train(self, trainingSet):
		pass
	
	def weights(self, featureNames):
		pass

class logOdds(baseModel):
	
	def __init__(self, weights = None, labelPositive = positive, labelNegative = negative):
		super().__init__()
		self._weights = weights
		self.labelPositive = labelPositive
		self.labelNegative = labelNegative
	
	def __str__(self):
		return 'Log-odds<Positive label: %s; Negative label: %s>'%(str(self.labelPositive), str(self.labelNegative))
	
	def score(self, featureVectors):
		return [
			[
				fv * w
				for fv, w in zip(fvec, self._weights)
			]
			for fvec in featureVectors
		]
	
	def train(self, trainingSet):
		fvPos = trainingSet[self.labelPositive]
		fvNeg = trainingSet[self.labelNegative]
		assert(len(fvPos) > 0)
		assert(len(fvPos[0]) > 0)
		assert(len(fvNeg) > 0)
		assert(len(fvPos[0]) == len(fvNeg[0]))
		weights = []
		for i, _ in enumerate(fvPos[0]):
			vPos = sum( fv[i] for fv in fvPos )
			vNeg = sum( fv[i] for fv in fvNeg )
			if vPos == 0.0 or vNeg == 0.0:
				vPos += 1.0
				vNeg += 1.0
			weight = log(vPos) - log(float(len(fvPos)))\
				   - (log(vNeg) - log(float(len(fvNeg))))
			weights.append(weight)
		return logOdds(
			weights = weights,
			labelPositive = self.labelPositive,
			labelNegative = self.labelNegative)
	
	def weights(self, featureNames):
		return nctable(
			'Weights: ' + str(self),
			{
				**{
					'Feature': featureNames
				},
				**{
					'Weight': self._weights
				}
			},
			align = { 'Feature': 'l' }
		)

class FNNModel(featureNetworkNode):
	
	def __init__(self, features, model, trainingSet = None):
		self.features = features
		self.mdl = model
		self.trainingSet = trainingSet
	
	def __str__(self):
		return 'Model<Base model: %s; Features: %s; Training set: %s>'%(str(self.mdl), str(self.features), str(self.trainingSet))
	
	def featureNames(self):
		return self.features.featureNames()
	
	def get(self, seq):
		return self.mdl.score(self.features.get(seq))
	
	def weights(self):
		return self.mdl.weights(self.features.featureNames())
	
	def windowSize(self):
		return self.features.windowSize()
	
	def windowStep(self):
		return self.features.windowStep()
	
	def train(self, trainingSet):
		fs = self.features.train(trainingSet)
		ts = {
			lbl: fs.get(trainingSet.withLabel(lbl))
			for lbl in trainingSet.labels()
		}
		model = self.mdl.train(ts)
		return FNNModel(features = fs, model = model, trainingSet = trainingSet)

#---------------------
# Sequence model

class sequenceModelFNN(sequenceModel):
	
	def __init__(self, name, features, windowSize = -1, windowStep = -1):
		super().__init__(name)
		self.threshold = 0.0
		self.features = features
		if windowSize == -1:
			windowSize = features.windowSize()
		if windowStep == -1:
			windowStep = features.windowStep()
		self.windowSize, self.windowStep = windowSize, windowStep
	
	def __str__(self):
		return 'Model<%s>'%(str(self.features))
	
	def __repr__(self): return self.__str__()
	
	def weights(self): return self.features.weights()
	
	def getTrainer(self):
		return lambda ts: sequenceModelFNN(self.name, self.features.train(ts), self.windowSize, self.windowStep)
	
	def scoreWindow(self, seq):
		cdef list fvs, fv
		cdef float ret = 0.0
		fvs = self.features.get(sequences('', [ seq ]))
		ret = max([
			sum(fv)
			for fv in fvs
		])
		return ret

