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
from .sequences import positive, negative
from .models import sequenceModel
from math import log
from .common import KLdiv
from .ioputil import nctable


############################################################################
# Feature network

#---------------------
# Base class

cdef class featureNetworkNode:
	"""
	The `feature` class is a base class for sequence network nodes.
	"""
	
	def __init__(self):
		pass
	
	cpdef list get(self, sequence seq):
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
		fv = [
			self.get(s)
			for s in seqs
		]
		_dict = {
			**{
				'Seq.': [ s.name for s in seqs ],
			},
			**{
				fName: [ fv[sI][fI] for sI in range(len(seqs)) ]
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
	
	def model(self, name, windowSize, windowStep):
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

#---------------------
# Node type: Motif occurrence frequencies

class FNNMotifOccurrenceFrequencies(featureNetworkNode):
	
	def __init__(self, motifs):
		super().__init__()
		self.motifs = motifs
	
	def __str__(self):
		return 'Motif occurrence frequency<%s>'%str(self.motifs)
	
	def featureNames(self):
		return [ 'occFreq(%s)'%m.name for m in self.motifs ]
	
	def get(self, seq):
		return [
			len(m.find(seq)) * 1000.0 / len(seq.seq)
			for m in self.motifs
		]
	
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
	
	cpdef list get(self, sequence seq):
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
	
	def train(self, trainingSet):
		return FNNMotifPairOccurrenceFrequencies(motifs = self.motifs, distCut = self.distCut)

#---------------------
# Node type: Log-odds

class FNNLogOdds(featureNetworkNode):
	
	def __init__(self, features, weights = None, labelPositive = positive, labelNegative = negative, trainingSet = None):
		super().__init__()
		self.features = features
		self.labelPositive = labelPositive
		self.labelNegative = labelNegative
		self.weights = weights
		self.trainingSet = trainingSet
	
	def __str__(self):
		return 'Log-odds<Features: %s; Positives: %s; Negatives: %s; Training set: %s>'%(str(self.features), str(self.labelPositive), str(self.labelNegative), str(self.trainingSet))
	
	def featureNames(self):
		return self.features.featureNames()
	
	def get(self, seq):
		return [
			w * fv
			for w, fv
			in zip(self.weights, self.features.get(seq))
		]
	
	def train(self, trainingSet):
		trainingPositives, trainingNegatives = trainingSet.withLabel([
			self.labelPositive, self.labelNegative ])
		tFeatures = self.features.train(trainingSet)
		fvPos = [ tFeatures.get(s) for s in trainingPositives ]
		fvNeg = [ tFeatures.get(s) for s in trainingNegatives ]
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
# Node type: Concatenation

class FNNCat(featureNetworkNode):
	
	def __init__(self, inputs):
		super().__init__()
		self.inputs = inputs
	
	def __str__(self):
		return '[ ' + '; '.join(str(i) for i in self.inputs) + ' ]'
	
	def featureNames(self):
		return [ n for i in self.inputs for n in i.featureNames() ]
	
	def get(self, seq):
		return [ v for i in self.inputs for v in i.get(seq) ]
	
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
		return [ sum(self.inputs.get(seq)) ]
	
	def train(self, trainingSet):
		return FNNSum(self.inputs.train(trainingSet))

#---------------------
# Sequence model

class sequenceModelFNN(sequenceModel):
	
	def __init__(self, name, features, windowSize, windowStep):
		super().__init__(name)
		self.threshold = 0.0
		self.features = features
		self.windowSize, self.windowStep = windowSize, windowStep
	
	def __str__(self):
		return 'Model<Features: %s>'%(str(self.features))
	
	def __repr__(self): return self.__str__()
	
	def getTrainer(self):
		return lambda ts: sequenceModelFNN(self.name, self.features.train(ts), self.windowSize, self.windowStep)
	
	def scoreWindow(self, seq):
		cdef list fv
		cdef float ret = 0.0
		fv = self.features.get(seq)
		for i in range(len(fv)):
			ret += fv[i]
		return ret

