# distutils: language=c++
# -*- coding: latin-1 -*-
############################################################################
# Gnocis
# Bjørn Bredesen, 2018-2019
# bjorn.bredesen@ii.uib.no
############################################################################

from __future__ import division
import random
from libcpp cimport bool
from libc.stdlib cimport malloc, free	
from libc.string cimport memcpy
import multiprocessing as mp
import math
from .features cimport *
from .validation cimport *
from .validation import getConfusionMatrix, getConfusionMatrixStatistics, getROC, getPRC, getAUC, printValidationStatistics
from .sequences import streamSequenceWindows, positive, negative
from .ioputil import nctable
from .common import mean, CI


############################################################################
# Multiprocessing

nCoresUse = 1
multiprocessingPool = None
nTreadFetch = 100000
maxThreadFetchNT = nTreadFetch * 1000

# Used for applying models in separate processes
def multiprocessModelDeployer(args):
	cdef sequences seqs
	cdef sequence seq, win
	cdef defaultValue = float('-INF')
	global model
	seqs = args
	cdef int windowSize = model.windowSize
	cdef int windowStep = model.windowStep
	return [
		max(
			(
				model.scoreWindow(win)
				for win in
				seq.windows(windowSize, windowStep)
				if len(win) == windowSize
			),
			default = defaultValue
		)
		for seq in seqs
	]

def initializePoolProcess(arg):
	global processID, model
	processID, model = None, None

def setPoolProcessID(newID):
	global processID
	processID = newID
	return True

def getPoolProcessID(arg):
	global processID
	return processID

def getPoolProcessModelID(arg):
	global processID, model
	if model == None:
		return processID, None
	return processID, '%d %s'%(model.modelID, str(model))

def initializePoolProcessIDs(pool):
	while True:
		pool.map( setPoolProcessID, list(range(len(pool._pool))) )
		procIDs = set( pool.map( getPoolProcessID, [ () for _ in range(len(pool._pool)) ] ) )
		if len(procIDs) == len(pool._pool):
			break

def getPoolProcessIDs(pool):
	pIDs = set()
	while True:
		pIDs = pIDs | set( v for v in pool.map( getPoolProcessID, [ () for _ in range(len(pool._pool)) ] ) if v != None )
		if len(pIDs) == len(pool._pool):
			return pIDs

def getPoolModelIDs(pool):
	pIDs = set()
	mIDs = set()
	while True:
		r = pool.map( getPoolProcessModelID, [ () for _ in range(len(pool._pool)) ] )
		pIDs = pIDs | set( pid for pid, mid in r if pid != None )
		mIDs = mIDs | set( mid for pid, mid in r if pid != None )
		if len(pIDs) == len(pool._pool):
			return mIDs

def setPoolProcessModel(mdl):
	global model
	model = mdl
	return True

def getPoolProcessHasModel(arg):
	global model, processID
	return (processID, model != None)

def setPoolModel(pool, mdl):
	pIDs = set()
	if mdl == None:
		# Purge models from all processes
		# The process-ID is used to ensure that all processes have updated data
		while True:
			pool.map( setPoolProcessModel, [ None for _ in range(len(pool._pool)) ] )
			r = pool.map( getPoolProcessHasModel, [ () for _ in range(len(pool._pool)) ] )
			# We append the process-IDs that we confirmed are done
			pIDs = pIDs | set( pid for pid, hasModel in r if not hasModel )
			if len(pIDs) == len(pool._pool):
				break
	if mdl != None:
		# For non-None model, first purge the original model
		setPoolModel(pool, None)
		# Then set the model for all processes
		# The process-ID is used to ensure that all processes have updated data
		while True:
			pool.map( setPoolProcessModel, [ mdl for _ in range(len(pool._pool)) ] )
			r = pool.map( getPoolProcessHasModel, [ () for _ in range(len(pool._pool)) ] )
			pIDs = pIDs | set( pid for pid, hasModel in r if hasModel )
			if len(pIDs) == len(pool._pool):
				break

def setNCores(n):
	"""
	Sets the number of cores to use for multiprocessing.
	
	:param n: The number of cores to use.
	
	:type n: int
	"""
	global nCoresUse, multiprocessingPool
	nCoresUse = n
	if multiprocessingPool != None:
		multiprocessingPool.close()
		del multiprocessingPool
		multiprocessingPool = None
	if n > 1:
		multiprocessingPool = mp.Pool(nCoresUse, initializer = initializePoolProcess, initargs = ('',))
		initializePoolProcessIDs(multiprocessingPool)


############################################################################
# Models

maxModelID = 0

# Represents sequence models.
class sequenceModel:
	"""
	The `sequenceModel` class is an abstract class for sequence models. A number of methods are implemented for machine learning and prediction with DNA sequences.
	"""
	
	def __init__(self, name):
		self.name = name
	
	# For pickling (used with multiprocessing)
	def __getstate__(self):
		return self.__dict__
	
	def __setstate__(self, state):
		self.__dict__ = state
	
	# Gets a list of scores for a sequence list or stream. A stream is recommended when the sequence list is large, in order to avoid running out of memory.
	def getSequenceScores(self, seqs):
		""" Scores a set of sequences, returning the maximum window score for each.
		
		:param seqs: Sequences to score.
		
		:type seqs: sequences/sequenceStream
		
		:return: List of the maximum window score per sequence
		:rtype: list
		"""
		cdef float score
		cdef list blkScores
		cdef sequences blk
		cdef sequence cseq
		if nCoresUse > 1:
			# Wipe the first model, to ensure we have a clean slate
			if not 'modelID' in self.__dict__.keys():
				global maxModelID
				self.modelID = maxModelID
				maxModelID += 1
			setPoolModel(multiprocessingPool, self)
			# Process in parallel
			if isinstance(seqs, sequenceStream):
				return [
					score
					for blk in seqs.fetch(nTreadFetch, maxThreadFetchNT)
					for blkScores in multiprocessingPool.map(
						multiprocessModelDeployer,
						blk.split(nCoresUse)
					)
					for score in blkScores
				]
			elif isinstance(seqs, sequences) or isinstance(seqs, list):
				return [
					score
					for blkScores in multiprocessingPool.map(
						multiprocessModelDeployer,
						seqs.split(nCoresUse)
					)
					for score in blkScores
				]
		if isinstance(seqs, sequenceStream):
			return [
				self.scoreSequence(cseq)
				for blk in seqs.fetch(nTreadFetch, maxThreadFetchNT)
				for cseq in blk
			]
		elif isinstance(seqs, sequences) or isinstance(seqs, list):
			return [ self.scoreSequence(cseq) for cseq in seqs ]
	
	# Gets the threshold that gives optimal accuracy on a pair of lists or streams of sequences. Streams are recommended when the sequence lists are large, in order to avoid running out of memory.
	def getOptimalAccuracyThreshold(self, seqs, labelPositive = positive, labelNegative = negative):
		""" Gets a threshold value optimized for accuracy to a set of positive and a set of negative sequences.
		
		:param seqs: Sequences.
		:param labelPositive: Label of positives.
		:param labelNegative: Label of negatives.
		
		:type seqs: sequences
		:type labelPositive: sequenceLabel
		:type labelNegative: sequenceLabel
		
		:return: List of the maximum window score per sequence
		:rtype: list
		"""
		positives, negatives = seqs.withLabel([ labelPositive, labelNegative ])
		positiveScores = [ validationPair(score = score, label = True) for score in self.getSequenceScores(positives) ]
		negativeScores = [ validationPair(score = score, label = False) for score in self.getSequenceScores(negatives) ]
		vPairs = sorted(positiveScores + negativeScores, key = lambda x: x.score)
		TP, FP, TN, FN = len(positiveScores), len(negativeScores), 0, 0
		threshold = float('-INF')
		ACC = float( TP + TN ) / float( TP + TN + FP + FN )
		for vp in vPairs:
			if vp.label:
				TP -= 1
				FN += 1
			else:
				FP -= 1
				TN += 1
			cACC = float( TP + TN ) / float( TP + TN + FP + FN )
			if cACC >= ACC:
				ACC = cACC
				threshold = vp.score
		return threshold
	
	# Gets the threshold that gives a desired precision for two lists or streams of sequences. Streams are recommended when the sequence lists are large, in order to avoid running out of memory.
	def getPrecisionThreshold(self, positives, negatives, wantedPrecision):
		""" Gets a threshold value for a desired precision to a set of positive and a set of negative sequences. Linear interpolation is used in order to achieve a close approximation.
		
		:param seqs: Sequences.
		:param labelPositive: Label of positives.
		:param labelNegative: Label of negatives.
		:param wantedPrecision: The precision to approximate.
		
		:type seqs: sequences
		:type labelPositive: sequenceLabel
		:type labelNegative: sequenceLabel
		:type wantedPrecision: float
		
		:return: List of the maximum window score per sequence
		:rtype: list
		"""
		positiveScores = [ validationPair(score = score, label = True) for score in self.getSequenceScores(positives) ]
		negativeScores = [ validationPair(score = score, label = False) for score in self.getSequenceScores(negatives) ]
		#
		classScoresSorted = sorted( positiveScores + negativeScores, key = lambda x: x.score )
		# Search for optimal threshold
		TP, FP = ( len(positiveScores), len(negativeScores) )
		aThreshold, bThreshold = (0, 0)
		aPrecision, bPrecision = (-1, -1)
		for e in classScoresSorted:
			aPrecision = bPrecision
			bPrecision = float(TP)/float(TP+FP)
			aThreshold = bThreshold
			bThreshold = e.score
			# This observation will be below the threshold in the next iteration,
			# so remove count depending on whether it was a true or false positive
			if bPrecision >= wantedPrecision:
				break
			if e.label:
				TP -= 1
			else:
				FP -= 1
		if aPrecision != bPrecision:
			wPrecision = bPrecision-aPrecision
			wThreshold = bThreshold-aThreshold
			threshold = (wantedPrecision-aPrecision)*(wThreshold/wPrecision) + aThreshold
		else:
			threshold = aThreshold
		return threshold
	
	# Calibrates the model threshold for an expected precision genome-wide. Returns self to facilitate chaining of operations. However, this operation does mutate the model object. A scaling factor can be applied to the genome with the 'factor' argument. If, for instance, the positive set has been divided in half for independent training and calibration, a factor of 0.5 can be used.
	def calibrateGenomewidePrecision(self, positives, genome, factor = 1.0, precision = 0.8, bgModelOrder = 4, bgModel = None):
		""" Calibrates the model threshold for an expected precision genome-wide. Returns self to facilitate chaining of operations. However, this operation does mutate the model object. A scaling factor can be applied to the genome with the 'factor' argument. If, for instance, the positive set has been divided in half for independent training and calibration, a factor of 0.5 can be used.
		
		:param positives: Positive sequences.
		:param genome: The genome.
		:param factor: Scaling factor for positive set versus genome.
		:param precision: The precision to approximate.
		:type bgModelOrder: The order of the background model.
		:type bgModel: A background model to use. If specified, the `bgModelOrder` argument will be ignored, and this model will be used to generate the background. Otherwise, a background model is trained on the genome.
		
		:type positives: sequences/sequenceStream
		:type genome: sequences/sequenceStream/str
		:type factor: float, optional
		:type precision: float, optional
		:type bgModelOrder: float, optional
		:type bgModel: object, optional
		
		:return: List of the maximum window score per sequence
		:rtype: list
		"""
		# Find genome size.
		if isinstance(genome, str):
			genome = getSequenceStreamFromPath(genome)
		seqLens = genome.sequenceLengths()
		genomeLen = sum(seqLens[chrom] for chrom in seqLens.keys())
		# Train background model
		if bgModel == None:
			bgModel = generatorIID(trainingSequences = genome) \
				if bgModelOrder <= 0 \
				else generatorMarkovChain(trainingSequences = genome, degree = bgModelOrder)
		# Generate calibration sequences
		meanLen = int(sum(len(s) for s in positives) / len(positives))
		negatives = bgModel.generateStream(n = int(factor * genomeLen / meanLen), length = meanLen)
		# Calibrate threshold
		self.threshold = self.getPrecisionThreshold(positives = positives, negatives = negatives, wantedPrecision = precision)
		return self
	
	# Applies model to a sequence with a sliding window, and returns the maximum score.
	def scoreSequence(self, seq):
		""" Scores a single sequence. The score is determined by applying the model with a sliding window, and taking the maximum score.
		
		:param seq: The sequence.
		
		:type seq: sequences/sequenceStream
		
		:return: Maximal window score
		:rtype: float
		"""
		cdef float score
		score = max(
			(
				self.scoreWindow(win)
				for win in
				seq.windows(self.windowSize, self.windowStep)
				if len(win) == self.windowSize
			),
			default = float('-INF')
		)
		return score
	
	# Gets and returns a confusion matrix.
	def getConfusionMatrix(self, seqs, labelPositive = positive, labelNegative = negative):
		""" Calculates and returns a confusion matrix for sets of positive and negative sequences.
		
		:param seqs: Sequences.
		:param labelPositive: Label of positives.
		:param labelNegative: Label of negatives.
		
		:type seqs: sequences
		:type labelPositive: sequenceLabel
		:type labelNegative: sequenceLabel
		
		:return: Confusion matrix
		:rtype: dict
		"""
		positives, negatives = seqs.withLabel([ labelPositive, labelNegative ])
		vPos = [ validationPair(score = score, label = True) for score in self.getSequenceScores(positives) ]
		vNeg = [ validationPair(score = score, label = False) for score in self.getSequenceScores(negatives) ]
		return getConfusionMatrix(vPos, vNeg, threshold = self.threshold)
	
	# Generates a Receiver Operating Characteristic curve.
	def getROC(self, seqs, labelPositive = positive, labelNegative = negative):
		""" Calculates and returns a Receiver Operating Characteristic (ROC) curve for sets of positive and negative sequences.
		
		:param seqs: Sequences.
		:param labelPositive: Label of positives.
		:param labelNegative: Label of negatives.
		
		:type seqs: sequences
		:type labelPositive: sequenceLabel
		:type labelNegative: sequenceLabel
		
		:return: Receiver Operating Characteristic (ROC) curve
		:rtype: list
		"""
		positives, negatives = seqs.withLabel([ labelPositive, labelNegative ])
		vPos = [ validationPair(score = score, label = True) for score in self.getSequenceScores(positives) ]
		vNeg = [ validationPair(score = score, label = False) for score in self.getSequenceScores(negatives) ]
		return getROC(vPos, vNeg)
	
	# Calculates the area under a Receiver Operating Characteristic curve.
	def getROCAUC(self, seqs, labelPositive = positive, labelNegative = negative):
		""" Calculates and returns the area under a Receiver Operating Characteristic (ROCAUC) curve for sets of positive and negative sequences.
		
		:param seqs: Sequences.
		:param labelPositive: Label of positives.
		:param labelNegative: Label of negatives.
		
		:type seqs: sequences
		:type labelPositive: sequenceLabel
		:type labelNegative: sequenceLabel
		
		:return: Area under the Receiver Operating Characteristic (ROCAUC) curve
		:rtype: list
		"""
		return getAUC(self.getROC(seqs, labelPositive = labelPositive, labelNegative = labelNegative))
	
	# Generates a Precision/Recall curve.
	def getPRC(self, seqs, labelPositive = positive, labelNegative = negative):
		""" Calculates and returns a Precision/Recall curve (PRC) for sets of positive and negative sequences.
		
		:param seqs: Sequences.
		:param labelPositive: Label of positives.
		:param labelNegative: Label of negatives.
		
		:type seqs: sequences
		:type labelPositive: sequenceLabel
		:type labelNegative: sequenceLabel
		
		:return: Precision/Recall curve (PRC)
		:rtype: list
		"""
		positives, negatives = seqs.withLabel([ labelPositive, labelNegative ])
		vPos = [ validationPair(score = score, label = True) for score in self.getSequenceScores(positives) ]
		vNeg = [ validationPair(score = score, label = False) for score in self.getSequenceScores(negatives) ]
		return getPRC(vPos, vNeg)
	
	# Calculates the area under a Precision/Recall Curve.
	def getPRCAUC(self, seqs, labelPositive = positive, labelNegative = negative):
		""" Calculates and returns the area under a Precision/Recall curve (PRCAUC) for sets of positive and negative sequences.
		
		:param seqs: Sequences.
		:param labelPositive: Label of positives.
		:param labelNegative: Label of negatives.
		
		:type seqs: sequences
		:type labelPositive: sequenceLabel
		:type labelNegative: sequenceLabel
		
		:return: Area under the Precision/Recall curve (PRCAUC)
		:rtype: list
		"""
		return getAUC(self.getPRC(seqs, labelPositive = labelPositive, labelNegative = labelNegative))
	
	def plotPRC(self, seqs, labelPositive = positive, labelNegative = negative, figsize = (8, 8), outpath = None, style = 'ggplot'):
		""" Plots a Precision/Recall curve and either displays it in an IPython session or saves it to a file.
		
		:param seqs: Sequences.
		:param labelPositive: Label of positives.
		:param labelNegative: Label of negatives.
		:param figsize: Tuple of figure dimensions.
		:param outpath: Path to save generated plot to. If not set, the plot will be output to IPython.
		:param style: Matplotlib style to use.
		
		:type seqs: sequences
		:type labelPositive: sequenceLabel
		:type labelNegative: sequenceLabel
		:type figsize: tuple, optional
		:type outpath: str, optional
		:type style: str, optional
		"""
		try:
			import matplotlib.pyplot as plt
			import base64
			from io import BytesIO
			from IPython.core.display import display, HTML
			with plt.style.context(style):
				fig = plt.figure(figsize = figsize)
				positives, negatives = seqs.withLabel([ labelPositive, labelNegative ])
				curve = self.getPRC(seqs, labelPositive = labelPositive, labelNegative = labelNegative)
				plt.plot([ 0., 1. ],
					[ len(positives) / (len(positives) + len(negatives)),
						len(positives) / (len(positives) + len(negatives))],
					linestyle = '--',
					label = 'Expected at random')
				plt.plot([ v.x for v in curve ],
					[ v.y for v in curve ],
					label = '%s - AUC = %.2f %%'%(self.name, getAUC(curve)*100.))
				plt.xlabel('Recall', fontsize = 14)
				plt.ylabel('Precision', fontsize = 14)
				plt.legend(loc = 'upper right', fancybox = True)
				if outpath is None:
					bio = BytesIO()
					fig.savefig(bio, format='png')
					plt.close('all')
					encoded = base64.b64encode(bio.getvalue()).decode('utf-8')
					html = '<img src=\'data:image/png;base64,%s\'>'%encoded
					display(HTML(html))
				else:
					fig.savefig(outpath)
					plt.close('all')
		except ImportError as err:
			raise err
	
	# Returns validation statistics dictionary.
	def getValidationStatistics(self, seqs, labelPositive = positive, labelNegative = negative):
		""" Returns common model validation statistics (confusion matrix values; ROCAUC; PRCAUC).
		
		:param seqs: Sequences.
		:param labelPositive: Label of positives.
		:param labelNegative: Label of negatives.
		
		:type seqs: sequences
		:type labelPositive: sequenceLabel
		:type labelNegative: sequenceLabel
		
		:return: Validation statistics
		:rtype: list
		"""
		CMStats = getConfusionMatrixStatistics( self.getConfusionMatrix(seqs, labelPositive = labelPositive, labelNegative = labelNegative) )
		stats = {
			'title': 'Validation', 'positives': labelPositive, 'negatives': labelNegative, 'model': self,
			'ROC AUC': self.getROCAUC(seqs, labelPositive = labelPositive, labelNegative = labelNegative),
			'PRC AUC': self.getPRCAUC(seqs, labelPositive = labelPositive, labelNegative = labelNegative),
		}
		for k in CMStats.keys():
			stats[k] = CMStats[k]
		return stats
	
	# Predicts regions inside a sequence stream using a sliding window.
	def predictSequenceStreamRegions(self, stream):
		""" Applies the model using a sliding window across an input sequence stream or sequence set. Windows with a score >= self.threshold are predicted, and predicted windows are merged into non-overlapping predictions.
		
		:param stream: Sequence material that the model is applied to for prediction.
		:type stream: sequences/sequenceStream/str
		
		:return: Set of non-overlapping (merged) predicted regions
		:rtype: regions
		"""
		cdef regions pred
		cdef list winPosSet = []
		cdef list winScoreSet = []
		cdef int i = 0
		cdef int nNT = 0
		cdef region winPos
		cdef sequence winSeq
		cdef sequences winSeqSet = sequences('', [])
		cdef float wScore
		cdef list cScores
		pred = regions('Predictions', [])
		
		for winSeq in streamSequenceWindows(stream, self.windowSize, self.windowStep):
			winPosSet.append(winSeq.sourceRegion)
			winSeqSet.sequences.append(winSeq)
			i += 1
			nNT += len(winSeq)
			if nNT >= maxThreadFetchNT:
				winScoreSet += self.getSequenceScores(winSeqSet)
				winSeqSet.sequences = []
				i = 0
				nNT = 0
		if len(winSeqSet.sequences) > 0:
			winScoreSet += self.getSequenceScores(winSeqSet)
		
		for winPos, wScore in zip(winPosSet, winScoreSet):
			if wScore <= self.threshold: continue
			if len(pred) == 0 or pred[-1].seq != winPos.seq or pred[-1].end < winPos.start-1:
				cScores = [ ]
				pred.regions.append(winPos)
			else:
				pred[-1].end = winPos.end
			cScores.append(wScore)
			pred[-1].score = sum(cScores) / len(cScores)
		pred.sort()
		return pred
	
	# Short-hand for predictSequenceStreamRegions.
	def predict(self, stream):
		""" Applies the model using a sliding window across an input sequence stream or sequence set. Windows with a score >= self.threshold are predicted, and predicted windows are merged into non-overlapping predictions.
		
		:param stream: Sequence material that the model is applied to for prediction.
		:type stream: sequences/sequenceStream/str
		
		:return: Set of non-overlapping (merged) predicted regions
		:rtype: regions
		"""
		return self.predictSequenceStreamRegions(stream)
	
	# Prints out test statistics.
	def printTestStatistics(self, seqs, labelPositive = positive, labelNegative = negative):
		printValidationStatistics( self.getValidationStatistics(seqs, labelPositive = labelPositive, labelNegative = labelNegative) )

# Uniformly weighted model
class sequenceModelDummy(sequenceModel):
	"""
	This model takes a feature set, and scores input sequences by summing feature values, without weighting.
	
	:param name: Model name.
	:param features: The feature set.
	:param windowSize: Window size to use.
	:param windowStep: Window step size to use.
	
	:type name: str
	:type features: features
	:type windowSize: int
	:type windowStep: int
	"""
	
	def __init__(self, name, features, windowSize, windowStep):
		super().__init__(name)
		self.threshold = 0.0
		self.features = features
		self.windowSize, self.windowStep = windowSize, windowStep
	
	def __str__(self):
		return 'Dummy model<Features: %s>'%(str(self.features))
	
	def __repr__(self): return self.__str__()
	
	def getTrainer(self):
		return lambda ts: sequenceModelDummy(self.name, self.features, self.windowSize, self.windowStep)
	
	def scoreWindow(self, seq):
		cdef list fv
		cdef float ret, v
		fv = self.features.getAll(seq)
		ret = 0.0
		for v in fv:
			ret += v
		return ret

# Log-odds model
class sequenceModelLogOdds(sequenceModel):
	"""
	Constructs a log-odds model based on an input feature set and binary training set, and scores input sequences by summing log-odds-weighted feature values.
	
	:param name: Model name.
	:param features: The feature set.
	:param trainingSet: Training sequences.
	:param windowSize: Window size to use.
	:param windowStep: Window step size to use.
	:param labelPositive: Positive training class label.
	:param labelNegative: Negative training class label.
	
	:type name: str
	:type features: features
	:type trainingSet: sequences
	:type windowSize: int
	:type windowStep: int
	:type labelPositive: sequenceLabel
	:type labelNegative: sequenceLabel
	"""
	
	def __init__(self, name, features, trainingSet, windowSize, windowStep, labelPositive = positive, labelNegative = negative):
		super().__init__(name)
		self.threshold = 0.0
		self.features = features
		self.trainingSet = trainingSet
		trainingPositives, trainingNegatives = trainingSet.withLabel([ labelPositive, labelNegative ])
		self.labelPositive, self.labelNegative = labelPositive, labelNegative
		self.trainingPositives, self.trainingNegatives = trainingPositives, trainingNegatives
		self.windowSize, self.windowStep = windowSize, windowStep
		self.weights = {}
		self.sortedweights = []
		for f in features:
			vPos = sum( f.get(s) for s in trainingPositives )
			vNeg = sum( f.get(s) for s in trainingNegatives )
			if vPos == 0.0 or vNeg == 0.0:
				vPos += 1.0
				vNeg += 1.0
			weight = math.log(vPos) - math.log(float(len(trainingPositives))) - (math.log(vNeg) - math.log(float(len(trainingNegatives))))
			self.weights[f] = weight
			self.sortedweights.append(weight)
	
	def __str__(self):
		return 'Log-odds model<Features: %s; Training set: %s; Positive label: %s; Negative label: %s>'%(str(self.features), str(self.trainingSet), str(self.labelPositive), str(self.labelNegative))
	
	def __repr__(self): return self.__str__()
	
	def getTrainer(self):
		return lambda ts: sequenceModelLogOdds(self.name, self.features, ts, windowSize = self.windowSize, windowStep = self.windowStep, labelPositive = self.labelPositive, labelNegative = self.labelNegative)
	
	def scoreWindow(self, seq):
		cdef list fv, w
		cdef int i
		cdef float ret = 0.0
		w = self.sortedweights
		fv = self.features.getAll(seq)
		for i in range(len(fv)):
			ret += fv[i] * w[i]
		return ret

# Trains a singular-motif PREdictor model with a given set of motifs and positive and negative training sequences.
def trainSinglePREdictorModel(name, motifs, trainingSet, windowSize=500, windowStep=10, labelPositive = positive, labelNegative = negative):
	return sequenceModelLogOdds(name, features.motifSpectrum(motifs), trainingSet, windowSize, windowStep, labelPositive = labelPositive, labelNegative = labelNegative)

# Trains a PREdictor model with a given set of motifs and positive and negative training sequences.
def createDummyPREdictorModel(name, motifs, windowSize=500, windowStep=10):
	return sequenceModelDummy(name, motifs.pairFreq(219), windowSize, windowStep)

# Trains a PREdictor model with a given set of motifs and positive and negative training sequences.
def trainPREdictorModel(name, motifs, trainingSet, windowSize=500, windowStep=10, labelPositive = positive, labelNegative = negative):
	return sequenceModelLogOdds(name, features.motifPairSpectrum(motifs, 219), trainingSet, windowSize, windowStep, labelPositive = labelPositive, labelNegative = labelNegative)

class crossvalidation:
	"""
	Helper class for cross-validations. Accepts binary training and validation sets and constructs cross-validation sets for a desired number of repeats. If a separate validation set is not given, the training set is used. The cross-validation set for each repeat contains numbers of training and test sequences determined by a training-to-testing sequence ratio, as well as a negative-per-positive test sequence ratio. When constructing the validation set, identities are checked for against the training set, to avoid contamination (will not work if sequences are cloned). Holds models and validation statistics. Integrates with terminal and IPython for visualization of results. Stores training and testing data, so new models can be added.
	
	:param models: List of models to cross-validate.
	:param tpos: Positive training set.
	:param tneg: Negative training set.
	:param vpos: Positive validation set.
	:param vneg: Negative validation set.
	:param repeats: Number of experimental repeats. Default = 20.
	:param ratioTrainTest: Ratio of training to testing sequences. Default = 80%.
	:param ratioNegPos: Ratio of validation negatives to positives. Default = 100.
	
	:type models: list
	:type tpos: sequences
	:type tneg: sequences
	:type vpos: sequences, optional
	:type vneg: sequences, optional
	:type repeats: int, optional
	:type ratioTrainTest: float, optional
	:type ratioNegPos: float, optional
	"""
	
	def __init__(self, models, trainingSet, validationSet = None, labelPositive = positive, labelNegative = negative, repeats = 20, ratioTrainTest = 0.8, ratioNegPos = 100.):
		self.models = []
		self.PRC = {}
		self.ROC = {}
		self.labelPositive, self.labelNegative = labelPositive, labelNegative
		self.ratioNegPos = ratioNegPos
		self.ratioTrainTest = ratioTrainTest
		self.trainingSet = trainingSet
		self.validationSet = validationSet
		# Construct cross-validation set
		if validationSet == None:
			validationSet = trainingSet
		tslabels = list(trainingSet.labels())
		tsByLabel = {
			lbl: seqs.sequences[:]
			for lbl, seqs in zip(tslabels, trainingSet.withLabel(tslabels))
		}
		ntrain = int( min(len(tsByLabel[lbl]) for lbl in tsByLabel) * ratioTrainTest )
		vpos, vneg = validationSet.withLabel([ labelPositive, labelNegative ])
		self.cvtrain = []
		self.cvval = []
		self.repeats = repeats
		print('Cross-validation')
		print(' - Training set: %s'%str(trainingSet))
		print(' - Validation set: %s'%str(validationSet))
		print(' - Training sequences per class: %d'%ntrain)
		print(' - Repeats: %d'%repeats)
		print('Generating training/test sets')
		for rep in range(repeats):
			print(' - Repeat %d/%d'%(rep+1, self.repeats))
			# Construct training set as balanced subset of shuffled sequences
			for lbl in tsByLabel:
				random.shuffle(tsByLabel[lbl])
			self.cvtrain.append( sequences('Training set', [
				s
				for lbl in tsByLabel
				for s in tsByLabel[lbl][:ntrain]
			]) )
			# Construct validation set from independent sequences
			rvpos = [ s for s in vpos if s not in self.cvtrain[-1] ]
			rvneg = [ s for s in vneg if s not in self.cvtrain[-1] ]
			random.shuffle(rvpos)
			random.shuffle(rvneg)
			nvpos = int( min(len(rvpos), len(rvneg)/ratioNegPos) )
			nvneg = int(nvpos * ratioNegPos)
			self.cvval.append( sequences('Validation set', rvpos[:nvpos] + rvneg[:nvneg]) )
		# Cross-validate models
		for mdl in models:
			self.addModel(mdl)
	
	"""
	Adds a model to the cross-validation.
	
	:param mdl: Model to add.
	
	:type mdl: model
	"""
	def addModel(self, mdl):
		self.PRC[mdl] = []
		self.ROC[mdl] = []
		print('Cross-validating - ' + mdl.name)
		for rep in range(self.repeats):
			print(' - Repeat %d/%d'%(rep+1, self.repeats))
			imdl = mdl.getTrainer()(self.cvtrain[rep])
			self.PRC[mdl].append(imdl.getPRC(self.cvval[rep], labelPositive = self.labelPositive, labelNegative = self.labelNegative))
			self.ROC[mdl].append(imdl.getROC(self.cvval[rep], labelPositive = self.labelPositive, labelNegative = self.labelNegative))
		self.models.append(mdl)
	
	def plotPRC(self, figsize = (8, 8), outpath = None, style = 'ggplot', returnHTML = False, fontsize = 14, legendLoc = 'lower left'):
		try:
			import matplotlib.pyplot as plt
			import base64
			from io import BytesIO
			from IPython.core.display import display, HTML
			with plt.style.context(style):
				fig = plt.figure(figsize = figsize)
				# Expected random generalization
				positives, negatives = self.cvval[0].withLabel([ self.labelPositive, self.labelNegative ])
				ry = len(positives) / (len(positives) + len(negatives))
				plt.plot(
					[ 0., 1. ],
					[ ry, ry ],
					linestyle = '--',
					color = 'grey',
					label = 'Expected at random')
				# Curves per model
				for mdl in self.models:
					curves = self.PRC[mdl]
					xs = sorted(list(set(pt.x for c in curves for pt in c)))
					curvebyx = [
						{
							x: [ pt.y for pt in curve if pt.x == x ]
							for x in xs
						}
						for curve in curves
					]
					curvebyxmax = {
						x: [
							max(curvebyx[i][x])
							for i, curve in enumerate(curves)
						]
						for x in xs
					}
					curvebyxmin = {
						x: [
							min(curvebyx[i][x])
							for i, curve in enumerate(curves)
						]
						for x in xs
					}
					meanCurve = [
						pt
						for x in xs
						for pt in [
							point2D(x, mean(curvebyxmax[x])),
							point2D(x, mean(curvebyxmin[x]))
						]
					]
					CICurve = [
						pt
						for x in xs
						for pt in [
							point2D(x, CI(curvebyxmax[x])),
							point2D(x, CI(curvebyxmin[x]))
						]
					]
					plt.fill_between(
						[ pt.x for pt in meanCurve ],
						[ pt.y - ci.y for pt, ci in zip(meanCurve, CICurve) ],
						[ pt.y + ci.y for pt, ci in zip(meanCurve, CICurve) ],
						alpha=.3)
					AUCs = [ getAUC(curve) * 100. for curve in curves ]
					plt.plot(
						[ pt.x for pt in meanCurve ],
						[ pt.y for pt in meanCurve ],
						label = '%s - AUC = %.2f +/- %.2f %%'%(mdl.name, mean(AUCs), CI(AUCs)))
				
				plt.xlabel('Recall', fontsize = fontsize)
				plt.ylabel('Precision', fontsize = fontsize)
				plt.legend(loc = legendLoc, fontsize = fontsize, fancybox = True)
				fig.tight_layout()
				if outpath is None:
					bio = BytesIO()
					fig.savefig(bio, format='png')
					plt.close('all')
					encoded = base64.b64encode(bio.getvalue()).decode('utf-8')
					html = '<img src=\'data:image/png;base64,%s\'>'%encoded
					if returnHTML:
						return html
					display(HTML(html))
				else:
					fig.savefig(outpath)
					plt.close('all')
		except ImportError as err:
			raise err
	
	def plotROC(self, figsize = (8, 8), outpath = None, style = 'ggplot', returnHTML = False, fontsize = 14, legendLoc = 'lower right'):
		try:
			import matplotlib.pyplot as plt
			import base64
			from io import BytesIO
			from IPython.core.display import display, HTML
			with plt.style.context(style):
				fig = plt.figure(figsize = figsize)
				# Expected random generalization
				plt.plot(
					[ 0., 1. ],
					[ 0., 1. ],
					linestyle = '--',
					color = 'grey',
					label = 'Expected at random')
				# Curves per model
				for mdl in self.models:
					curves = self.ROC[mdl]
					xs = sorted(list(set(pt.x for c in curves for pt in c)))
					curvebyx = [
						{
							x: [ pt.y for pt in curve if pt.x == x ]
							for x in xs
						}
						for curve in curves
					]
					curvebyxmax = {
						x: [
							max(curvebyx[i][x])
							for i, curve in enumerate(curves)
						]
						for x in xs
					}
					curvebyxmin = {
						x: [
							min(curvebyx[i][x])
							for i, curve in enumerate(curves)
						]
						for x in xs
					}
					meanCurve = [
						pt
						for x in xs
						for pt in [
							point2D(x, mean(curvebyxmax[x])),
							point2D(x, mean(curvebyxmin[x]))
						]
					]
					CICurve = [
						pt
						for x in xs
						for pt in [
							point2D(x, CI(curvebyxmax[x])),
							point2D(x, CI(curvebyxmin[x]))
						]
					]
					plt.fill_between(
						[ pt.x for pt in meanCurve ],
						[ pt.y - ci.y for pt, ci in zip(meanCurve, CICurve) ],
						[ pt.y + ci.y for pt, ci in zip(meanCurve, CICurve) ],
						alpha=.3)
					AUCs = [ getAUC(curve) * 100. for curve in curves ]
					plt.plot(
						[ pt.x for pt in meanCurve ],
						[ pt.y for pt in meanCurve ],
						label = '%s - AUC = %.2f +/- %.2f %%'%(mdl.name, mean(AUCs), CI(AUCs)))
				
				plt.xlabel('False Positive Rate', fontsize = fontsize)
				plt.ylabel('True Positive Rate', fontsize = fontsize)
				plt.legend(loc = legendLoc, fontsize = fontsize, fancybox = True)
				fig.tight_layout()
				if outpath is None:
					bio = BytesIO()
					fig.savefig(bio, format='png')
					plt.close('all')
					encoded = base64.b64encode(bio.getvalue()).decode('utf-8')
					html = '<img src=\'data:image/png;base64,%s\'>'%encoded
					if returnHTML:
						return html
					display(HTML(html))
				else:
					fig.savefig(outpath)
					plt.close('all')
		except ImportError as err:
			raise err
	
	def getConfigurationTable(self):
		return nctable(
			'Configuration',
			{
				'Training set:': [ str(self.trainingSet) ],
				'Validation set:': [ str(self.validationSet) ],
				'Training sequences per repeat': [ len(self.cvtrain[0]) ],
				'Validation sequences per repeat': [ len(self.cvval[0]) ],
				'Repeats:': [ str(self.repeats) ],
				'Negatives per positive:': [ str(self.ratioNegPos) ],
				'Train/test ratio:': [ str(self.ratioTrainTest) ],
			}
		).noCropNames()
	
	def getAUCTable(self):
		return nctable(
			'Evaluation statistics',
			{
				'Model': [ mdl.name for mdl in self.models ],
				'PRC AUC': [
					'%.2f +/- %.2f %%'%(
						mean([getAUC(c) for c in self.PRC[mdl]]) * 100.,
						CI([getAUC(c) for c in self.PRC[mdl]]) * 100.
					)
					for mdl in self.models
				],
				'ROC AUC': [
					'%.2f +/- %.2f %%'%(
						mean([getAUC(c) for c in self.ROC[mdl]]) * 100.,
						CI([getAUC(c) for c in self.ROC[mdl]]) * 100.
					)
					for mdl in self.models
				],
			})
	
	def __repr__(self):
		hdr = 'Cross-validation\n'
		config = self.getConfigurationTable()
		t = self.getAUCTable()
		return hdr + '\n' + config.__repr__() + '\n' + t.__repr__()
	
	def _repr_html_(self):
		hdr = '<div><b>Cross-validation</b></div>'
		config = self.getConfigurationTable()
		t = self.getAUCTable()
		return '<div>' + hdr + config._repr_html_() + t._repr_html_() +\
			'<div style="float: left;">%s</div><div style="float: right;">%s</div></div>'%(
				self.plotPRC(returnHTML = True, figsize = (5., 5.), fontsize = 8),
				self.plotROC(returnHTML = True, figsize = (5., 5.), fontsize = 8)
			) + '</div>'

def crossvalidate(models, trainingSet, validationSet = None, labelPositive = positive, labelNegative = negative, repeats = 20, ratioTrainTest = 0.8, ratioNegPos = 100.):
	return crossvalidation(models = models, trainingSet = trainingSet, validationSet = validationSet, labelPositive = labelPositive, labelNegative = labelNegative, repeats = repeats, ratioTrainTest = ratioTrainTest, ratioNegPos = ratioNegPos)


