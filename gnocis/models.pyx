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
from .sequences import streamSequenceWindows


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
				seq.getWindows(windowSize, windowStep, cache = False)
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
	
	# For pickling (used with multiprocessing)
	def __getstate__(self):
		return self.__dict__
	
	def __setstate__(self, state):
		self.__dict__ = state
	
	# Gets a list of scores for a sequence list or stream. A stream is recommended when the sequence list is large, in order to avoid running out of memory.
	def getSequenceScores(self, seqs, cache = True):
		""" Scores a set of sequences, returning the maximum window score for each.
		
		:param seqs: Sequences to score.
		:param cache: Whether or not to use caching of results.
		
		:type seqs: sequences/sequenceStream
		:type cache: bool
		
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
				self.scoreSequence(cseq, cache=False)
				for blk in seqs.fetch(nTreadFetch, maxThreadFetchNT)
				for cseq in blk
			]
		elif isinstance(seqs, sequences) or isinstance(seqs, list):
			return [ self.scoreSequence(cseq, cache=cache) for cseq in seqs ]
	
	# Gets the threshold that gives optimal accuracy on a pair of lists or streams of sequences. Streams are recommended when the sequence lists are large, in order to avoid running out of memory.
	def getOptimalAccuracyThreshold(self, positives, negatives, cache = False):
		""" Gets a threshold value optimized for accuracy to a set of positive and a set of negative sequences.
		
		:param positives: Positive sequences.
		:param negatives: Negative sequences.
		:param cache: Whether or not to use caching of results.
		
		:type positives: sequences/sequenceStream
		:type negatives: sequences/sequenceStream
		:type cache: bool
		
		:return: List of the maximum window score per sequence
		:rtype: list
		"""
		positiveScores = [ validationPair(score = score, label = True) for score in self.getSequenceScores(positives, cache = cache) ]
		negativeScores = [ validationPair(score = score, label = False) for score in self.getSequenceScores(negatives, cache = cache) ]
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
	def getPrecisionThreshold(self, positives, negatives, wantedPrecision, cache = False):
		""" Gets a threshold value for a desired precision to a set of positive and a set of negative sequences. Linear interpolation is used in order to achieve a close approximation.
		
		:param positives: Positive sequences.
		:param negatives: Negative sequences.
		:param wantedPrecision: The precision to approximate.
		:param cache: Whether or not to use caching of results.
		
		:type positives: sequences/sequenceStream
		:type negatives: sequences/sequenceStream
		:type wantedPrecision: float
		:type cache: bool
		
		:return: List of the maximum window score per sequence
		:rtype: list
		"""
		positiveScores = [ validationPair(score = score, label = True) for score in self.getSequenceScores(positives, cache = cache) ]
		negativeScores = [ validationPair(score = score, label = False) for score in self.getSequenceScores(negatives, cache = cache) ]
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
		seqLens = genome.getSequenceLengths()
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
	def scoreSequence(self, seq, cache=True):
		""" Scores a single sequence. The score is determined by applying the model with a sliding window, and taking the maximum score.
		
		:param seq: The sequence.
		:param cache: Whether or not to use caching of results.
		
		:type seq: sequences/sequenceStream
		:type cache: bool
		
		:return: Maximal window score
		:rtype: float
		"""
		cdef float score
		if cache:
			if not 'score' in seq.cache.keys():
				seq.cache['score'] = {}
			if self in seq.cache['score'].keys():
				return seq.cache['score'][self]
		score = max(
			(
				self.scoreWindow(win)
				for win in
				seq.getWindows(self.windowSize, self.windowStep)
				if len(win) == self.windowSize
			),
			default = float('-INF')
		)
		if cache:
			seq.cache['score'][self] = score
		return score
	
	# Gets and returns a confusion matrix.
	def getConfusionMatrix(self, positives, negatives, cache = True):
		""" Calculates and returns a confusion matrix for sets of positive and negative sequences.
		
		:param positives: Positives.
		:param negatives: Negatives.
		:param cache: Whether or not to use caching of results.
		
		:type positives: sequences/sequenceStream
		:type negatives: sequences/sequenceStream
		:type cache: bool
		
		:return: Confusion matrix
		:rtype: dict
		"""
		vPos = [ validationPair(score = score, label = True) for score in self.getSequenceScores(positives, cache = cache) ]
		vNeg = [ validationPair(score = score, label = False) for score in self.getSequenceScores(negatives, cache = cache) ]
		return getConfusionMatrix(vPos, vNeg, threshold = self.threshold)
	
	# Generates a Receiver Operating Characteristic curve.
	def getROC(self, positives, negatives, cache = True):
		""" Calculates and returns a Receiver Operating Characteristic (ROC) curve for sets of positive and negative sequences.
		
		:param positives: Positives.
		:param negatives: Negatives.
		:param cache: Whether or not to use caching of results.
		
		:type positives: sequences/sequenceStream
		:type negatives: sequences/sequenceStream
		:type cache: bool
		
		:return: Receiver Operating Characteristic (ROC) curve
		:rtype: list
		"""
		vPos = [ validationPair(score = score, label = True) for score in self.getSequenceScores(positives, cache = cache) ]
		vNeg = [ validationPair(score = score, label = False) for score in self.getSequenceScores(negatives, cache = cache) ]
		return getROC(vPos, vNeg)
	
	# Calculates the area under a Receiver Operating Characteristic curve.
	def getROCAUC(self, positives, negatives):
		""" Calculates and returns the area under a Receiver Operating Characteristic (ROCAUC) curve for sets of positive and negative sequences.
		
		:param positives: Positives.
		:param negatives: Negatives.
		
		:type positives: sequences/sequenceStream
		:type negatives: sequences/sequenceStream
		
		:return: Area under the Receiver Operating Characteristic (ROCAUC) curve
		:rtype: list
		"""
		return getAUC(self.getROC(positives, negatives))
	
	# Generates a Precision/Recall curve.
	def getPRC(self, positives, negatives, cache = True):
		""" Calculates and returns a Precision/Recall curve (PRC) for sets of positive and negative sequences.
		
		:param positives: Positives.
		:param negatives: Negatives.
		:param cache: Whether or not to use caching of results.
		
		:type positives: sequences/sequenceStream
		:type negatives: sequences/sequenceStream
		:type cache: bool
		
		:return: Precision/Recall curve (PRC)
		:rtype: list
		"""
		vPos = [ validationPair(score = score, label = True) for score in self.getSequenceScores(positives, cache = cache) ]
		vNeg = [ validationPair(score = score, label = False) for score in self.getSequenceScores(negatives, cache = cache) ]
		return getPRC(vPos, vNeg)
	
	# Calculates the area under a Precision/Recall Curve.
	def getPRCAUC(self, positives, negatives):
		""" Calculates and returns the area under a Precision/Recall curve (PRCAUC) for sets of positive and negative sequences.
		
		:param positives: Positives.
		:param negatives: Negatives.
		
		:type positives: sequences/sequenceStream
		:type negatives: sequences/sequenceStream
		
		:return: Area under the Precision/Recall curve (PRCAUC)
		:rtype: list
		"""
		return getAUC(self.getPRC(positives, negatives))
	
	# Returns validation statistics dictionary.
	def getValidationStatistics(self, positives, negatives):
		""" Returns common model validation statistics (confusion matrix values; ROCAUC; PRCAUC).
		
		:param positives: Positives.
		:param negatives: Negatives.
		
		:type positives: sequences/sequenceStream
		:type negatives: sequences/sequenceStream
		
		:return: Validation statistics
		:rtype: list
		"""
		CMStats = getConfusionMatrixStatistics( self.getConfusionMatrix(positives, negatives) )
		stats = {
			'title': 'Validation', 'positives': positives, 'negatives': negatives, 'model': self,
			'ROC AUC': self.getROCAUC(positives, negatives),
			'PRC AUC': self.getPRCAUC(positives, negatives),
		}
		for k in CMStats.keys():
			stats[k] = CMStats[k]
		return stats
	
	# Returns cross-validation statistics dictionary, with random training set splitting and training and testing on independent halves.
	def getCrossValidationStatisticsRandomHalves(self, positives, negatives, nfolds):
		cpos, cneg = positives[:], negatives[:]
		npos, nneg = len(positives), len(negatives)
		stats = { 'title': 'Cross-validation (random halves, %d folds)'%nfolds, 'positives': positives, 'negatives': negatives, 'model': self }
		foldStats = []
		for i in range(nfolds):
			random.shuffle(cpos)
			random.shuffle(cneg)
			tpos, tneg = cpos[:int(npos/2)], cneg[:int(nneg/2)]
			vpos, vneg = cpos[int(npos/2):], cneg[int(nneg/2):]
			cvmdl = self.getCrossValidationConstructor()(tpos, tneg)
			foldStats.append({
				'ROC AUC': cvmdl.getROCAUC(vpos, vneg),
				'PRC AUC': cvmdl.getPRCAUC(vpos, vneg),
			})
			CMStats = getConfusionMatrixStatistics( cvmdl.getConfusionMatrix(vpos, vneg) )
			for k in CMStats.keys():
				foldStats[-1][k] = CMStats[k]
		for stat in foldStats[0].keys():
			stats[stat] = sum(v[stat] for v in foldStats) / nfolds
		return stats
	
	# Returns cross-validation statistics dictionary, with leaving each training set sequence out and classifying it.
	def getCrossValidationStatisticsLOO(self, positives, negatives):
		cpos, cneg = positives[:], negatives[:]
		npos, nneg = len(positives), len(negatives)
		vPairs = [  ]
		TP, FP, TN, FN = 0, 0, 0, 0
		for i, e in enumerate(positives):
			cvmdl = self.getCrossValidationConstructor()(positives[:i] + positives[i+1:], negatives)
			score = cvmdl.scoreSequence(positives[i])
			if score > cvmdl.threshold:
				TP += 1
			else:
				FN += 1
			vPairs.append( { 'score': score, 'class': True } )
		for i, e in enumerate(negatives):
			cvmdl = self.getCrossValidationConstructor()(positives, negatives[:i] + negatives[i+1:])
			score = cvmdl.scoreSequence(negatives[i])
			if score > cvmdl.threshold:
				FP += 1
			else:
				TN += 1
			vPairs.append( { 'score': score, 'class': False } )
		stats = { 'title': 'Cross-validation (leave one out)', 'positives': positives, 'negatives': negatives, 'model': self }
		CMStats = getConfusionMatrixStatistics({ 'TP':TP, 'FP':FP, 'TN':TN, 'FN':FN })
		for k in CMStats.keys():
			stats[k] = CMStats[k]
		vPairs = sorted(vPairs, key = lambda x: -x['score'])
		TP, FP, AUC = 0, 0, 0.0
		ROCcurve = [ { 'x': 0.0, 'y': 0.0 } ]
		PRCcurve = [ { 'x': 0.0, 'y': 0.0 } ]
		for vp in vPairs:
			if vp['class']:
				TP += 1
			else:
				FP += 1
			ROCcurve .append({ 'x': (FP / len( negatives )), 'y': (TP / float(len(positives))) })
			PRCcurve .append({ 'x': (TP / float(len(positives))), 'y': (TP / float( TP + FP )) })
		ROCcurve.append({ 'x': 1.0, 'y':1.0 })
		PRCcurve.append({ 'x': 1.0, 'y':(len(positives)/float(len(positives)+len(negatives))) })
		stats['ROC AUC'] = getAUC(ROCcurve)
		stats['PRC AUC'] = getAUC(PRCcurve)
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
		
		print('Handling scores')
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
	def printTestStatistics(self, positives, negatives):
		printValidationStatistics( self.getValidationStatistics(positives, negatives) )

# Uniformly weighted model
class sequenceModelDummy(sequenceModel):
	"""
	This model takes a feature set, and scores input sequences by summing feature values, without weighting.
	
	:param features: The feature set.
	:param windowSize: Window size to use.
	:param windowStep: Window step size to use.
	
	:type features: features
	:type windowSize: int
	:type windowStep: int
	"""
	
	def __init__(self, features, windowSize, windowStep):
		self.threshold = 0.0
		self.features = features
		self.windowSize, self.windowStep = windowSize, windowStep
	
	def __str__(self):
		return 'Dummy model<Features: %s>'%(str(self.features))
	
	def __repr__(self): return self.__str__()
	
	def getCrossValidationConstructor(self):
		return lambda pos, neg: sequenceModelDummy(self.features, self.windowSize, self.windowStep)
	
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
	
	:param features: The feature set.
	:param trainingPositives: Positive training sequences.
	:param trainingNegatives: Negative training sequences.
	:param windowSize: Window size to use.
	:param windowStep: Window step size to use.
	
	:type features: features
	:type trainingPositives: sequences
	:type trainingNegatives: sequences
	:type windowSize: int
	:type windowStep: int
	"""
	
	def __init__(self, features, trainingPositives, trainingNegatives, windowSize, windowStep):
		self.threshold = 0.0
		self.features = features
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
		return 'Log-odds model<Features: %s; Positives: %s; Negatives: %s>'%(str(self.features), str(self.trainingPositives), str(self.trainingNegatives))
	
	def __repr__(self): return self.__str__()
	
	def getCrossValidationConstructor(self):
		return lambda pos, neg: sequenceModelLogOdds(self.features, pos, neg, self.windowSize, self.windowStep)
	
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
def trainSinglePREdictorModel(motifs, positives, negatives, windowSize=500, windowStep=10):
	return sequenceModelLogOdds(features.getMotifSpectrum(motifs), positives, negatives, windowSize, windowStep)

# Trains a PREdictor model with a given set of motifs and positive and negative training sequences.
def createDummyPREdictorModel(motifs, windowSize=500, windowStep=10):
	return sequenceModelDummy(features.getPREdictorMotifPairSpectrum(motifs, 219), windowSize, windowStep)

# Trains a PREdictor model with a given set of motifs and positive and negative training sequences.
def trainPREdictorModel(motifs, positives, negatives, windowSize=500, windowStep=10):
	return sequenceModelLogOdds(features.getPREdictorMotifPairSpectrum(motifs, 219), positives, negatives, windowSize, windowStep)

