import gnocis as nc
import gnocis.sklearnModels as sklnc
import unittest
from models_data import *

class testModels(unittest.TestCase):
	
	def testSequenceWindows(self):
		tpos = nc.sequences('PcG/TrxG 1', PRESeq[:int(len(PRESeq)/2)]).label(nc.positive)
		tneg = MC.generateSet(n = int(len(PRESeq)/2), length = 3000).label(nc.negative)
		testset = tpos
		nSpectrum = 5
		winSize = 500
		winStep = 250
		kDegree = 2
		featureSet = nc.features.kSpectrumMM(nSpectrum)
		mdl = sklnc.sequenceModelSVM( name = 'SVM', features = featureSet, trainingSet = tpos + tneg, windowSize = winSize, windowStep = winStep, kDegree = kDegree )
		# Single-core
		nc.setNCores(1)
		scoresA = mdl.getSequenceScores(testset)
		# Multi-core
		nc.setNCores(4)
		scoresB = mdl.getSequenceScores(testset)
		#
		diff = sum( 1 if a != b else 0 for a, b in zip(scoresA, scoresB) )
		self.assertEqual(diff, 0)
		# Train second set of models, to ensure old model cannot get stuck
		# in one process
		tpos = nc.sequences('PcG/TrxG 2', PRESeq[int(len(PRESeq)/2):]).label(nc.positive)
		tneg = MC.generateSet(n = len(PRESeq)-int(len(PRESeq)/2), length = 3000).label(nc.negative)
		featureSet = nc.features.kSpectrumMM(nSpectrum)
		mdl2 = sklnc.sequenceModelSVM( name = 'SVM', features = featureSet, trainingSet = tpos + tneg, windowSize = winSize, windowStep = winStep, kDegree = kDegree )
		#
		scoresD = mdl2.getSequenceScores(testset)
		# Single-core
		nc.setNCores(1)
		scoresC = mdl2.getSequenceScores(testset)
		#
		diff = sum( 1 if a != b else 0 for a, b in zip(scoresC, scoresD) )
		self.assertEqual(diff, 0)
	
	def testSequenceScoringWithStream(self):
		Kahn2014Rgn = nc.loadGFF('tutorial/Kahn2014.GFF')
		Kahn2014Seq = Kahn2014Rgn \
				       .recenter(3000) \
				       .extract(genome)
		tneg = MC.generateSet(n = len(Kahn2014Seq), length = 3000)
		trainingSet = Kahn2014Seq.label(nc.positive) + tneg.label(nc.negative)
		PyPREdictor = nc.motifs.Ringrose2003GTGT() \
				       .pairFreq(distCut = 219) \
				       .model(nc.logOdds(
				              labelPositive = nc.positive,
				              labelNegative = nc.negative)
				       ) \
				       .sequenceModel(name = 'PyPREdictor (M2003+GTGT)',
				                      windowSize = 500, windowStep = 250) \
				       .train(trainingSet)
		PyPREdictor.batchsize = 100
		MC.generateSet(n = 1000, length = 3000).saveFASTA('./temp/test.Background.fasta')
		seq1 = nc.streamFASTA('./temp/test.Background.fasta')
		seq2 = nc.loadFASTA('./temp/test.Background.fasta')
		score1 = PyPREdictor.getSequenceScores(seq1, nStreamFetch = 1000)
		score2 = PyPREdictor.getSequenceScores(seq2, nStreamFetch = 1000)
		diff = sum(abs(b - a) for a, b in zip(score1, score2))
		assert(diff == 0.0)
		assert(len(score1) == len(score2))

