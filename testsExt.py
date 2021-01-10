#!/usr/bin/python3
# -*- coding: latin-1 -*-
############################################################################

import random

import gnocis as nc
import gnocis.sklearnModels as sklnc
import gnocis.sklearnModelsOpt as sklncOpt
import time
from os import system

import unittest

# Unit tests that require dependencies not available for CI

#-----------------------------------
# Data

def prep():
	global genome
	global rsA, rsB
	global PcG
	global gwWin
	global PcGTargets
	global PRESeq
	global MC
	global testSeqs
	system('rm -rf ./temp')
	system('mkdir ./temp')
	genome = nc.streamFASTAGZ('tutorial/DmelR5.fasta.gz',
			restrictToSequences = [ '2L', '2R', '3L', '3R', '4', 'X' ])
	#
	rsA = nc.regions( 'A', [
		nc.region('X', 20, 50), nc.region('X', 80, 100), nc.region('X', 150, 300), nc.region('X', 305, 400), nc.region('X', 500, 600),
		nc.region('Y', 40, 100), nc.region('Y', 120, 200)
		] )
	#
	rsB = nc.regions( 'B', [
		nc.region('X', 30, 40), nc.region('X', 90, 120), nc.region('X', 140, 150), nc.region('X', 300, 310),
		nc.region('Y', 40, 100), nc.region('Y', 130, 300), nc.region('Y', 600, 700)
		] )
	#
	# Prepare data set
	#
	PcG = nc.biomarkers('PcG', [
		nc.loadGFFGZ('tutorial/Pc.gff3.gz').deltaResize(1000).rename('Pc'),
		nc.loadGFFGZ('tutorial/Psc.gff3.gz').deltaResize(1000).rename('Psc'),
		nc.loadGFFGZ('tutorial/dRING.gff3.gz').deltaResize(1000).rename('dRING'),
		nc.loadGFFGZ('tutorial/H3K27me3.gff3.gz').rename('H3K27me3'),
	])
	#
	gwWin = nc.getSequenceWindowRegions(
		genome,
		windowSize = 1000, windowStep = 100)
	#
	PcGTargets = PcG.HBMEs(gwWin, threshold = 4)
	#
	PRESeq = PcGTargets.recenter(3000).extract(genome)
	random.shuffle(PRESeq.sequences)
	#
	MC = nc.MarkovChain(trainingSequences = genome, degree = 4, pseudoCounts = 1, addReverseComplements = True)
	#
	testSeqs = nc.sequences('Test', [
		 nc.sequence('X', ''.join( random.choice(['A', 'C', 'G', 'T'])
		 	for _ in range(800) )),
		 nc.sequence('Y', ''.join( random.choice(['A', 'C', 'G', 'T'])
		 	for _ in range(1000) )),
	 ])


#-----------------------------------
# Modelling

# Test

class testModels(unittest.TestCase):
	
	def testOptimizations_sequenceModelSVMOptimizedQuadraticCUDA(self):
		tpos = nc.sequences('PcG/TrxG 1', PRESeq[:int(len(PRESeq)/2)]).label(nc.positive)
		tneg = MC.generateSet(n = int(len(PRESeq)/2), length = 3000).label(nc.negative)
		nSpectrum = 5
		winSize = 500
		winStep = 250
		kDegree = 2
		nc.setNCores(4)
		testset = tpos
		#
		featureSet = nc.features.kSpectrumMM(nSpectrum)
		mdl = sklnc.sequenceModelSVM( name = 'SVM', features = featureSet, trainingSet = tpos + tneg, windowSize = winSize, windowStep = winStep, kDegree = kDegree )
		#
		scoresA = mdl.getSequenceScores(testset)
		#
		featureSet = nc.features.kSpectrumMM(nSpectrum)
		mdl2 = sklncOpt.sequenceModelSVMOptimizedQuadraticCUDA( name = 'SVM',features = featureSet, trainingSet = tpos + tneg, windowSize = winSize, windowStep = winStep, kDegree = kDegree )
		#
		scoresB = mdl2.getSequenceScores(testset)
		diff = sum( 1 if a != b else 0 for a, b in zip(scoresA, scoresB) )
		self.assertEqual(diff, 0)


#-----------------------------------

if __name__ == '__main__':
	prep()
	unittest.main()

