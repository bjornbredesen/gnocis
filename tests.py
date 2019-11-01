#!/usr/bin/python3
# -*- coding: latin-1 -*-
############################################################################

import random

import gnocis as cis
import gnocis.sklearnModels as sklcis
import time

import unittest

#-----------------------------------
# Regions

rsA = cis.regions( 'A', [
	cis.region('X', 20, 50), cis.region('X', 80, 100), cis.region('X', 150, 300), cis.region('X', 305, 400), cis.region('X', 500, 600),
	cis.region('Y', 40, 100), cis.region('Y', 120, 200)
	] )

rsB = cis.regions( 'B', [
	cis.region('X', 30, 40), cis.region('X', 90, 120), cis.region('X', 140, 150), cis.region('X', 300, 310),
	cis.region('Y', 40, 100), cis.region('Y', 130, 300), cis.region('Y', 600, 700)
	] )

def getRegionsStr(rs):
	return '; '.join( r.bstr() for r in rs )

class testRegions(unittest.TestCase):
	
	def testMerge(self):
		self.assertEqual( getRegionsStr( rsA.getMerged(rsB) ),
			'X:20..50 (+); X:80..120 (+); X:140..400 (+); X:500..600 (+); Y:40..100 (+); Y:120..300 (+); Y:600..700 (+)' )
	
	def testIntersection(self):
		self.assertEqual( getRegionsStr( rsA.getIntersection(rsB) ),
			'X:30..40 (+); X:90..100 (+); X:150..150 (+); X:300..300 (+); X:305..310 (+); Y:40..100 (+); Y:130..200 (+)' )
	
	def testExcluded(self):
		self.assertEqual( getRegionsStr( rsA.getExcluded(rsB) ),
			'X:20..29 (+); X:41..50 (+); X:80..89 (+); X:151..299 (+); X:311..400 (+); X:500..600 (+); Y:120..129 (+)' )
	
	def testOverlap(self):
		self.assertEqual( getRegionsStr( rsA.getOverlap(rsB) ),
			'X:20..50 (+); X:80..100 (+); X:150..300 (+); X:305..400 (+); Y:40..100 (+); Y:120..200 (+)' )
	
	def testNonOverlap(self):
		self.assertEqual( getRegionsStr( rsA.getNonOverlap(rsB) ),
			'X:500..600 (+)' )
	
	def testSaveLoadGFF(self):
		rsA.saveGFF('tmp/test.GFF')
		self.assertEqual( getRegionsStr( rsA ),
			getRegionsStr( cis.loadGFF('tmp/test.GFF') ) )
	
	def testSaveLoadBED(self):
		rsA.saveBED('tmp/test.BED')
		self.assertEqual( getRegionsStr( rsA ),
			getRegionsStr( cis.loadBED('tmp/test.BED') ) )


#-----------------------------------
# Sequences

testSeqs = cis.sequences('Test', [
		 cis.sequence('X', ''.join( random.choice(['A', 'C', 'G', 'T']) for _ in range(800) )),
		 cis.sequence('Y', ''.join( random.choice(['A', 'C', 'G', 'T']) for _ in range(1000) )),
	 ])

class testSequences(unittest.TestCase):
	
	def testSaveLoadFASTA(self):
		testSeqs.saveFASTA('tmp/test.fasta')
		sA = '\n'.join( '%s: %s'%(s.name, s.seq) for s in testSeqs )
		sB = '\n'.join( '%s: %s'%(s.name.split(' from FASTA file')[0], s.seq) for s in cis.loadFASTA('tmp/test.fasta') )
		self.assertEqual(
			sA, 
			sB )
	
	def testGeneratingSequences(self):
		MC = cis.generatorMarkovChain(trainingSequences = cis.streamFASTAGZ('./tutorial/dmel-all-chromosome-r5.57.fasta.gz'), degree = 4, pseudoCounts = 1, addReverseComplements = True)
		MC.generateSet(n = 50, length = 3000).saveFASTA('./temp/test.Background.fasta')
	
	def testExtractRegionSequences(self):
		testSeqs.saveFASTA('temp/test.fasta')
		seqByName = { seq.name: seq.seq for seq in testSeqs }
		# Ensure that saving and re-loading sequences yields identical sequences
		seqs = rsA.extractSequences( cis.loadFASTA( 'temp/test.fasta' ) )
		self.assertEqual(
			[ s.seq for s in seqs ], 
			[ seqByName[r.seq][ r.start : r.end+1 ] for r in rsA ] )
		seqs = rsA.extractSequences( cis.streamFASTA( 'temp/test.fasta' ) )
		self.assertEqual(
			[ s.seq for s in seqs ], 
			[ seqByName[r.seq][ r.start : r.end+1 ] for r in rsA ] )
		# Ensure that streaming of short blocks yields identical final sequences
		seqs = rsA.extractSequences( cis.streamFASTA( 'temp/test.fasta', wantBlockSize = 50 ) )
		self.assertEqual(
			[ s.seq for s in seqs ], 
			[ seqByName[r.seq][ r.start : r.end+1 ] for r in rsA ] )
	
	def testSequenceWindows(self):
		testSeqs.saveFASTA('temp/test.fasta')
		windows = [ w for w in cis.streamSequenceWindows(cis.streamFASTA('temp/test.fasta'), 40, 20) ]
		# Ensure that window boundaries are correct (the last index is inclusive)
		self.assertEqual(
			windows[1].sourceRegion.bstr(),
			'X:20..59 (+)' )
		# Ensure that the window sequence is correct
		self.assertEqual(
			windows[1].seq,
			testSeqs[0].seq[20:59+1] )
		# Ensure that all sequences are covered
		self.assertEqual(
			'; '.join( sorted( set( s.name.split(':')[0] for s in windows ) ) ),
			'X; Y' )
	
	def testSequenceStreamBlockFetchingAndSplitting(self):
		nTreadFetch = 20000
		maxThreadFetchNT = nTreadFetch * 1000
		nCoresUse = 4
		# Get regular sequence names
		testset = cis.streamFASTA('./temp/test.Background.fasta')
		snA = [
			seq.name
			for seq in testset.streamFullSequences()
		]
		# Get block-fetched sequence names
		testset = cis.streamFASTA('./temp/test.Background.fasta')
		snB = [
			seq.name
			for blk in testset.fetch(nTreadFetch, maxThreadFetchNT)
			for seq in blk
		]
		# Get block-fetched, core-split sequence names
		testset = cis.streamFASTA('./temp/test.Background.fasta')
		snC = [
			seq.name
			for blk in testset.fetch(nTreadFetch, maxThreadFetchNT)
			for coresplit in blk.split(nCoresUse)
			for seq in coresplit
		]
		# Check name differences
		self.assertEqual( snA, snB )
		self.assertEqual( snA, snC )
		self.assertEqual( snB, snC )


#-----------------------------------
# Modelling

# Prepare data set

PcG = cis.biomarkers('PcG', [
	cis.loadGFFGZ('temp/Pc.gff3.gz').getDeltaResized(1000).getRenamed('Pc'),
	cis.loadGFFGZ('temp/Psc.gff3.gz').getDeltaResized(1000).getRenamed('Psc'),
	cis.loadGFFGZ('temp/dRING.gff3.gz').getDeltaResized(1000).getRenamed('dRING'),
	cis.loadGFFGZ('temp/H3K27me3.gff3.gz').getRenamed('H3K27me3'),
])

gwWin = cis.getSequenceWindowRegions(
	cis.streamFASTA('demo_data/dmel.fa',
		restrictToSequences = [ '2L', '2R', '3L', '3R', '4', 'X' ]),
	windowSize = 1000, windowStep = 100)

PcGTargets = PcG.getHBMEs(gwWin, threshold = 4)

PRESeq = PcGTargets.getRandomlyRecentered(3000).extractSequencesFromFASTA('./demo_data/dmel.fa')
import random
random.shuffle(PRESeq.sequences)
cis.sequences('PcG/TrxG 1', PRESeq[:int(len(PRESeq)/2)]).saveFASTA('./temp/PcGTrxG1.fasta')
cis.sequences('PcG/TrxG 2', PRESeq[int(len(PRESeq)/2):]).saveFASTA('./temp/PcGTrxG2.fasta')

MC = cis.generatorMarkovChain(trainingSequences = cis.streamFASTA('./demo_data/dmel.fa'), degree = 4, pseudoCounts = 1, addReverseComplements = True)

MC.generateSet(n = int(len(PRESeq)/2), length = 3000).saveFASTA('./temp/NonPcGTrxG1.fasta')
MC.generateSet(n = len(PRESeq)-int(len(PRESeq)/2), length = 3000).saveFASTA('./temp/NonPcGTrxG2.fasta')

# Test

class testModels(unittest.TestCase):
	
	def testSequenceWindows(self):
		testsetPath = './temp/PcGTrxG1.fasta'
		tpos = cis.loadFASTA('./temp/PcGTrxG1.fasta')
		tneg = cis.loadFASTA('./temp/NonPcGTrxG1.fasta')
		nSpectrum = 5
		winSize = 500
		winStep = 250
		kDegree = 2
		featureSet = cis.featureScaler( cis.features.getKSpectrumMM(nSpectrum), positives = tpos, negatives = tneg )
		mdl = sklcis.sequenceModelSVM( features = featureSet, positives = tpos, negatives = tneg, windowSize = winSize, windowStep = winStep, kDegree = kDegree )
		# Single-core
		cis.setNCores(1)
		testset = cis.streamFASTA(testsetPath)
		scoresA = mdl.getSequenceScores(testset)
		# Multi-core
		cis.setNCores(4)
		testset = cis.streamFASTA(testsetPath)
		scoresB = mdl.getSequenceScores(testset)
		#
		diff = sum( 1 if a != b else 0 for a, b in zip(scoresA, scoresB) )
		self.assertEqual(diff, 0)
		# Train second set of models, to ensure old model cannot get stuck
		# in one process
		tpos = cis.loadFASTA('./temp/PcGTrxG2.fasta')
		tneg = cis.loadFASTA('./temp/NonPcGTrxG2.fasta')
		featureSet = cis.featureScaler( cis.features.getKSpectrumMM(nSpectrum), positives = tpos, negatives = tneg )
		mdl2 = sklcis.sequenceModelSVM( features = featureSet, positives = tpos, negatives = tneg, windowSize = winSize, windowStep = winStep, kDegree = kDegree )
		#
		testset = cis.streamFASTA(testsetPath)
		scoresD = mdl2.getSequenceScores(testset)
		# Single-core
		cis.setNCores(1)
		testset = cis.streamFASTA(testsetPath)
		scoresC = mdl2.getSequenceScores(testset)
		#
		diff = sum( 1 if a != b else 0 for a, b in zip(scoresC, scoresD) )
		self.assertEqual(diff, 0)


#-----------------------------------

if __name__ == '__main__':
	unittest.main()

