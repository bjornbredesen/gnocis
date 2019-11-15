#!/usr/bin/python3
# -*- coding: latin-1 -*-
############################################################################

import random

import gnocis as nc
import gnocis.sklearnModels as sklnc
import time
from os import system

import unittest

system('rm -rf ./temp')
system('mkdir ./temp')

#-----------------------------------
# Data

genome = nc.streamFASTAGZ('tutorial/DmelR5.fasta.gz',
		restrictToSequences = [ '2L', '2R', '3L', '3R', '4', 'X' ])

#-----------------------------------
# Regions

rsA = nc.regions( 'A', [
	nc.region('X', 20, 50), nc.region('X', 80, 100), nc.region('X', 150, 300), nc.region('X', 305, 400), nc.region('X', 500, 600),
	nc.region('Y', 40, 100), nc.region('Y', 120, 200)
	] )

rsB = nc.regions( 'B', [
	nc.region('X', 30, 40), nc.region('X', 90, 120), nc.region('X', 140, 150), nc.region('X', 300, 310),
	nc.region('Y', 40, 100), nc.region('Y', 130, 300), nc.region('Y', 600, 700)
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
		rsA.saveGFF('temp/test.GFF')
		self.assertEqual( getRegionsStr( rsA ),
			getRegionsStr( nc.loadGFF('temp/test.GFF') ) )
	
	def testSaveLoadBED(self):
		rsA.saveBED('temp/test.BED')
		self.assertEqual( getRegionsStr( rsA ),
			getRegionsStr( nc.loadBED('temp/test.BED') ) )


#-----------------------------------
# Sequences

testSeqs = nc.sequences('Test', [
		 nc.sequence('X', ''.join( random.choice(['A', 'C', 'G', 'T']) for _ in range(800) )),
		 nc.sequence('Y', ''.join( random.choice(['A', 'C', 'G', 'T']) for _ in range(1000) )),
	 ])

class testSequences(unittest.TestCase):
	
	def testSaveLoadFASTA(self):
		testSeqs.saveFASTA('temp/test.fasta')
		sA = '\n'.join( '%s: %s'%(s.name, s.seq) for s in testSeqs )
		sB = '\n'.join( '%s: %s'%(s.name.split(' from FASTA file')[0], s.seq) for s in nc.loadFASTA('temp/test.fasta') )
		self.assertEqual(
			sA, 
			sB )
	
	def testGeneratingSequences(self):
		MC = nc.generatorMarkovChain(trainingSequences = genome, degree = 4, pseudoCounts = 1, addReverseComplements = True)
		MC.generateSet(n = 50, length = 3000).saveFASTA('./temp/test.Background.fasta')
	
	def testExtractRegionSequences(self):
		testSeqs.saveFASTA('temp/test.fasta')
		seqByName = { seq.name: seq.seq for seq in testSeqs }
		# Ensure that saving and re-loading sequences yields identical sequences
		seqs = rsA.extractSequences( nc.loadFASTA( 'temp/test.fasta' ) )
		self.assertEqual(
			[ s.seq for s in seqs ], 
			[ seqByName[r.seq][ r.start : r.end+1 ] for r in rsA ] )
		seqs = rsA.extractSequences( nc.streamFASTA( 'temp/test.fasta' ) )
		self.assertEqual(
			[ s.seq for s in seqs ], 
			[ seqByName[r.seq][ r.start : r.end+1 ] for r in rsA ] )
		# Ensure that streaming of short blocks yields identical final sequences
		seqs = rsA.extractSequences( nc.streamFASTA( 'temp/test.fasta', wantBlockSize = 50 ) )
		self.assertEqual(
			[ s.seq for s in seqs ], 
			[ seqByName[r.seq][ r.start : r.end+1 ] for r in rsA ] )
	
	def testSequenceWindows(self):
		testSeqs.saveFASTA('temp/test.fasta')
		windows = [ w for w in nc.streamSequenceWindows(nc.streamFASTA('temp/test.fasta'), 40, 20) ]
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
		testset = nc.streamFASTA('./temp/test.Background.fasta')
		snA = [
			seq.name
			for seq in testset.streamFullSequences()
		]
		# Get block-fetched sequence names
		testset = nc.streamFASTA('./temp/test.Background.fasta')
		snB = [
			seq.name
			for blk in testset.fetch(nTreadFetch, maxThreadFetchNT)
			for seq in blk
		]
		# Get block-fetched, core-split sequence names
		testset = nc.streamFASTA('./temp/test.Background.fasta')
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

PcG = nc.biomarkers('PcG', [
	nc.loadGFFGZ('tutorial/Pc.gff3.gz').getDeltaResized(1000).getRenamed('Pc'),
	nc.loadGFFGZ('tutorial/Psc.gff3.gz').getDeltaResized(1000).getRenamed('Psc'),
	nc.loadGFFGZ('tutorial/dRING.gff3.gz').getDeltaResized(1000).getRenamed('dRING'),
	nc.loadGFFGZ('tutorial/H3K27me3.gff3.gz').getRenamed('H3K27me3'),
])

gwWin = nc.getSequenceWindowRegions(
	genome,
	windowSize = 1000, windowStep = 100)

PcGTargets = PcG.getHBMEs(gwWin, threshold = 4)

PRESeq = PcGTargets.getRandomlyRecentered(3000).extract(genome)
import random
random.shuffle(PRESeq.sequences)
nc.sequences('PcG/TrxG 1', PRESeq[:int(len(PRESeq)/2)]).saveFASTA('./temp/PcGTrxG1.fasta')
nc.sequences('PcG/TrxG 2', PRESeq[int(len(PRESeq)/2):]).saveFASTA('./temp/PcGTrxG2.fasta')

MC = nc.generatorMarkovChain(trainingSequences = genome, degree = 4, pseudoCounts = 1, addReverseComplements = True)

MC.generateSet(n = int(len(PRESeq)/2), length = 3000).saveFASTA('./temp/NonPcGTrxG1.fasta')
MC.generateSet(n = len(PRESeq)-int(len(PRESeq)/2), length = 3000).saveFASTA('./temp/NonPcGTrxG2.fasta')

# Test

class testModels(unittest.TestCase):
	
	def testSequenceWindows(self):
		testsetPath = './temp/PcGTrxG1.fasta'
		tpos = nc.loadFASTA('./temp/PcGTrxG1.fasta')
		tneg = nc.loadFASTA('./temp/NonPcGTrxG1.fasta')
		nSpectrum = 5
		winSize = 500
		winStep = 250
		kDegree = 2
		featureSet = nc.featureScaler( nc.features.getKSpectrumMM(nSpectrum), positives = tpos, negatives = tneg )
		mdl = sklnc.sequenceModelSVM( features = featureSet, positives = tpos, negatives = tneg, windowSize = winSize, windowStep = winStep, kDegree = kDegree )
		# Single-core
		nc.setNCores(1)
		testset = nc.streamFASTA(testsetPath)
		scoresA = mdl.getSequenceScores(testset)
		# Multi-core
		nc.setNCores(4)
		testset = nc.streamFASTA(testsetPath)
		scoresB = mdl.getSequenceScores(testset)
		#
		diff = sum( 1 if a != b else 0 for a, b in zip(scoresA, scoresB) )
		self.assertEqual(diff, 0)
		# Train second set of models, to ensure old model cannot get stuck
		# in one process
		tpos = nc.loadFASTA('./temp/PcGTrxG2.fasta')
		tneg = nc.loadFASTA('./temp/NonPcGTrxG2.fasta')
		featureSet = nc.featureScaler( nc.features.getKSpectrumMM(nSpectrum), positives = tpos, negatives = tneg )
		mdl2 = sklnc.sequenceModelSVM( features = featureSet, positives = tpos, negatives = tneg, windowSize = winSize, windowStep = winStep, kDegree = kDegree )
		#
		testset = nc.streamFASTA(testsetPath)
		scoresD = mdl2.getSequenceScores(testset)
		# Single-core
		nc.setNCores(1)
		testset = nc.streamFASTA(testsetPath)
		scoresC = mdl2.getSequenceScores(testset)
		#
		diff = sum( 1 if a != b else 0 for a, b in zip(scoresC, scoresD) )
		self.assertEqual(diff, 0)
	
	def testKSpectrum(self):
		spec = nc.features.getKSpectrum(5)
		testSeq = nc.sequence('', 'GAGAGAGAGT')
		kmerFreq = {
			k: v
			for k, v in zip( [ f.kmer for f in spec ],
					spec.getAll(testSeq)
			)
		}
		self.assertEqual(kmerFreq['AGAGT'], 100.)
		self.assertEqual(kmerFreq['ACTCT'], 100.)
		self.assertEqual(kmerFreq['AGAGA'], 200.)
		self.assertEqual(kmerFreq['TCTCT'], 200.)
		self.assertEqual(kmerFreq['GAGAG'], 300.)
		self.assertEqual(kmerFreq['CTCTC'], 300.)
		self.assertEqual(
			sum(
				kmerFreq[kmer]
				for kmer in kmerFreq.keys()
				if kmer not in [ 'AGAGT', 'ACTCT', 'AGAGA', 'TCTCT', 'GAGAG', 'CTCTC' ]),
			0.)
		# Ensure that application to the reverse complement yields an identical spectrum
		specmotifs = set(
			k
			for k, f in zip(
				[ f.kmer for f in spec ],
				spec.getAll(testSeq)
			)
			if f > 0
		)
		specmotifsRC = set(
			k
			for k, f in zip(
				[ f.kmer for f in spec ],
				spec.getAll(nc.sequence('',
					nc.getReverseComplementaryDNASequence(testSeq.seq)))
			)
			if f > 0
		)
		self.assertEqual(specmotifs, specmotifsRC)
	
	def testKSpectrumMM(self):
		spec = nc.features.getKSpectrumMM(5)
		testSeq = nc.sequence('', 'GAGAGAGAGT')
		motifs = set([ testSeq.seq[i:i+5] for i in range(len(testSeq)-4) ])
		motifs = motifs | set( nc.getReverseComplementaryDNASequence(m) for m in motifs )
		def getMM(m):
			return set(
				m[:i] + mut + m[i+1:]
				for i in range(len(m))
				for mut in [ 'A', 'C', 'G', 'T' ]
			)
		motifs = motifs | set( mm for m in motifs for mm in getMM(m) )
		# Ensure that only the main motif and mismatches are registered
		specmotifs = set(
			k
			for k, f in zip(
				[ f.kmer for f in spec ],
				spec.getAll(testSeq)
			)
			if f > 0
		)
		self.assertEqual(specmotifs, motifs)
		# Ensure that application to the reverse complement yields an identical spectrum
		specmotifsRC = set(
			k
			for k, f in zip(
				[ f.kmer for f in spec ],
				spec.getAll(nc.sequence('',
					nc.getReverseComplementaryDNASequence(testSeq.seq)))
			)
			if f > 0
		)
		self.assertEqual(specmotifs, specmotifsRC)
	
	def testKSpectrumMMD(self):
		spec = nc.features.getKSpectrumMMD(5)
		testSeq = nc.sequence('', 'GAGAGAGAGT')
		motifs = set([ testSeq.seq[i:i+5] for i in range(len(testSeq)-4) ])
		motifs = motifs | set( nc.getReverseComplementaryDNASequence(m) for m in motifs )
		def getMM(m):
			return set(
				m[:i] + mut + m[i+1:]
				for i in range(len(m))
				for mut in [ 'A', 'C', 'G', 'T' ]
			)
		motifs = motifs | set( mm for m in motifs for mm in getMM(m) )
		# Ensure that only the main motif and mismatches are registered
		specmotifs = set(
			k
			for k, f in zip(
				[ f.kmer for f in spec ],
				spec.getAll(testSeq)
			)
			if f > 0
		)
		self.assertEqual(specmotifs, motifs)


#-----------------------------------

if __name__ == '__main__':
	unittest.main()

