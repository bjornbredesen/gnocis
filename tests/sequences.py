import gnocis as nc
import unittest
from regions_data import *
from sequences_data import *

class testSequences(unittest.TestCase):
	
	def testSaveLoadFASTA(self):
		testSeqs.saveFASTA('temp/test.fasta')
		sA = '\n'.join( '%s: %s'%(s.name, s.seq) for s in testSeqs )
		sB = '\n'.join( '%s: %s'%(s.name.split(' from FASTA file')[0], s.seq) for s in nc.loadFASTA('temp/test.fasta') )
		self.assertEqual(
			sA, 
			sB )
	
	def testGeneratingSequences(self):
		MC = nc.MarkovChain(trainingSequences = genome, degree = 4, pseudoCounts = 1, addReverseComplements = True)
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

