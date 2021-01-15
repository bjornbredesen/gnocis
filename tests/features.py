import gnocis as nc
import unittest

class testFeatures(unittest.TestCase):
	
	def testKSpectrum(self):
		spec = nc.features.kSpectrum(5)
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
		spec = nc.features.kSpectrumMM(5)
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
		spec = nc.features.kSpectrumMMD(5)
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

