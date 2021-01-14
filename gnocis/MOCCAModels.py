#!/usr/bin/python3
# -*- coding: latin-1 -*-
############################################################################
# Gnocis
# Bjørn Bredesen, 2018-2019
# bjorn.bredesen@ii.uib.no
############################################################################
# Interfacing with MOCCA

from .regions import loadGFF
from .sequences import sequences, sequenceStream, positive, negative, MarkovChain, IID
from .models import sequenceModel
from os import system

class SVMMOCCA(sequenceModel):
	def __init__(self, name, MOCCAPath, motifs, windowSize, windowStep, kDegree, labelsPositive = [ positive ], labelsNegative = [ negative ], trainingSet = None, silent = True, TempPath = '/dev/shm/mocca/'):
		super().__init__(name, enableMultiprocessing = False)
		self.MOCCAPath = MOCCAPath
		self.tmpPath = TempPath
		self.motifs = motifs
		self.kDegree = kDegree
		self.threshold = 0.0
		self.labelsPositive, self.labelsNegative = labelsPositive, labelsNegative
		self.windowSize, self.windowStep = windowSize, windowStep
		self.silent = silent
		
		# Create temporary files
		system("if [ -e \"" + self.tmpPath + "\" ]; then rm -r " + self.tmpPath + "; fi")
		system('mkdir ' + self.tmpPath)
		labelFiles = {
			lbl: self.tmpPath + 'train_' + lbl.name.replace(' ', '_') + '.fa'
			for lbl in labelsPositive + labelsNegative
		}
		if trainingSet is not None:
			for lbl in labelFiles:
				trainingSet.withLabel(lbl).saveFASTA(labelFiles[lbl])
		
		# Build call path
		self.cpath = '' + MOCCAPath
		self.cpath += ' -C:SVM-MOCCA'
		self.cpath += ' -f:MOCCA:nOcc'
		self.cpath += ' -f:MOCCA:DNT'
		self.cpath += ' -validate:no'
		self.cpath += ' -wSize %d'%self.windowSize
		self.cpath += ' -wStep %d'%self.windowStep
		self.cpath += ' -k:%s'%[ 'linear', 'quadratic', 'cubic' ][ int(kDegree)-1 ]
		self.cpath += ' '
		self.cpath += ' '.join( '-motif:IUPAC "%s" %s %d'%(m.name, m.motif, m.nmismatches) for m in motifs )
		self.cpath += ' '
		self.cpath += ' '.join( '-class "%s" %d +'%(lbl.name, i+1) for i, lbl in enumerate(labelsPositive) )
		self.cpath += ' '
		self.cpath += ' '.join( '-class "%s" -%d -'%(lbl.name, i+1) for i, lbl in enumerate(labelsNegative) )
		self.cpath += ' '
		self.cpath += ' '.join( '-train:FASTA %s %d full'%(labelFiles[lbl], i+1) for i, lbl in enumerate(labelsPositive) )
		self.cpath += ' '
		self.cpath += ' '.join( '-train:FASTA %s -%d full'%(labelFiles[lbl], i+1) for i, lbl in enumerate(labelsNegative) )
	
	def getSequenceScores(self, seqs, cache = True):
		if isinstance(seqs, sequenceStream):
			return self.scoreSequences(seqs.streamFullSequences())
		elif isinstance(seqs, sequences) or isinstance(seqs, list):
			return self.scoreSequences(seqs)
	
	def scoreWindow(self, seq):
		tPath = self.tmpPath + 'vSeq.fasta'
		sequences('', [seq]).saveFASTA(tPath)
		oPath = tPath + '.scores'
		cpath = self.cpath[:] + ' -validate:FASTA ' + tPath + ' + -validate:outSCTable ' + oPath
		if not self.silent:
			print('Debug: SVM-MOCCA - Classifier call: ' + self.cpath)
		if self.silent:
			cpath += ' > ' + self.tmpPath + 'log.txt'
		system(cpath)
		with open(oPath, 'r') as f:
			_ = next(f)
			scores = [ float(p[0]) for p in ( l.strip().split('\t') for l in f ) if len(p) == 2 ]
		return max(scores) if len(scores) > 0 else 0.0
	
	def scoreSequence(self, seq):
		tPath = self.tmpPath + 'vSeq.fasta'
		sequences('', [seq]).saveFASTA(tPath)
		oPath = tPath + '.scores'
		cpath = self.cpath[:] + ' -validate:FASTA ' + tPath + ' + -validate:outSCTable ' + oPath
		if not self.silent:
			print('Debug: SVM-MOCCA - Classifier call: ' + self.cpath)
		if self.silent:
			cpath += ' > ' + self.tmpPath + 'log.txt'
		system(cpath)
		with open(oPath, 'r') as f:
			_ = next(f)
			scores = [ float(p[0]) for p in ( l.strip().split('\t') for l in f ) if len(p) == 2 ]
		score = max(scores) if len(scores) > 0 else 0.0
		return score
	
	def scoreSequences(self, seq):
		tPath = self.tmpPath + 'vSeq.fasta'
		sseq = sequences('', [ s for s in seq ])
		sseq.saveFASTA(tPath)
		oPath = tPath + '.scores'
		cpath = self.cpath[:] + ' -validate:FASTA ' + tPath + ' + -validate:outSCTable ' + oPath
		if not self.silent:
			print('Debug: SVM-MOCCA - Classifier call: ' + self.cpath)
		if self.silent:
			cpath += ' > ' + self.tmpPath + 'log.txt'
		system(cpath)
		with open(oPath, 'r') as f:
			_ = next(f)
			scores = [ float(p[0]) for p in ( l.strip().split('\t') for l in f ) if len(p) == 2 ]
		return scores
		
	def predictSequenceStreamRegions(self, stream):
		pred = regions('Predictions', [])
		windowSP = list(streamSequenceWindows(stream, self.windowSize, self.windowStep))
		windowSequences = sequences('', [ winSeq for winSeq in windowSP ])
		windowPositions = [ winSeq.sourceRegion for winSeq in windowSP ]
		wScores = self.scoreSequences(windowSequences)
		for wScore, winPos in zip(wScores, windowPositions):
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
	
	def predictCore(self, vpos, genome, factor = 0.5, precision = 0.8, bgModelOrder = 4):
		cmd = self.cpath
		
		# Positives
		vposPath = self.tmpPath + 'vpos.fa'
		vpos.saveFASTA(vposPath)
		cmd += ' -calibrate:FASTA ' + vposPath + ' +'

		# Negatives
		seqLens = genome.sequenceLengths()
		genomeLen = sum(seqLens[chrom] for chrom in seqLens.keys())
		meanLen = int(sum(len(s) for s in vpos) / len(vpos))
		if bgModelOrder > 0:
			bgModel = MarkovChain(trainingSequences = genome, degree = bgModelOrder)
		else:
			bgModel = IID(trainingSequences = genome, degree = bgModelOrder)
		vneg = bgModel.generateSet(n = int(factor * genomeLen / meanLen), length = meanLen)
		vnegPath = self.tmpPath + 'vneg.fa'
		vneg.saveFASTA(vnegPath)
		cmd += ' -calibrate:FASTA ' + vnegPath + ' -'

		# Run
		cmd += ' -calibrate:precision ' + str(precision)
		cmd += ' -predict:core'
		outGFFPath = self.tmpPath + 'pred.gff'
		cmd += ' -predict:GFF ' + outGFFPath
		genomePath = self.tmpPath + 'genome.fa'
		sequences('', [ s for s in genome.sequences.streamFullSequences() ]).saveFASTA(genomePath)
		cmd += ' -genome:FASTA ' + genomePath
		if self.silent:
			cmd += ' > ' + self.tmpPath + 'log.txt'
		system(cmd)
		return loadGFF(outGFFPath)
	
	def getTrainer(self):
		return lambda ts: SVMMOCCA(name = self.name, MOCCAPath = self.MOCCAPath, motifs = self.motifs, windowSize = self.windowSize, windowStep = self.windowStep, kDegree = self.kDegree, labelsPositive = self.labelsPositive, labelsNegative = self.labelsNegative, trainingSet = ts, silent = self.silent, TempPath = self.tmpPath)
	
	def __str__(self):
		return 'SVM-MOCCA<Motifs: %s; kDegree: %d>'%(str(self.motifs), self.kDegree)
	
	def __repr__(self): return self.__str__()

