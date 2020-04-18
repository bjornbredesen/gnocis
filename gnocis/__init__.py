# -*- coding: latin-1 -*-
############################################################################
# Gnocis
# Bjørn Bredesen, 2018-2019
# bjorn.bredesen@ii.uib.no
############################################################################

from .about import *
from .ioputil import nctable
from .common import setSeed, nucleotides, complementaryNucleotides, getReverseComplementaryDNASequence, IUPACNucleotideCodes, IUPACNucleotideCodeSemantics, mean, std, SE, CI, KLdiv
from .regions import region, regions, loadBED, loadBEDGZ, loadGFF, loadGFFGZ, loadCoordinateList
from .sequences import sequence, sequences, sequenceStream, loadFASTA, loadFASTAGZ, streamFASTA, streamFASTAGZ, stream2bit, streamSequenceWindows, getSequenceWindowRegions, generatorMarkovChain, generatorIID, positive, negative
from .motifs import motifOccurrence, motifs, IUPACMotif, PWMMotif, loadMEMEPWMDatabase
from .features import feature, features, scaledFeature, featureScaler, featureMotifOccurrenceFrequency, featurePREdictorMotifPairOccurrenceFrequency, kSpectrum, kSpectrumMM, kSpectrumGPS
from .featurenetwork import featureNetworkNode, FNNMotifOccurrenceFrequencies, FNNMotifPairOccurrenceFrequencies, FNNLogOdds, FNNkSpectrum, FNNkSpectrumMM, FNNScaler
from .models import setNCores, sequenceModel, sequenceModelDummy, sequenceModelLogOdds, trainSinglePREdictorModel, createDummyPREdictorModel, trainPREdictorModel, crossvalidate
from .biomarkers import biomarkers
from .validation import point2D, validationPair, getROC, getPRC, getAUC, getConfusionMatrix, getConfusionMatrixStatistics, printValidationStatistics
from .genome import gene, genome, plotGenomeTracks

