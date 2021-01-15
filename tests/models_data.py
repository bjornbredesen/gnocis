import random
import gnocis as nc
from regions_data import *
from sequences_data import *

PcG = nc.biomarkers('PcG', [
	nc.loadGFFGZ('tutorial/Pc.gff3.gz').deltaResize(1000).rename('Pc'),
	nc.loadGFFGZ('tutorial/Psc.gff3.gz').deltaResize(1000).rename('Psc'),
	nc.loadGFFGZ('tutorial/dRING.gff3.gz').deltaResize(1000).rename('dRING'),
	nc.loadGFFGZ('tutorial/H3K27me3.gff3.gz').rename('H3K27me3'),
])

gwWin = nc.getSequenceWindowRegions(
	genome,
	windowSize = 1000, windowStep = 100)

PcGTargets = PcG.HBMEs(gwWin, threshold = 4)

PRESeq = PcGTargets.recenter(3000).extract(genome)
random.shuffle(PRESeq.sequences)

MC = nc.MarkovChain(trainingSequences = genome, degree = 4, pseudoCounts = 1, addReverseComplements = True)

