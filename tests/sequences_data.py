import random
import gnocis as nc

genome = nc.streamFASTAGZ('tutorial/DmelR5.fasta.gz',
		restrictToSequences = [ '2L', '2R', '3L', '3R', '4', 'X' ])

testSeqs = nc.sequences('Test', [
	 nc.sequence('X', ''.join( random.choice(['A', 'C', 'G', 'T'])
	 	for _ in range(800) )),
	 nc.sequence('Y', ''.join( random.choice(['A', 'C', 'G', 'T'])
	 	for _ in range(1000) )),
 ])

