import gnocis as nc
import unittest
from regions_data import *

def getRegionsStr(rs):
	return '; '.join( r.bstr() for r in rs )

class testRegions(unittest.TestCase):
	
	def testMerge(self):
		self.assertEqual( getRegionsStr( rsA.merge(rsB) ),
			'X:20..50 (+); X:80..120 (+); X:140..400 (+); X:500..600 (+); Y:40..100 (+); Y:120..300 (+); Y:600..700 (+)' )
	
	def testIntersection(self):
		self.assertEqual( getRegionsStr( rsA.intersection(rsB) ),
			'X:30..40 (+); X:90..100 (+); X:150..150 (+); X:300..300 (+); X:305..310 (+); Y:40..100 (+); Y:130..200 (+)' )
	
	def testExcluded(self):
		self.assertEqual( getRegionsStr( rsA.exclusion(rsB) ),
			'X:20..29 (+); X:41..50 (+); X:80..89 (+); X:151..299 (+); X:311..400 (+); X:500..600 (+); Y:120..129 (+)' )
	
	def testOverlap(self):
		self.assertEqual( getRegionsStr( rsA.overlap(rsB) ),
			'X:20..50 (+); X:80..100 (+); X:150..300 (+); X:305..400 (+); Y:40..100 (+); Y:120..200 (+)' )
	
	def testNonOverlap(self):
		self.assertEqual( getRegionsStr( rsA.nonOverlap(rsB) ),
			'X:500..600 (+)' )
	
	def testSaveLoadGFF(self):
		rsA.saveGFF('temp/test.GFF')
		self.assertEqual( getRegionsStr( rsA ),
			getRegionsStr( nc.loadGFF('temp/test.GFF') ) )
	
	def testSaveLoadBED(self):
		rsA.saveBED('temp/test.BED')
		self.assertEqual( getRegionsStr( rsA ),
			getRegionsStr( nc.loadBED('temp/test.BED') ) )

