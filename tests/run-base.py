#!/usr/bin/python3
# -*- coding: latin-1 -*-
############################################################################

from os import mkdir, rmdir
import unittest


#-----------------------------------
# Data

if __name__ == '__main__':
	try:
		rmdir('temp')
	except:
		pass
	try:
		mkdir('temp')
	except:
		pass


#-----------------------------------

from regions import *
# from sequences import *
# from features import *
# from models import *


#-----------------------------------

if __name__ == '__main__':
	print('Running all tests')
	unittest.main()

