#!/usr/bin/python3
# -*- coding: latin-1 -*-
############################################################################
# Gnocis
# Bjørn Bredesen, 2018-2019
# bjorn.bredesen@ii.uib.no
############################################################################

from setuptools import setup
from setuptools.extension import Extension
#from Cython.Build import cythonize

with open('README.md') as f:
	readme = '\n'.join(l for l in f)

setup(
	#ext_modules = cythonize('gnocis/*.pyx', language="c++"),
	ext_modules = [ Extension(name = 'gnocis.' + x, sources = ['gnocis/'+ x + '.cpp']) for x in ['biomarkers', 'common', 'features', 'featurenetwork', 'models', 'motifs', 'regions', 'sequences', 'validation'] ],
	name = "gnocis",
	packages = [ "gnocis" ],
	author = "Bjørn Bredesen",
	author_email = "bjorn@bjornbredesen.no",
	version = "0.9.8a1",
	url = "https://github.com/bjornbredesen/gnocis",
	license = "MIT",
	description = "Gnocis is a system for the analysis and the modelling of cis-regulatory DNA sequences.",
	long_description = readme,
	long_description_content_type = "text/markdown",
	classifiers=[
		"Development Status :: 4 - Beta",
		"Environment :: Console",
		"Intended Audience :: Developers",
		"Intended Audience :: Information Technology",
		"License :: OSI Approved :: MIT License",
		"Operating System :: POSIX :: Linux",
		"Programming Language :: Cython",
		"Programming Language :: Python :: 3.6",
		"Programming Language :: Python :: 3.7",
		"Programming Language :: Python :: 3.8",
		"Topic :: Scientific/Engineering",
	],
)

