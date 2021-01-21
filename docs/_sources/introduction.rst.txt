Introduction
==============

About
----------

Large parts of complex genomes of multi-cellular organisms are non-coding. *Cis*-regulatory elements (CREs) are non-coding sequences that establish or modify gene transcription by multiple mechanisms. Multiple classes of CREs have been identified, including promoters, enhancers, silencers, insulators and Polycomb/Trithorax Response Elements. CREs can be identified experimentally or by means of *in silico* prediction. Experimental identification of CREs can depend on the cells that are used. Genome-wide *in silico* prediction, on the other hand, can potentially comprehensively predict CREs in a genome. In order to use machine learning for CRE prediction, a variety of functionality is required. A variety of packages exist for Python 3 for machine learning and sequence analysis, but successfully combining them requires the implementation of interfacing between them. Ensuring that the solution is efficient is important for large genomes, but can be challenging for end-users.

Gnocis is a system in Python 3 for the interactive and reproducible analysis and modelling of CRE DNA sequences. A broad suite of tools is implemented for data preparation, feature set definition, model formulation, training, cross-validation and genome-wide prediction. Gnocis employs Cython and a variety of techniques in order to optimally implement the glue necessary in order to apply machine learning for CRE analysis and prediction.

