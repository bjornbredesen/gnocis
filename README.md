
![Gnocis](https://github.com/bjornbredesen/gnocis/blob/e6098dc9333f198d1d8d071e7ff4f7eede5a66bf/markdown/gnocis.png "")

# Gnocis
Bjørn André Bredesen-Aa, 2018-

#### Stability

![](https://github.com/bjornbredesen/gnocis/workflows/Tests%20(Linux,%20single%20version%20of%20Python)/badge.svg)
![](https://github.com/bjornbredesen/gnocis/workflows/Tests%20(Windows,%20single%20version%20of%20Python)/badge.svg)
![](https://github.com/bjornbredesen/gnocis/workflows/Tests%20(MacOS,%20single%20version%20of%20Python)/badge.svg)
![](https://github.com/bjornbredesen/gnocis/workflows/Publish%20to%20PyPI/badge.svg)


----------------------------------------------------------------------

## About

Large parts of complex genomes of multi-cellular organisms are non-coding. *Cis*-regulatory elements (CREs) are non-coding sequences that establish or modify gene transcription by multiple mechanisms. Multiple classes of CREs have been identified, including promoters, enhancers, silencers, insulators and Polycomb/Trithorax Response Elements. CREs can be identified experimentally or by means of *in silico* prediction. Experimental identification of CREs can depend on the cells that are used. Genome-wide *in silico* prediction, on the other hand, can potentially comprehensively predict CREs in a genome. In order to use machine learning for CRE prediction, a variety of functionality is required. A variety of packages exist for Python 3 for machine learning and sequence analysis, but successfully combining them requires the implementation of interfacing between them. Ensuring that the solution is efficient is important for large genomes, but can be challenging for end-users.

Gnocis is a system in Python 3 for the interactive and reproducible analysis and modelling of CRE DNA sequences. A broad suite of tools is implemented for data preparation, feature set definition, model formulation, training, cross-validation and genome-wide prediction. Gnocis employs Cython and a variety of techniques in order to optimally implement the glue necessary in order to apply machine learning for CRE analysis and prediction.


----------------------------------------------------------------------

## Installing

The recommended way to install Gnocis is through the PyPI package manager. In order to install via the PyPI package manager, open a terminal and execute:

`pip install gnocis`


Alternatively, Gnocis can be built from source. To build a wheel and install it, run:

```
make wheel
pip install dist/*.whl
```

Finally, Gnocis can be used by building from source and including the entire `gnocis` directory in the source tree. In order to do so, run

`make all`


## Installing dependencies for tutorial

```
sudo apt-get install python3-sphinx
sudo pip3 install notebook pandas numpy matplotlib cupy sklearn tensorflow
```


----------------------------------------------------------------------

## Documentation

For the complete manual, see: https://bjornbredesen.github.io/gnocis/

For an in-depth tutorial, see the Jupyter Notebooks in the `tutorial/` folder.


----------------------------------------------------------------------

## Features

 * DNA sequence handling
     * File format support - Loading and streaming
        - FASTA
        - 2bit
     * File format support - Saving
        - FASTA
     * Operations
        - Printing
        - Sliding window extraction
        - Reverse complement generation
 * Sequence region handling
     * File format support - Loading and saving
        - GFF
        - BED
        - Coordinate lists (`chromosome:start..end`)
     * Operations
        - Overlap acquisition
        - Non-overlap acquisition
        - Merged set generation
        - Exclusion set generation
        - Sequence region extraction
 * Modelling
     * Generative DNA sequence models, with training and sequence generation
        - I.i.d.
        - N'th order Markov chains
     * Confusion matrices
        - Generation from model statistics
        - Printing
        - Receiver Operating Characteristic curve generation
        - Precision Recall Curve generation
        - Area Under the Curve calculation
     * Feature models
        - Log-odds
        - Dummy
        - Support Vector Machines (via sklearn)
        - Random Forest (via sklearn)
     * Features
        - *k*-mer spectrum
        - Motif occurrence spectrum
        - Motif pair occurrence spectrum
 * Motifs
     * Types
        - IUPAC nucleotide motifs
        - Position Weight Matrices
        - *k*-mer spectra
 * Feature networks
    * Directed acyclic graphs of features
    * Transformations of feature sets: filtering; concatenation; scaling; square; ...
    * Feature network nodes for constructing models
    * Application to sequences
 * Optionally integrates with established packages
    * Numpy – for integration with external methods
    * Pandas – for integration with external methods
    * Scikit-learn – for extended analyses and classic machine learning
    * TensorFlow – for neural networks
    * Jupyter Notebooks – for interactive and reproducible analysis and modelling
 * Easy to use
 * Objects are represented by classes, with human-readable descriptions
 * Optimized with Cython
 * ...


-------------------------------------------------

## Requirements

 * Python 3.6, 3.7, 3.8, 3.9
 * Windows, MacOS or Linux
 * C++ compiler when installing on Linux
 * Optional: Cython – required only when building from source
 * Optional: sklearn – for SVM and RF modelling
 * Optional: CuPy and CUDA – for CUDA-optimized SVM
 * Optional: TensorFlow – for neural networks


-------------------------------------------------

## Citing
If you use Gnocis in published research, Gnocis must be cited. An article for Gnocis is in the process of being submitted for peer review. Please check back for an updated citation policy.


----------------------------------------------------------------------

## License

MIT License

Copyright (c) 2018- Bjørn André Bredesen-Aa

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

----------------------------------------------------------------------

## Logo

Logo: Copyright Bjørn André Bredesen-Aa


