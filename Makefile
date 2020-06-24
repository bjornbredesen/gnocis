
all: gnocis/biomarkers.cpp gnocis/common.cpp gnocis/features.cpp gnocis/featurenetwork.cpp gnocis/models.cpp gnocis/motifs.cpp gnocis/regions.cpp gnocis/sequences.cpp gnocis/validation.cpp library help

clean:
	rm ./gnocis/*.cpp
	rm ./gnocis/*.so
	rm -r ./build/
	rm -r ./dist/
	rm -r ./gnocis.egg-info/

gnocis/biomarkers.cpp: gnocis/biomarkers.pyx
	cython -3 --cplus gnocis/biomarkers.pyx --output-file gnocis/biomarkers.cpp

gnocis/common.cpp: gnocis/common.pyx
	cython -3 --cplus gnocis/common.pyx --output-file gnocis/common.cpp

gnocis/features.cpp: gnocis/features.pyx
	cython -3 --cplus gnocis/features.pyx --output-file gnocis/features.cpp

gnocis/featurenetwork.cpp: gnocis/featurenetwork.pyx
	cython -3 --cplus gnocis/featurenetwork.pyx --output-file gnocis/featurenetwork.cpp

gnocis/models.cpp: gnocis/models.pyx
	cython -3 --cplus gnocis/models.pyx --output-file gnocis/models.cpp

gnocis/motifs.cpp: gnocis/motifs.pyx
	cython -3 --cplus gnocis/motifs.pyx --output-file gnocis/motifs.cpp

gnocis/regions.cpp: gnocis/regions.pyx
	cython -3 --cplus gnocis/regions.pyx --output-file gnocis/regions.cpp

gnocis/sequences.cpp: gnocis/sequences.pyx
	cython -3 --cplus gnocis/sequences.pyx --output-file gnocis/sequences.cpp

gnocis/validation.cpp: gnocis/validation.pyx
	cython -3 --cplus gnocis/validation.pyx --output-file gnocis/validation.cpp

help:
	rm -f docsrc/_static/gnocis_icon.png
	cp markdown/gnocis_icon.png docsrc/_static/gnocis_icon.png
	rm -rf docs/
	sphinx-build -M html docsrc .
	mv html docs
	touch docs/.nojekyll

library:
	python3.6 setup.py build_ext --build-lib=gnocis sdist bdist_wheel
	rm -f ./gnocis/*.so
	cp -f ./gnocis/gnocis/*.so ./gnocis/
	rm -rf ./gnocis/gnocis

