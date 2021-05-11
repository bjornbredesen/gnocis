
all: library help

clean:
	rm ./gnocis/*.cpp
	rm ./gnocis/*.so
	rm -r ./build/
	rm -r ./dist/
	rm -r ./gnocis.egg-info/

help:
	rm -f docsrc/_static/gnocis_icon.png
	cp markdown/gnocis_icon.png docsrc/_static/gnocis_icon.png
	rm -rf docs/
	sphinx-build -M html docsrc .
	mv html docs
	touch docs/.nojekyll

library:
	cython -3 --cplus gnocis/*.pyx
	python3 setup.py build_ext --build-lib=gnocis sdist bdist_wheel
	rm -f ./gnocis/*.so
	cp -f ./gnocis/gnocis/*.so ./gnocis/
	rm -rf ./gnocis/gnocis

wheel:
	cython -3 --cplus gnocis/*.pyx
	python3 setup.py sdist bdist_wheel

