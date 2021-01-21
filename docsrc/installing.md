Installing
==============

The recommended way to install Gnocis is through the PyPI package manager. In order to install via the PyPI package manager, open a terminal and execute:

`pip install gnocis`


Alternatively, Gnocis can be built from source. To build a wheel and install it, run:

```
make wheel
pip install dist/*.whl
```

Finally, Gnocis can be used by building from source and including the entire `gnocis` directory in the source tree. In order to do so, run

`make all`

