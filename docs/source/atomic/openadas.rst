Open-ADAS
---------

Although a typical Open-ADAS data set is installed to the local atomic data repository
using the `populate()` function, additional atomic data can be installed manually.

The following functions allow to parse the Open-ADAS files and install the rates of the atomic processes
to the local atomic data repository.

Parse
^^^^^

.. autofunction:: cherab.openadas.parse.adf11.parse_adf11

.. autofunction:: cherab.openadas.parse.adf12.parse_adf12

.. autofunction:: cherab.openadas.parse.adf15.parse_adf15

.. autofunction:: cherab.openadas.parse.adf21.parse_adf21

.. autofunction:: cherab.openadas.parse.adf22.parse_adf22bmp

.. autofunction:: cherab.openadas.parse.adf22.parse_adf22bme

Install
^^^^^^^

.. automodule:: cherab.openadas.install
    :members:
