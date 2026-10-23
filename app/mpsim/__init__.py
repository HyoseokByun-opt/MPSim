"""MPSim (Material Property Simulation) - RVE homogenisation engine.

The engine knows nothing about the web front-end: `pipeline.run(spec, ...)`
takes a plain dict and writes plain files, so the browser, the command line
and the self-test all drive exactly the same code.
"""
import os as _os
import sys as _sys

__version__ = "5.6.0"

# A conda environment's math libraries (MKL) load their parts from the
# environment's Library\bin by searching PATH. Started without activating the
# environment on a machine whose PATH holds another Anaconda's Library\bin,
# they load a different MKL version and the process dies at the first matrix
# product; the environment's own folder first on PATH avoids that.
_lib = _os.path.join(_sys.prefix, "Library", "bin")
if _os.name == "nt" and _os.path.isdir(_lib):
    _path = _os.environ.get("PATH", "")
    if _os.path.normcase(_path.split(_os.pathsep, 1)[0]) != _os.path.normcase(_lib):
        _os.environ["PATH"] = _lib + _os.pathsep + _path
