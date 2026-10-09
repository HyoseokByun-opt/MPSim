"""Report which parts of the simulator can be imported, and from where.

Used by 1_INSTALL_OFFLINE.bat. On an offline PC this is the fastest way to
see whether a failure is a missing package, a missing DLL, or the wrong Python.
Exit code 1 when a required part is missing.
"""
import importlib
import sys
import warnings

PARTS = [
    ("numpy", "arrays"),
    ("scipy", "solvers"),
    ("numba", "structure generation and the finite-volume solver"),
    ("skimage", "image analysis"),
    ("pumapy", "NASA PuMA - elasticity, flow, finite elements"),
    ("porespy", "porosimetry and pore network"),
    ("sklearn", "surrogate model on the Optimization tab"),
    ("skrf", "EMI shielding"),
    ("tifffile", "structure export"),
    ("pyvista", "3D figures"),
    ("vtk", "3D figures"),
    ("matplotlib", "section figures"),
    ("psutil", "memory check"),
    ("flask", "the web interface"),
    ("taichi", "particle dynamics of the viscosity analysis (CPU or CUDA GPU)"),
]
OPTIONAL = []


def main():
    warnings.simplefilter("ignore")
    print("     python : %s  (%s)" % (sys.version.split()[0], sys.executable))
    missing = []
    for name, what in PARTS + OPTIONAL:
        try:
            m = importlib.import_module(name)
            ver = getattr(m, "__version__", "")
            print("  ok  %-11s %-12s %s" % (name, ver, what))
        except Exception as e:                                      # noqa: BLE001
            opt = (name, what) in OPTIONAL
            print("  %s %-11s %s -> %s" % ("--  " if opt else "FAIL", name, what, str(e).splitlines()[0][:120]))
            if not opt:
                missing.append(name)
    if missing:
        print("\n  missing: " + ", ".join(missing))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
