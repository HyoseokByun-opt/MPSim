"""Serve MPSim (Material Property Simulation) on the network.

    python run_web.py                       # http://<this machine>:8000
    python run_web.py --check               # why can nobody else connect?
    python run_web.py --token auto          # put a shared secret in front
    python run_web.py --host 127.0.0.1      # this machine only
    python run_web.py --max-voxels 200      # allow larger RVEs (more memory)

Anyone who can reach the address can start simulations and spend this
machine's cores, so --token puts a shared secret in front of it. That is a
door lock for a lab network, not authentication: do not expose this to the
public internet.

The __main__ guard matters: every job runs in a spawned child process, and on
Windows each child re-imports this module; without the guard each would start
its own web server.
"""
import argparse
import multiprocessing
import os
import secrets
import socket
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main(argv=None):
    from mpsim import rve as R                      # the grid limits live with the planner
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--token", default=None, help="shared access token; 'auto' generates one")
    ap.add_argument("--workspace", default=None, help="where runs are kept (default: ../workspace)")
    ap.add_argument("--concurrency", type=int, default=1,
                    help="simulations at once; more than one splits the cores between them")
    ap.add_argument("--max-voxels", type=int, default=R.DEFAULT_LIMITS["N_max"],
                    help="largest RVE edge in voxels (conduction); 0 = no limit")
    ap.add_argument("--max-voxels-elastic", type=int, default=R.DEFAULT_LIMITS["N_max_elastic"],
                    help="largest edge for CTE/elasticity; 0 = no limit")
    ap.add_argument("--max-voxels-flow", type=int, default=R.DEFAULT_LIMITS["N_max_flow"],
                    help="largest edge for permeability; 0 = no limit")
    ap.add_argument("--check", action="store_true", help="report why other machines may not connect, then exit")
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args(argv)

    from web import netcheck
    if a.check:
        netcheck.report(a.port, a.host)
        return 0

    token = secrets.token_urlsafe(9) if a.token == "auto" else a.token
    from web.server import create_app
    limits = {"N_max": a.max_voxels, "N_max_elastic": a.max_voxels_elastic, "N_max_flow": a.max_voxels_flow}
    app = create_app(workspace=a.workspace, token=token, concurrency=a.concurrency, limits=limits)

    addrs = netcheck.local_addresses()
    print("=" * 66)
    print("  MPSim  ·  Material Property Simulation  ·  RVE homogenisation of semiconductor materials")
    print("=" * 66)
    print(f"  this PC   http://127.0.0.1:{a.port}")
    if a.host in ("0.0.0.0", "::"):
        for ip in addrs:
            print(f"  network   http://{ip}:{a.port}")
        try:
            name = socket.gethostname()
            socket.gethostbyname(name)
            print(f"  by name   http://{name}:{a.port}")
        except Exception:                                      # noqa: BLE001
            pass
    if token:
        print(f"  token     {token}   (first visit: ?token={token})")
    _cap = lambda n: "no limit" if not n or n <= 0 else f"{n}³"      # noqa: E731
    print(f"  jobs      {a.concurrency} at a time · RVE {_cap(a.max_voxels)} "
          f"(elasticity {_cap(a.max_voxels_elastic)}, flow {_cap(a.max_voxels_flow)})")
    print("  Ctrl+C to stop")
    print("=" * 66, flush=True)
    app.run(host=a.host, port=a.port, threaded=True, debug=a.debug, use_reloader=False)
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
