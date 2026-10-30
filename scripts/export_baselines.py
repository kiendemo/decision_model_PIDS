#!/usr/bin/env python
"""Server-only export of the PIDSMaker baseline rows (PIDSMaker 2289cd9 runs in paths.upstream_artifacts).

    python scripts/export_baselines.py [--hosts CADETS_E3,...] [--upstream-artifacts DIR]
                                       [--results-dir results/baselines] [--runs-dir runs/baselines]
                                       [--source nodeindex|cards] [--cards-dir DIR]

Writes results/baselines/{pidsmaker_rows.csv, pidsmaker_missing.csv, pidsmaker_args.sh, PROVENANCE.md} and
runs/baselines/<ds>_best_baseline.npz (scripts/make_release.py copies the npz into the cache bundle; this
script never writes release/). Needs the upstream run artifacts (torch pickles, loaded with
weights_only=False: trusted local files only), torch, the host data (node index or cards) and the GT
(settings.paths.ground_truth, or HABITUS_GT). CPU only.
"""

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
try:
    import habitus  # noqa: F401
except ImportError:                                    # run from a checkout without installing
    sys.path.insert(0, str(REPO / "src"))


def main(argv=None):
    from habitus.config import settings
    from habitus.eval import baselines
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--hosts", default=",".join(baselines.HOSTS))
    ap.add_argument("--upstream-artifacts", default=None, help="default: settings.paths.upstream_artifacts")
    ap.add_argument("--results-dir", default=str(REPO / "results" / "baselines"))
    ap.add_argument("--runs-dir", default=None, help="default: <settings.paths.runs>/baselines")
    ap.add_argument("--source", default="nodeindex", choices=("nodeindex", "cards"))
    ap.add_argument("--cards-dir", default=None, help="cards root for --source cards (default settings.paths.cards)")
    a = ap.parse_args(argv)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    st = settings()
    runs_dir = a.runs_dir or str(Path(str(st.paths.runs)) / "baselines")
    factory = None
    if a.cards_dir:
        from habitus.data.host import Host

        def factory(ds):
            return Host(ds, source="cards", cards_dir=a.cards_dir).keep_clean_days()
    hosts = [h for h in a.hosts.split(",") if h]
    best = baselines.export(hosts, up=a.upstream_artifacts, results_dir=a.results_dir, runs_dir=runs_dir,
                            source=a.source, settings=st, host_factory=factory)
    for ds, info in best.items():
        print(f"{ds}: best baseline {info['system']} ({info['log']}, epoch {info['epoch']}) own MCC {info['own_mcc']:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
