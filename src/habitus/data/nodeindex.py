"""Node index: a string-free copy of the card files, for evaluation without card text.

    from habitus.data import nodeindex
    nodeindex.build("CADETS_E3", cards_dir, out_dir)     # -> [out_dir/CADETS_E3/<day>.json.gz, ...]
    Host("CADETS_E3", source="nodeindex", nodeindex_dir=out_dir)

One file per card file of the host, for EVERY card day (HostSpec.card_days: clean, test and empty
days; 45 over the five hosts), because clean-day null pools are read as well as test days:

    {"format": 1, "ds", "day",
     "nodes": {node_id: sig}       in card-file order,
     "types": {sig: node type},
     "kappa": {sig: inv_key}}      kappa = signature({type, invariant}) (Host.inv_key)

That is everything Host needs for nodes / ntype / inv_key / universe / ukey, so score files can be
evaluated with no card strings at hand. `out_dir` is always given by the caller (runs/ in tests,
the release staging directory in scripts/make_release.py).
"""

import os

from habitus import hosts
from habitus.data.cards import card_path, signature
from habitus.data.host import MissingDay, nodeindex_path
from habitus.io import load_json, write_json_gz
from habitus.log import log

FORMAT = 1


def index_day(ds, day, raw):
    """The node-index record of one parsed card file."""
    cards = raw["cards"]
    types = {sig: c["type"] for sig, c in cards.items()}
    kappa = {sig: signature({"type": c["type"], "invariant": c["invariant"]}) for sig, c in cards.items()}
    missing = {s for s in raw["nodes"].values() if s not in cards}
    if missing:
        raise ValueError(f"{ds} {day}: {len(missing)} node signatures have no card")
    return {"format": FORMAT, "ds": ds, "day": day, "nodes": dict(raw["nodes"]), "types": types,
            "kappa": kappa}


def build(ds, cards_dir, out_dir, days=None, *, spec=None):
    """Write out_dir/<ds>/<day>.json.gz for every card day of `ds` (or `days`). Returns the paths.
    A card day without a card file raises MissingDay (an empty day's file must exist too).
    `spec`: a fake HostSpec (tests only); default hosts.spec(ds)."""
    if spec is None:
        spec = hosts.spec(ds)
    elif spec.ds != ds:
        raise ValueError(f"spec is for {spec.ds!r}, not {ds!r}")
    from habitus.config import check_writable
    check_writable(out_dir)
    todo = list(spec.card_days) if days is None else list(days)
    out = []
    for day in todo:
        p = card_path(cards_dir, ds, day)
        if not os.path.exists(p):
            raise MissingDay(f"{ds} {day}: no card file {p}")
        raw = load_json(p)
        q = nodeindex_path(out_dir, ds, day)
        write_json_gz(index_day(ds, day, raw), q)
        out.append(q)
        log(f"{ds} {day}: {len(raw['nodes'])} nodes, {len(raw['cards'])} signatures -> {q}", "data.nodeindex")
    return out
