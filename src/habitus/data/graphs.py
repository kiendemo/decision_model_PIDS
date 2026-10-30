"""PIDSMaker graph windows for the card builder.

The transformation stage writes one torch.save'd networkx.MultiDiGraph per 15-minute window:
<graphs_dir>/graph_<YYYY-MM-DD>/<t0>~<t1>. Nodes carry node_type and label; edges carry label (the
collector's operation), time (ns), event_uuid and a per-edge ground-truth label `y`. The card code
never reads `y`, and `windows()` deletes it from every edge as the graph is loaded, so no consumer
of a Window can reach it.

torch.load(weights_only=False) is used ONLY for these trusted PIDSMaker graph files.

Vendored from the original reflex/ledger.py (CREATION_OPS, _windows), defense/context.py
(path_of, incident_events, exec_source) and upstream utils.get_all_graphs_for_dates (window order =
int of all digits of the absolute path).
"""

import os

from habitus.data.profiles import profile_for

# Process creation, parent -> child, per collector family (THEIA fluxbox->firefox, OpTC
# svchost.exe->cmd.exe, ATLASv2 services.exe->svchost.exe). CADETS and CLEARSCOPE record no
# subject->subject edge.
CREATION_OPS = frozenset({"EVENT_CLONE", "EVENT_FORK", "CREATE", "PROCESS_EXECUTED"})
LABEL_ATTRS = ("y",)          # edge attributes removed at load time (ground truth)


def graph_paths(graphs_dir, day):
    """The day's window files, in PIDSMaker order (= get_all_graphs_for_dates(graphs_dir, [day]))."""
    sub = f"graph_{day}"
    if not os.path.isdir(graphs_dir):
        raise FileNotFoundError(f"graph directory {graphs_dir} does not exist")
    paths = [os.path.abspath(os.path.join(graphs_dir, s, f))
             for s in os.listdir(graphs_dir) if os.path.isdir(os.path.join(graphs_dir, s)) and s == sub
             for f in os.listdir(os.path.join(graphs_dir, s))]
    paths.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    return paths


def load_graph(path):
    """One window graph, with the ground-truth edge attributes removed."""
    import torch
    G = torch.load(path, weights_only=False)
    for _u, _v, d in G.edges(data=True):
        for a in LABEL_ATTRS:
            d.pop(a, None)
    return G


def windows(paths):
    """(window name, graph, (day, t0, t1)) for every graph file with >= 1 edge."""
    for gp in paths:
        G = load_graph(gp)
        times = [d["time"] for _u, _v, d in G.edges(data=True)]
        if not times:
            continue
        day = os.path.basename(os.path.dirname(gp)).replace("graph_", "")
        yield f"{day}/{os.path.basename(gp)}", G, (day, min(times), max(times))


class Window:
    """One window: node types, labels and resolved entities, plus the per-node event view."""

    def __init__(self, window, resolver, graph, dataset):
        self.window, self.G, self.resolver = window, graph, resolver
        self.ntype = {n: graph.nodes[n].get("node_type") for n in graph.nodes()}
        self.label = {n: graph.nodes[n].get("label", "") for n in graph.nodes()}
        self.ent = {n: resolver.resolve(n, self.ntype.get(n), self.label.get(n)) for n in graph.nodes()}
        self.profile = profile_for(dataset)
        self._events = {}

    def path_of(self, nid):
        e = self.ent.get(nid)
        return e.path if e is not None else None

    def incident_events(self, nid):
        """[(direction, op, time, other node)] for the edges touching `nid`, time order (cached)."""
        if nid in self._events:
            return self._events[nid]
        ev = [("out", d.get("label"), d.get("time"), dst) for _, dst, d in self.G.out_edges(nid, data=True)]
        ev += [("in", d.get("label"), d.get("time"), src) for src, _, d in self.G.in_edges(nid, data=True)]
        ev.sort(key=lambda e: (e[2] if e[2] is not None else 0))
        self._events[nid] = ev
        return ev

    def exec_source(self, nid):
        """Path of the file this subject was executed from (its first EXECUTE edge to a file with a
        path in this window), else the image path recorded on the process itself, else None."""
        if self.ntype.get(nid) != "subject":
            return None
        for _d, op, _t, other in self.incident_events(nid):
            if op in self.profile.execute and self.ntype.get(other) == "file":
                path = self.path_of(other)
                if path:
                    return path
        ent = self.ent.get(nid)
        if ent is not None and ent.path:
            return ent.path
        return None
