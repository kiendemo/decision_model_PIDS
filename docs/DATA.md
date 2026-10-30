# Data: from the database dumps to evidence cards

HABITUS scores five hosts: `CADETS_E3`, `THEIA_E3`, `CLEARSCOPE_E3` (DARPA TC engagement 3),
`optc_h501` (DARPA OpTC) and `ATLASV2_EDR` (ATLASv2, Carbon Black EDR). The scoring pipeline reads
only the **evidence cards** (one bounded summary per node and day) and their **link files**; the
release bundle ships them, so nothing in this document is needed to reproduce the published numbers
from `release/habitus-cache-v1`. This document is for rebuilding the cards from the raw data and
checking that the rebuild matches the shipped ones.

```
dump download -> pg_restore -> habitus data graphs (upstream PIDSMaker, subprocess)
              -> habitus data resolver -> habitus data cards -> habitus data manifest / index / verify
```

No step needs a GPU. Nothing in this layer reads ground truth: the graph loader deletes the
per-edge label attribute `y` as each window is read, and the resolver and card builder never look
at labels.

## Two environments

| environment | what runs there |
|---|---|
| upstream PIDSMaker (its Docker image, Python 3.9, or an equivalent env with PIDSMaker 2289cd9 + `docs/upstream.patch` installed) | `habitus data graphs` only: `scripts/build_graphs.py` imports all of `pidsmaker.main` (torch, wandb, every task), so it is started as a subprocess under `paths.pidsmaker_python` |
| this repository (Python 3.13, `requirements-lock.txt`) | everything else; `habitus.data.upstream` imports only the upstream `pidsmaker.config` package (yacs, PyYAML; it pulls `pidsmaker.utils`), either installed or from the source tree at `paths.upstream_src` |

Set the server paths in the gitignored `configs/local.yaml` (template: `configs/local.example.yaml`):
`pidsmaker_python`, `upstream_src`, and, to read existing PIDSMaker outputs, `pidsmaker_artifacts`.
Configured inputs are read only; every file the commands below write goes under `paths.runs`
(they refuse an output directory inside a configured input).

## 0. Prerequisites

1. **This repository**, installed editable (`pip install -e .`; a non-editable install cannot find
   `configs/`) or used from a checkout with `PYTHONPATH=src`, in the environment of
   `requirements-lock.txt`. Its torch pin `torch==2.12.0+cu132` is the CUDA 13.2 wheel: install it
   with `pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cu132`.
2. **Upstream PIDSMaker at commit 2289cd9** (Apache-2.0):

   ```bash
   git clone https://github.com/ubc-provenance/PIDSMaker.git
   cd PIDSMaker
   git checkout 2289cd9
   git apply <this repository>/docs/upstream.patch     # the optc_h501 database name (section 1)
   ```

   Build its environment (its `Dockerfile` / `compose-pidsmaker.yml`, Python 3.9) and set, in
   `configs/local.yaml`, `paths.upstream_src` to the clone and `paths.pidsmaker_python` to that
   environment's interpreter.
3. **Postgres**, from the clone: `INPUT_DIR=<dump directory> docker compose -f compose-postgres.yml up -d`
   (image postgres:17, container `postgres` on the docker network `shared_network`, user `postgres`,
   password `postgres`, port `${DOCKER_PORT:-5432}`; the dump directory is mounted at `/data` in the
   container). These upstream defaults are the HABITUS defaults too (`habitus.data.upstream.DB_DEFAULTS`:
   host `postgres`, user `postgres`, password `postgres`, port 5432); from outside the docker network
   pass `--db-host localhost`. Give any other password through `--db-password-file FILE` (first line,
   mode 0600) or `$HABITUS_DB_PASSWORD`, never on the command line, where other users of a shared
   machine can read it in `ps`; HABITUS passes it to the graph-build child in its environment.
4. **A Google Drive access token** for `download_datasets.sh`: the dumps are files in the upstream
   authors' Google Drive, fetched with the Drive API. Get a short-lived OAuth 2.0 access token with the
   scope `https://www.googleapis.com/auth/drive.readonly` (Google's OAuth 2.0 Playground: paste the
   scope, "Authorize APIs", "Exchange authorization code for tokens", copy `access_token`; it expires
   after about an hour). Or download the five dumps by hand from the Drive folder linked in upstream
   `docs/docs/ten-minute-install.md`.

## 1. Download and restore the dumps

Use upstream PIDSMaker's `download_datasets.sh` with the five names spelled out (never `all`):

```bash
./download_datasets.sh cadets_e3,theia_e3,clearscope_e3,optc_h501,atlasv2_edr <ACCESS_TOKEN>
```

| dataset | download id (`download_datasets.sh`) | dump (compressed / loaded, GB) | Postgres database |
|---|---|---|---|
| CADETS_E3 | `1DGcGBhpavNmXTnCDd_s4NWBNh2n4-6nd` | 1.4 / 10 | `cadets_e3` |
| THEIA_E3 | `1p7HoH5SdMEFi0vkrEtMeG7B-hpw741p6` | 1.1 / 12 | `theia_e3` |
| CLEARSCOPE_E3 | `1v8vsgJxv4OPFVxx8UlYvixDq8FpkIqkd` | 0.6 / 4.8 | `clearscope_e3` |
| optc_h501 | `1046BVjpMql1bb5WHr9yQeB6Uq6RpngbM` | 1.5 / 6.7 | `optc_h501` |
| ATLASV2_EDR | `1t7pW7i23Ry-Z7Tpu2B-E4gezT7Wu74ma` | 0.2 / 1 | `atlasv2_edr` |

Restore with upstream `scripts/load_dumps.sh`, which creates one database per dump file, named
after it (`pg_restore --clean --if-exists --no-owner --no-privileges -d <name> <name>.dump`).

**The optc_h501 database name.** Upstream 2289cd9 configures the optc_h501 database as `optc_501`,
but the restore creates `optc_h501`. HABITUS sets `cfg.dataset.database` (and
`database_all_file`) to the name in `configs/datasets.yaml` programmatically, in
`habitus.data.upstream.orthrus_cfg` and in `scripts/build_graphs.py`; the task hash does not depend
on it. To run upstream PIDSMaker on its own, apply `docs/upstream.patch` (that one rename; the other
local edits of our PIDSMaker worktree are not needed) or restore the dump into `optc_501`.

## 2. Graphs: `habitus data graphs`

```bash
habitus data graphs --ds CADETS_E3 [--python PATH] [--db-host H --db-user U --db-password-file F --db-port 5432]
```

Runs `<paths.pidsmaker_python> scripts/build_graphs.py --db-name <db> orthrus <DS> --artifact_dir
runs/pidsmaker/ ...`: upstream construction (DB -> one networkx MultiDiGraph per 15-minute window,
`torch.save`d) and transformation (with the orthrus config `used_methods: none`, a copy of the
construction output), on CPU, with `CUDA_VISIBLE_DEVICES=''` and `WANDB_MODE=disabled`. Output:
`runs/pidsmaker/transformation/<DS>/transformation/<sha256>/nx/graph_<day>/<t0>~<t1>`.

The task-hash directories of the five hosts start with `380c3a7a` (CADETS_E3), `330ca3db`
(THEIA_E3), `340b203b` (CLEARSCOPE_E3), `c6d5d014` (optc_h501) and `70db4319` (ATLASV2_EDR); the
transformed graphs take 811 MB / 1.4 GB / 180 MB / 2.2 GB / 411 MB.

## 3. Resolver cache: `habitus data resolver`

```bash
habitus data resolver --ds CADETS_E3
```

One pass over `subject_node_table(index_id, path, cmd)`, `file_node_table(index_id, path)` and
`netflow_node_table(index_id, src_addr, src_port, dst_addr, dst_port)` of the restored database ->
`runs/defense/semantics/<DS>.pkl` = `{subject: {id: (path, cmd)}, file: {id: path}, netflow: {id:
(remote_ip, remote_port, local_port)}, host_ip}`. The monitored host's address is detected as the
dominant routable `src_addr` and logged (CADETS_E3 128.55.12.73, CLEARSCOPE_E3 128.55.12.166,
ATLASV2_EDR 192.168.223.128). Cache sizes: 10 / 136 / 11 / 161 / 84 MB.

The card builder looks for the cache under `runs/defense/semantics/`, then under
`<pidsmaker_artifacts>/defense/semantics/`. **It fails hard without one** (`ResolverUnavailable`):
the original code silently fell back to parsing the detector's node labels, which gives different
cards.

## 4. Cards: `habitus data cards`

```bash
habitus data cards --ds CADETS_E3 [--days D1,D2] [--out DIR] [--pidsmaker-artifacts DIR]   # default out: runs/cards
```

Builds every card day of the host (attack-free days, then configured test days; the `card_days` of
`configs/datasets.yaml`, 45 files over the five hosts, including the empty days THEIA_E3 04-02 and
04-06..08 and ATLASV2_EDR 07-20) into `<out>/<DS>/<day>.json.gz` and `<out>/<DS>/links/<day>.json.gz`.
The graphs are read from `--pidsmaker-artifacts DIR` when given (the artifact root a `habitus data
graphs --pidsmaker-artifacts DIR` built), else from the first of `runs/pidsmaker` and
`paths.pidsmaker_artifacts` that holds the task-hash directory (`manifest` searches the same way).
A day whose link file exists is skipped (the link file is written last; delete it to rebuild the
day, or call `habitus.data.cards.build(..., force=True)`). No statistics file is written.

- card file: `{"day", "dataset", "version": 2, "cards": {sig: card}, "nodes": {node_id: sig}}`,
  sig = `sha1(json.dumps(card, sort_keys=True))[:16]`;
- link file: `{"verbs": {"1": "spawn", "2": "write", "3": "exec", "4": "net"}, "links": [[cause,
  effect, code], ...]}` (spawn parent->child, write actor->file, exec file->process, net
  actor->netflow; sorted, self links removed).

The card text, regexes, tables, count bins, `TOP = 5`, `MAX_CMD = 300`, `CARD_VERSION = 2` and the
OS-family strings are byte-identical to the original builder: every answer, score and precedent
cache is keyed on card signatures, so any change orphans them.

## 5. Checks: `habitus data manifest`, `habitus data verify`

```bash
habitus data manifest --ds CADETS_E3 [--pidsmaker-artifacts DIR]   # recompute usable clean days from the graphs
habitus data verify   --ds CADETS_E3 [--cards DIR]   # card / link JSON sha256 vs release MANIFEST.json
```

`manifest` recomputes the usable clean days (train + val dates minus attack dates, keeping days whose
graphs hold >= 10% of the median clean-day bytes) and FAILS when they differ from the frozen lists.
`verify` compares the canonical-JSON sha256 of every rebuilt card and link file with the release
manifest. **Run both after any rebuild.** The frozen day lists, and every cached answer and score,
are valid only for graphs identical to ours.

What is and is not verified:

- Verified: the vendored card builder (`habitus.data.cards`) on Python 3.13 with the upstream
  `pidsmaker.config` package only, from the stored graphs and resolver caches, rebuilt all 45 card
  and link files JSON-equal to the shipped ones (735 s on one process, about 12 minutes).
  `tests/test_data_cards.py` repeats this for ATLASV2_EDR 2022-07-19 when `paths.pidsmaker_artifacts`
  is configured.
- Verified: the upstream config reproduces the task-hash directories of all five hosts, and the
  graph bytes of an independent upstream run are md5-identical to ours for CADETS_E3 2018-04-06 and
  ATLASV2_EDR 2022-07-19.
- Not verified: that pristine 2289cd9 + `docs/upstream.patch` gives the same hash directories and
  graph bytes as the locally patched upstream worktree used for the check above; CLEARSCOPE_E3 graph
  bytes were never compared with an upstream run. Run `habitus data manifest` and `habitus data
  verify`; any difference invalidates the shipped caches.
- Measured, environment not recorded: construction takes 87 s (CLEARSCOPE_E3) to 383 s
  (optc_h501) per host on CPU, transformation about 1 s. The interpreter of those runs was not
  recorded (TBD).

## 6. Node index: `habitus data index`

```bash
habitus data index --ds CADETS_E3 [--cards DIR] [--out runs/nodeindex]
```

One string-free file per card file (all 45, clean, test and empty days, because clean-day null
pools are read too): `{"format": 1, "ds", "day", "nodes": {node_id: sig} (card-file order), "types":
{sig: type}, "kappa": {sig: inv_key}}`, kappa = `signature({type, invariant})`. `Host(ds,
source="nodeindex")` serves nodes / types / identity keys / the test-node universe from it, so
score files can be evaluated without card text.

## Days and universes (frozen in `configs/datasets.yaml`)

| host | clean (usable attack-free) | test | pre-deployment | test-node universe | eval universe |
|---|---|---|---|---|---|
| CADETS_E3 | 04-02..05, 04-07..10 | 04-06, 04-11..13 | 04-02..05 | 281,585 | 281,589 |
| THEIA_E3 | 04-03, 04-04, 04-05, 04-09 | 04-10, 04-12, 04-13 | = clean | 700,902 | 700,902 |
| CLEARSCOPE_E3 | 04-03..05, 04-07..10 | 04-11, 04-12 | = clean | 111,403 | 111,403 |
| optc_h501 | 09-19..22 | 09-23..25 | = clean | 1,489,198 | 1,489,198 |
| ATLASV2_EDR | 07-16..18 | 07-19 | = clean | 152,763 | 152,763 |

THEIA_E3 drops 04-02 and 04-06..08 (no graph bytes) and CLEARSCOPE_E3 drops 04-02 (0.4 MB against a
median of 8.2 MB) from the clean days; ATLASV2_EDR's configured test day 07-20 has no graph (an
empty card file) and is not a test day, but still closes the pre-deployment window. CADETS_E3's
clean days 04-07..10 come after its first test day, so its main configuration uses the k = 2
reference (04-02, 04-03). The CADETS_E3 evaluation universe adds the 4 ground-truth nodes that have
no test-day card (they count as false negatives).

## Redistribution

Cards and links (about 65 MB) contain paths, command lines, IP addresses and e-mail-like strings
from DARPA TC, OpTC and ATLASv2. They live only in the local `release/`; publishing them is the
user's decision (see the release notes).
