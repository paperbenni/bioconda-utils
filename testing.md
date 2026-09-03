# Testing bioconda-utils against patched conda / conda-build / conda-libmamba-solver

This document describes a local testing setup that removes three proven
performance bugs from the conda stack that bioconda-utils drives during
`bioconda-utils build`. None of these patches are in a release yet, so they
must **not** go into the official image — they are applied at *test time*
only, either on the host via `PYTHONPATH`, or inside a throwaway docker image
derived from the official one.

Measured effect on a real `seqtk` docker build (this machine, warm
repodata cache, token active):

| configuration | wall | host pre-render | in-container |
|---|---|---|---|
| stock 26.3 everywhere | 535s | ~250s | ~288s |
| official image bumped to 26.7 (released only) | 347s | ~133s | ~193s |
| host patched + official 26.7 image | 214.5s | ~15s | ~190s |
| host patched + custom image, cold caches | 214.7s / 221.5s | ~15s | ~190s |
| host patched + custom image, warm caches | 123.4s | ~15s | ~100s |
| + render.py: skip index re-lookup (solved record has fn) | **86.7s** | ~15s | ~70s |

Build strings are identical across all runs (`...-hb7acf71_1`), i.e. the
patches are semantics-preserving.

## 1. The local patch sets

Shallow clones live next to this repo, each on its own branch:

| clone | branch | fixes |
|---|---|---|
| `../conda` | `fix-shard-url-token-mangling` | (a) `strip_pkg_extension` recognizes `.msgpack.zst` so sharded-repodata URLs round-trip when a binstar token is configured (with-subdir variant of conda#16516); (b) `Index.copy()` is lazy (Python ≥3.13 `UserDict.copy` reads `.data` and realizes everything) |
| `../conda-build` | `fix-index-truthiness-realize` | (a) `install_actions()` uses `if index is not None:` instead of `if index:` — the truthiness check forced `Index._realize()` (~105s / 1.3M PackageRecords) on every conda-build process; (b) `execute_download_actions()` downloads solved records directly instead of scanning the whole index (`for rec in index` → realize ~105s) or re-looking them up via `index[prec]` (which pulls the full monolithic repodata.json.zst of that subdir, ~57MB for conda-forge/linux-64, in every fresh container) — this fires in the *build* phase (run_exports fetch) of every package, including inside docker containers |
| `../conda-libmamba-solver` | `lazy-index-channel-discovery` | (a) discover conda-build's `file://` channels via `index.expanded_channels` instead of iterating all records; (b) `_called_from_conda_build()` uses `is not None` instead of truthiness — both triggered the same realize |

Background: conda/conda-build#4961 ("conda-build traverses the whole index
greedily"), nominally fixed by the 2024 lazy-index work, but the realize is
merely *deferred* to the first truthiness check / iteration — the patches
above remove the remaining triggers. The realize accounts for ~105s per
conda-build process on conda-forge + bioconda (1.28M records).

## 2. Running bioconda-utils against the patched host stack

The pixi env keeps the *released* stack; the clones shadow it per-invocation:

```bash
export PYTHONPATH=../conda:../conda-build:../conda-libmamba-solver
bioconda-utils build <recipes> <config.yml> ...   # any command
```

Everything spawned by that process (conda-build, the solver plugin,
multiprocessing children) inherits the shadowing. Verify with:

```bash
PYTHONPATH=../conda:../conda-build:../conda-libmamba-solver python -c \
  "import conda, conda_build, conda_libmamba_solver; print(conda.__file__, conda_build.__file__)"
```

(paths should point into the clones)

Note: the `../conda` clone also fixes the token-URL mangling, so builds run
with `PYTHONPATH` work with a binstar token configured. On the *released*
26.7 stack the equivalent workaround is `CONDA_ADD_ANACONDA_TOKEN=false`
(breaks private-channel auth, fine for public channels); without *either*,
sharded repodata 404s and each solve falls back to ~17s monolithic parsing.

## 3. The custom test image (patched stack inside docker)

The official image must contain released code only, so testing the in-container
patches uses a derived image. The patched sources are exported with
`git archive` (clean tree, no `.git`) and shadowed via `PYTHONPATH`:

```bash
# 1. export clean trees of the three patched branches
mkdir -p /tmp/opencode/imgctx && cd /vol/storage1/home/benjamin/workspace
for r in conda conda-build conda-libmamba-solver; do
    mkdir -p /tmp/opencode/imgctx/$r
    git -C $r archive HEAD | tar -x -C /tmp/opencode/imgctx/$r
done

# 2. derived Dockerfile (/tmp/opencode/imgctx/Dockerfile)
```

```dockerfile
FROM bu-build-env:x86_64-26.7        # or the plain bu-build-env:x86_64

COPY conda /opt/patches/conda
COPY conda-build /opt/patches/conda-build
COPY conda-libmamba-solver /opt/patches/conda-libmamba-solver
ENV PYTHONPATH=/opt/patches/conda:/opt/patches/conda-build:/opt/patches/conda-libmamba-solver

# conda main imports backports.zstd on python < 3.14 (stdlib module lands in 3.14)
RUN /opt/conda/bin/pip install --no-cache-dir backports.zstd
```

```bash
# 3. build (--network host: docker's default bridge has no DNS on this machine)
docker build --network host -t bu-build-env:x86_64-patched -f /tmp/opencode/imgctx/Dockerfile /tmp/opencode/imgctx/
```

### 3b. Warm-cache variant (recommended for repeat builds)

Build containers run with `--rm`, so conda's repodata/shards caches are cold
on **every** run — repodata downloads and per-solve index loads never get
faster on their own. Bake them into a derived image instead:

```dockerfile
FROM bu-build-env:x86_64-patched

# warm shards indexes + monolithic repodata caches for both channels
RUN . /opt/conda/etc/profile.d/conda.sh && \
    conda create --dry-run --yes -n _w zlib >/dev/null 2>&1; \
    conda env remove --yes -n _w >/dev/null 2>&1; \
    /opt/conda/bin/python -c "from concurrent.futures import ThreadPoolExecutor as T; from conda.core.subdir_data import SubdirData; from conda.models.channel import Channel; urls=['https://conda.anaconda.org/'+c+'/'+s for c in ('conda-forge','bioconda') for s in ('linux-64','noarch')]; f=lambda u: SubdirData(Channel(u)).repo_fetch.fetch_latest_parsed(); list(T(max_workers=4).map(f, urls))"; \
    chmod -R a+rwX /opt/conda/pkgs
```

```bash
docker build --network host -t bu-build-env:x86_64-patched-warm -f /tmp/opencode/imgctx/Dockerfile.warm /tmp/opencode/imgctx/
```

Notes:
- the `chmod -R a+rwX` is required: conda updates cache state (mtime/etag)
  even on cache hits, and the warmup layer runs as root while builds run as
  another user — root-owned caches cause `PermissionError` mid-build.
- conda-forge repodata changes constantly; baked caches get revalidated
  (conditional GET, ~10s) or re-downloaded when upstream changed — still far
  cheaper than cold monolithic fetches (~57MB + parse per subdir).
- same trick is a candidate for the official image (uses released code only).

## 4. Test builds

```bash
# official released stack everywhere (the image bioconda-utils ships):
docker build --network host -t bu-build-env:x86_64-26.7 .   # after `pixi run regenerate-requirements`

# unpatched host + released image:
bioconda-utils build . config.yml --packages <pkg> --docker --force \
    --docker-base-image bu-build-env:x86_64-26.7 \
    --repodata-cache /tmp/repodata-cache.pkl -t 16

# patched host + released image:   prepend PYTHONPATH (see §2) to the same command
# patched host + patched image:    prepend PYTHONPATH and use --docker-base-image bu-build-env:x86_64-patched
```

`--repodata-cache` (pickle, 8h TTL) removes the ~43s repodata download on
warm starts; it is a bioconda-utils feature and orthogonal to the conda patches.

## 5. Profiling harness

Scratch scripts used for the measurements (in `/tmp/opencode/prof/`):

- `instrumented.py` — per-solve attribution; wraps `Index._realize`,
  `LibMambaIndexHelper.__init__`, `solve_final_state`, `_install_actions`
  and prints per-recipe totals:
  `PYTHONPATH=... python instrumented.py <recipe-name>...`
- `whorealizes3.py` — prints a stack trace at every `Index._realize()` call;
  use it to find new realize triggers after version bumps
- `phases.py <log>` — extracts build-phase timings from a build log
- `fastcopy.py` — the old single-process monkeypatch version of the fixes
  (superseded by the clones; kept for reference)

## 6. Status of the fixes

All four commits are minimal and reference conda/conda-build#4961; they are
candidates for upstream PRs. Until they (or equivalents) are released:

- official images: released versions only (currently 26.7.x, bumped via
  `pixi.toml` → `pixi run regenerate-requirements` → rebuild)
- unreleased patches: only via `PYTHONPATH` on the host or the derived test
  image, as described above

## 7. Production caching (implemented)

`bioconda-recipes` CI (PR.yml, Bulk.yml, nightly.yml) now uses three caches;
all are additive and guarded so they no-op on releases lacking support:

1. **Host repodata cache** — `--repodata-cache /opt/bioconda-repodata-cache/`
   (gzip pickle, 8h TTL) persisted via `actions/cache/restore`/`save`
   (unique save key per run, prefix restore). Saves the ~45s channel
   repodata download per job.
2. **Container package cache** — `BIOCONDA_UTILS_CONTAINER_PKGS_CACHE=/opt/
   bioconda-container-pkgs` makes docker_utils bind-mount that host dir at
   `/opt/conda/pkgs` in every build container of the job. Containers are
   `--rm`, so without it every recipe re-downloads repodata/shards and its
   build/host env packages. Measured (seqtk): cold 186s → warm **57s**
   within one job; container phase 190s → 27s.
3. **`--threads`** — parallel DAG build (45s → ~4s on an 11k-recipe folder).

The flag guard (`bioconda-utils build --help | grep -- --repodata-cache`)
keeps the workflows working with releases that predate the flags; the env
var is a no-op there. Requires bioconda-utils ≥ this commit to take effect
(bioconda-common `BIOCONDA_UTILS_TAG` bump).

Ops notes: mount dirs are `chmod 777` because build containers run as uid
9001 (conda) and conda writes cache state even on hits; bulk jobs should
watch runner disk (cache grows to ~1GB/job with extracted toolchains) and
can prune in a cleanup step.

Purity: caches only change *when the channel was observed*, never the built
artifact — build strings were byte-identical (`hb7acf71_1`) across every
configuration tested (stock/patched × cold/warm).
