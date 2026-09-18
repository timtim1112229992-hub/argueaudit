# argueaudit

A field-separated census of argument components in school science writing.

Most measurement of written argument scores continuous prose, which means a
pupil who reasoned badly and a pupil who never reasoned at all can receive the
same mark. Where an instrument instead stores the parts of an argument in
separate fields, an empty field is a record of omission rather than of
inarticulate expression. This package is built for corpora of that second kind:
it codes each stored field by deterministic ruleset, estimates component rates
with exact and resampled intervals, and writes the provenance needed to
re-execute the whole of it.

The package is released so that the procedure can be inspected and re-run. The
corpus it was written for is writing by primary-age children, is held under
restricted custody, and is not part of this repository.

## What it computes

Each stored field is registered with the response format the interface used to
ask for it and the epistemic act it demanded, the two being recorded separately
so that completion can be read against format rather than against demand alone.
Formats are `menu`, `exemplar`, `frame`, `preset` and `open`; demands are
`describe`, `claim`, `interpret` and `justify`. The register is in
`src/argueaudit/config.py` and is taken from the instrument documentation, which
keeps the empirical distinctness measure an independent check on it rather than
a restatement of it.

| Model | Question it asks |
|---|---|
| M1 | Populated and substantive rate per argument component, with Clopper-Pearson intervals |
| M2 | Whether the components differ, by Cochran Q with exact McNemar for each pair |
| M3 | Agreement between the two rulesets, with a cluster bootstrap interval |
| M4 | Whether entries of different types differ in genre |
| M5 | One stage's completion against every other, by exact sign test |
| M6 | Whether the instrument's own controls predict the presence of a warrant |
| M7 | Time on stage and sequence position as competing explanations |
| M8 | Reference to particular materials and properties, by component |
| M9 | The substantive rate across a declared threshold sweep |
| M10 | Cluster bootstrap over groups for every non-exact quantity |
| M11 | Rule-of-three upper bound wherever a substantive count is zero |
| M12 | Populated rate by response format and by epistemic demand |
| M13 | Recorded completion against composed content |
| M14 | The instructional support present while the writing was done |

Every model returns a result object carrying its estimand, its values and an
`estimable` flag. A predictor with no variance supports no test, so each
registered predictor is checked for variance before anything is fitted, and a
constant one is reported as constant rather than entered into a procedure that
cannot estimate from it. The same applies to a model whose fitted coefficient
would describe an interface template rather than the writing: it is reported as
non-estimable, with the grounds recorded, instead of being quietly replaced by a
number that would read like a result. Two of the registered procedures are
returned this way on the corpus the package was written for.

## Coding is by ruleset, not by hand

Ruleset A decides from an entry's character composition; ruleset B decides from
its marker content. Neither is derived from the other, so their agreement is
informative rather than tautological, and disagreements are resolved by a rule
declared in advance: the stricter decision stands, which makes a positive
finding harder to obtain rather than easier.

These are rulesets, not people. The package does not claim human double coding,
and M3 reports agreement between two operationalisations of one construct rather
than between two coders. The coding decisions are written out as data, with
every text column removed, so that a clean re-execution reproduces each reported
figure without repeating any judgement.

## Custody

Text composed by participants never leaves the machine that holds the restricted
corpus. What the package writes for release is the column map recording which
fields the analysis was permitted to read, a digest per analysed record, the
coding decisions with every text column dropped, and a run manifest recording
code commit, interpreter, package versions, seed and every parameter value.

`tests/test_release_guard.py` fails if any released artefact carries participant
text, if the manifest writer is handed a key outside the release policy, or if a
file this policy excludes has been staged into the repository.
`tests/test_synthetic_guard.py` fails if a run on synthetic data could overwrite
results computed from the corpus, or publish a provenance package describing
records that do not exist.

The corpus location is supplied through `ARGUEAUDIT_DATA_DIR` and is recorded
neither in the repository nor in the manifest.

## Running it

```bash
pip install -e ".[test]"
python scripts/make_synthetic.py     # writes synthetic/, already committed
python scripts/run_pipeline.py       # runs against synthetic/ when no corpus is set
pytest
```

With no `ARGUEAUDIT_DATA_DIR` set, the pipeline reads the synthetic corpus in
`synthetic/`. That corpus has the schema, field register and shape of the real
one and nothing else in common: its entries are assembled from a small English
vocabulary by a seeded generator, and its rates were chosen to differ from the
observed ones, so a reader who runs the package cannot mistake its output for a
replication. Every run against it is stamped `source: synthetic`, it will not
overwrite results computed from the corpus, and it will not publish provenance.

Against the restricted corpus:

```bash
export ARGUEAUDIT_DATA_DIR=/path/to/corpus
python scripts/run_pipeline.py
python scripts/render_assets.py
```

## Layout

```
src/argueaudit/     ingest, derive, rubric, models, figures, tables, provenance
scripts/            pipeline runner, asset renderer, synthetic corpus generator
synthetic/          the committed stand-in corpus
provenance/         column map, record digests, run manifest
tests/              analysis tests, release guard, synthetic guard
```

`outputs/` is written at run time and is excluded from the repository, because
results belong to the paper that reports them rather than to the archive of the
procedure that produced them.

## Licence

MIT. See `LICENSE`.
