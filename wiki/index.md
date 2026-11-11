# XAIP Knowledge Wiki — Start Here

**Summary**: Entry point for the XAIP knowledge base: what the project is, features, models, per-mode flows, and output schemas.
**Tags**: #index #meta
**Created**: 2026-11-09
**Last Updated**: 2026-11-09

---

The wiki documents **architecture and rationale that is verified against source code**. Live sources (code, config files, `GUIDELINE.md`) remain the source of truth for current values and implementation details — the wiki points at them instead of copying.

## Pages

| Page | Answers |
|---|---|
| [[project/overview|Overview]] | What this project is, architecture, config flow, how it runs |
| [[project/features|Features]] | How many features are supported — verified inventory with counts |
| [[project/models|Models]] | How many models are supported — registry truth, tasks, datasets, weights |
| [[project/precision-quantization|Precision & Quantization]] | Precision labels, weight mapping, IO casting, INT8/BF16 variants, per-model validated behaviors |
| [[pipeline/mode-flows|Mode Flows]] | High-level (macro) flow of each mode: inference, tracking, debug, evaluate |
| [[pipeline/debug-internals|Debug Internals]] | ONNX graph rewrite, topological ordering, disk streaming, profiling, cleanup |
| [[pipeline/output-schemas|Output Schemas]] | Exact output artifacts and schemas per model-task × mode |

## Conventions (follow on every change)

1. **Verified only.** A claim is written only after it is checked against the live source code (file + symbol). If it cannot be verified, do not write it — record a pointer to the source instead and let the reader read the code.
2. **Pointer, don't copy.** Volatile facts (paths, counts, version pins, file-name templates) are written as "what it is + where to read it". Locations: counts → the registry/config that defines them; formats → the code that builds the string. Never paste implementation code, never cite line numbers (they shift).
3. **Stable facts** (architecture decisions, established file roles) may be stated directly; update on next ingest when they change.
4. **Path scope.** Files inside this repo are referenced **relative to the repo root** (e.g. `AI_Script/core/factory.py`). Machine-anchored paths (weights on another machine, GUIDELINE sections) use their absolute form.
5. **New pages** follow [[template|template]] (front matter + `## Content` + `## Related Notes`), use only new-file categories already present (`project/`, `pipeline/`), and get an index entry above. No audit log is kept by design.
6. Wikilinks use the `[[category/page|Display]]` form.

Wiki-maintenance helpers exist in-repo: `README.md` (introduction), `GUIDELINE.md` (run guide § 0–4), `AGENTS.md` (agent contract + hardware constraints).
