# Working preferences

Applies to this repository. Keep the project small, understandable, and useful
for learning. The user can override these defaults in conversation.

## Communicate

- Lead with the result; keep prose short.
- Prefer bullets, tables, formulas, examples, and small diagrams when clearer.
- Ask targeted questions when uncertainty materially affects the outcome.
- State minor assumptions; avoid questions about routine, reversible details.
- Distinguish existing behavior, proposed work, and measured results.

## Implement

- Prefer clear names, small functions, explicit inputs/outputs, and few dependencies.
- Use type hints and concise Google-style docstrings where useful.
- Use classes for state or a useful interface; avoid frameworks for hypothetical needs.
- Check for a simpler solution before adding abstraction or another file.
- Share logic when it prevents real inconsistency; allow small duplication when
  sharing would couple unrelated behavior or make the code harder to read.
- Experiment in `notebooks/`; promote selected reusable behavior into `src/`.
- Keep adjustable settings in `config/`; preserve `data/raw/`.
- Add tests under `tests/unit/` and `tests/integration/` as implementation grows.
- Docker is available for API runs, integration tests, isolation, and volumes.
  Use it where helpful; keep the first notebook baseline runnable locally.
- Reuse existing containers/volumes where appropriate; avoid unnecessary copies.
- Preserve unrelated edits and stored data; do not prune containers or volumes
  containing unrelated work.

## Check coherence

- After each coherent change, inspect affected callers, imports, configurations,
  tests, documentation links, and commands. Check only what the change can affect.
- For renames, search references and check packaging/entry points as well as imports.
- Run meaningful focused checks; report gaps honestly. Do not create tests merely
  to mirror a trivial implementation or repeatedly rerun unchanged checks.
- Review the diff for accidental scope, redundant logic, and simpler alternatives.

## Keep documentation lean

- Code/configuration/tests define behavior; docs explain navigation and intent.
- Keep the repository guide accurate; link rather than repeat explanations.
- Add a glossary only when domain language actually needs explanation.
- Add decision records only for substantive alternatives tried and rejected:
  evidence, choice, reason. Do not document every visible code choice.
- Add skills or specialized agents only for a concrete recurring need.
