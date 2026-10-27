# Working preferences

Applies to this repository. Keep the project small, understandable, and useful
for learning. The user can override these defaults in conversation.

## Communicate

- Lead with the result; keep prose short.
- Before tool-backed or sustained work, send a short user-facing progress update.
- While actively working, send an update at least every 60 seconds (aim for 30–60),
  including analysis, tool runs, waits and reviewer work. Tool output and subagent
  messages do not count.
  State the current action, meaningful finding/blocker and next step; also update
  when scope or plans change. Use bounded waits where possible and explain an
  unavoidable long tool delay when it returns.
- Begin completion summaries with a one-line goal, followed by the result.
- Persist general working preferences in agent instructions for future sessions;
  distinguish repository-local instructions from user-wide preferences.
- Prefer bullets, tables, formulas, examples, and small diagrams when clearer.
- Ask targeted questions when uncertainty materially affects the outcome.
- State minor assumptions; avoid questions about routine, reversible details.
- Distinguish existing behavior, proposed work, and measured results.

## Implement

- Prefer clear names, small functions, explicit inputs/outputs, and few dependencies.
- Use type hints and concise Google-style docstrings where useful.
- Use classes for state or a useful interface; avoid frameworks for hypothetical needs.
- Check for a simpler solution before adding abstraction or another file.
- Keep foundational tooling, packaging, editor configuration, and package scaffolding
  stable during file/documentation cleanup. Propose changes to those separately,
  with their impact explained.
- Share logic when it prevents real inconsistency; allow small duplication when
  sharing would couple unrelated behavior or make the code harder to read.
- Prefer abstract contracts with separate concrete implementations for replaceable
  experiment stages. Compose actual component objects; keep name resolution at the
  configuration boundary and fitting orchestration separate from estimators.
- Reuse native estimator constructors, parameters, cloning, and fit/predict through
  thin native-first inheritance where practical; avoid parallel wrapper frameworks.
- Experiment in `notebooks/`; promote only selected inference behavior into `src/`.
- Keep `src/` focused on serving the already-trained model. Training, fitting
  policies, candidate models, selection and artifact export belong in `notebooks/`.
- Keep experiment entry points short; put cohesive helpers in `notebooks/utils/`.
  Split files by responsibility as they grow; use classes only for useful state
  or interfaces, not merely to divide a long script.
- Keep adjustable settings in `config/`; preserve `data/raw/`.
- Keep Make targets as fixed shortcuts without argument forwarding or duplicated
  CLI defaults. Use the underlying CLI directly for occasional overrides.
- Mirror source ownership under `tests/`, such as `tests/notebooks/utils/` and
  `tests/src/banking_query_triage/`. Use pytest markers to select integration tests
  and parametrize to cover input cases; do not divide directories by test scope.
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

## Independent review after implementation

- At the end of each implementation cycle or substantive subcycle, run a
  separate reviewer agent with fresh context (`fork_turns="none"`).
- Give the reviewer the intended behavior, constraints, and files to inspect;
  ask it to challenge correctness, assumptions, clarity, and maintainability.
- Include a project-wide simplification/ablation pass: what can be removed,
  merged, or replaced to make behavior easier to understand and debug?
- Favor quality and expected behavior over code volume, file count, or prose.
- Discuss material disagreements, apply justified changes, and seek agreement
  before declaring the cycle complete. Report unresolved concerns honestly.
- Delete obsolete or unnecessary code/files when safe and in scope; preserve
  unrelated user work and relevant experiment evidence. Record meaningful
  rejected approaches briefly in an architectural decision note when useful.
- Keep review feedback concise and evidence-based. Reviewers do not recursively
  launch reviewers; the implementing agent owns this review loop.
