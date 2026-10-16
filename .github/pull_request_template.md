<!--
Thanks for contributing to metalog-jax!

Title: one imperative sentence that stands on its own, e.g.
"Coerce list coefficients to a JAX array in MetalogBase". It becomes the
commit subject and the starting point for release notes.

Base branch: open feature and fix PRs against `development`. Only `development`
may merge into `main` (enforced by the Main Source Check workflow), and every
merge into `main` publishes to PyPI.

Keep each PR to one logical change. Put refactors in their own PR.

Sections marked "(if applicable)" can be deleted when they don't apply.
Comments like this one are hidden in the rendered PR.
-->

## Summary

<!--
What does this change, and why? Start with the problem, then the approach.
Include what a future reader can't get from the diff: why this approach,
alternatives you rejected, and known limitations.
-->

Fixes #

## Type of change

- [ ] Bug fix
- [ ] Regression fix (worked in an earlier release; name the version: )
- [ ] New feature
- [ ] Breaking change (public API, accepted input types, defaults, or saved JSON format)
- [ ] Numerical change (same inputs now give different outputs)
- [ ] Performance
- [ ] Documentation only
- [ ] Refactor, tests, CI, or dependencies (no user-facing change)

## Test plan

<!--
The exact commands you ran and what they showed. For a bug fix, paste the
reproduction and add a test that fails on `development` and passes with this PR.
-->

Environment: <!-- OS, Python, jax/jaxlib versions, CPU or GPU -->

- [ ] New or updated tests cover the change (`tests/unit`, `tests/integration`, or `tests/feasibility`)
- [ ] A bug fix includes a test that fails without the fix
- [ ] `make quality-gate` passes locally (ruff format and lint, ty, radon MI grade A and CC grade B or better, pytest with at least 75% coverage, license check, semgrep, bandit)

## Numerical correctness (if applicable)

<!-- For changes to fitting, ppf/pdf/cdf/q, moments, sampling, or feasibility. -->

- [ ] Results are checked against an independent reference: <!-- equation or proposition in Keelin (2016) or Baucells, Chrisman, Keelin and Xu (2025); R rmetalog; the reference implementation credited in NOTICE; a closed form; or scipy -->
- [ ] Tolerances in new assertions are explained in a comment or here
- [ ] All four boundedness types are tested, not just `UNBOUNDED`
- [ ] Every affected fit method (OLS, Lasso, Feasible) is tested
- [ ] float64 precision holds end to end (no float32 constants, casts, or default dtypes slipping in)
- [ ] Densities stay strictly positive, and `check_feasibility` still certifies Feasible fits
- [ ] If `metalog_jax/feasibility/` changed: both `TermOrder.KEELIN_2016` and `TermOrder.METALOG_2` are handled

## JAX compatibility (if applicable)

- [ ] Works under `jax.jit` and `jax.vmap` (grid search vmaps over datasets, L1 penalties, and term counts)
- [ ] No Python control flow, `float()`, `bool()`, or `.item()` on traced values; static arguments such as `num_terms` stay static
- [ ] Changes to `flax.struct.dataclass` classes survive pytree unflattening, which re-runs `__init__` and `__post_init__` with tracers, `None`, or `ShapeDtypeStruct` as field values
- [ ] Validation that must run inside transforms uses `checkify`, not Python `assert`
- [ ] Inputs that worked before still work (Python lists and tuples, NumPy arrays, JAX arrays)

## Compatibility and release note

**Breaking change?** <!-- "No", or what breaks and how users migrate. Covers public names, accepted input types, defaults, and the dumps()/save() JSON format: files saved by earlier versions must still load(). -->

<!-- One user-facing line for the release notes inside the block, or NONE -->
```release-note

```

## Performance (if applicable)

<!--
For changes to fitting, feasibility, grid search, or sampling: before and after
numbers, the hardware, and the command, e.g.
uv run python benchmarks/feasible_scaling/bench.py MODE N K REPS
-->

## Documentation

- [ ] Public functions and classes have Google-style docstrings (ruff `D` rules)
- [ ] A new module has an `automodule` entry in `docs/source/api/modules.rst`
- [ ] Notebooks in `docs/source/` are re-run and saved with outputs if results changed (the docs build does not execute them)
- [ ] `README.md` and `examples/` match the new behavior
- [ ] `make docs` builds

## Dependencies and licensing (if applicable)

- [ ] New dependencies are justified in the summary, `uv.lock` is updated, and `make license-check` passes
- [ ] Code adapted from a third party is credited in `NOTICE` with its source and license

## AI assistance

<!--
"None", or the tools used and what for (code, tests, docs, research).
Either way, the author can explain every line of the diff and has personally
checked the numerical results.
-->

## Notes for reviewers

<!-- Where to start reading, what you're unsure about, and follow-up work deliberately left out. -->

<details>
<summary>Release PRs only (<code>development</code> → <code>main</code>)</summary>

Merging runs Quality Gate → Build → Publish to PyPI, then resets `development` to `main`.

- [ ] Version in `pyproject.toml` is bumped (PyPI rejects a version that already exists): patch for fixes, minor for features, major for breaking changes
- [ ] `uv.lock` is refreshed (`uv lock`)
- [ ] `make ci` passes
- [ ] `development` contains nothing that shouldn't ship
- [ ] The summary lists every included PR with its release-note line

</details>
