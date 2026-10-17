# Contributing to metalog-jax

Thanks for helping improve metalog-jax. This guide covers how to report problems, set up a development environment, make a change that passes CI, and open a pull request. The rules here are the ones the repository actually enforces, through its Makefile, CI workflows, branch rulesets and pull request template.

## The short version

- Open pull requests against **`development`**. Only `development` can merge into `main`, and every merge into `main` publishes a release to PyPI.
- Keep each pull request to one logical change. Put refactors in their own pull request.
- Run `make quality-gate` before asking for review. Pull requests with failing CI are not reviewed.
- A bug fix includes a test that fails without the fix.
- Fill in the pull request template, including the AI assistance section.

## Reporting bugs and requesting features

Search the [existing issues](https://github.com/tjefferies/metalog_jax/issues) first. GitHub Discussions is disabled, so use an issue for questions too.

A good bug report has:

- **A minimal script** that reproduces the problem, with the data or coefficients inline.
- **Expected and actual results**, including the full traceback.
- **Your environment.** This prints everything needed:

  ```bash
  python -c "import sys, platform, importlib.metadata as md; print(sys.version.split()[0], platform.platform()); [print(p, md.version(p)) for p in ('metalog-jax', 'jax', 'jaxlib', 'flax', 'chex', 'numpy')]; import jax; print('backend', jax.default_backend())"
  ```

- **The last version that worked**, if it's a regression.

For a feature request, describe the use case. If it involves a new algorithm or statistic, cite the paper and the equation or proposition it comes from.

Looking for something to work on? Try issues labeled `good first issue` or `help wanted`.

## Reporting security issues

Don't report vulnerabilities in a public issue. Use **Report a vulnerability** on the repository's [Security tab](https://github.com/tjefferies/metalog_jax/security) to reach the maintainer privately.

## Development setup

**Prerequisites:**

- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/)
- `make` and `git`
- [pandoc](https://pandoc.org/installing.html), only for building the docs
- [syft](https://github.com/anchore/syft) and [grype](https://github.com/anchore/grype), only for `make sbom`

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/<your-username>/metalog_jax.git
cd metalog_jax
git checkout development
make install        # uv sync --all-groups
make test-quick     # about 3 minutes on a laptop CPU
```

`make help` lists every target. `make install` installs the package in editable mode along with these dependency groups from `pyproject.toml`:

| Group | Contents |
|---|---|
| `dev` | marimo, for the example apps |
| `lint` | ruff, ty, interrogate, pydoclint |
| `test` | pytest, pytest-cov, absl-py, coverage, radon, xenon, plus scipy and scikit-learn as reference implementations |
| `security` | bandit, semgrep |
| `license` | pip-licenses |
| `docs` | sphinx, nbsphinx, ipykernel, kaleido and friends |

Runtime dependencies are kept to four: `chex`, `flax`, `jax` and `plotly`. The optional `gpu` extra installs `jax[cuda12]`. The test suite and CI run on CPU.

## Branches and workflow

```
your branch ──PR──▶ development ──release PR──▶ main ──▶ PyPI + documentation
```

- **Branch from `development`.** Name branches by intent, for example `feature/<topic>`, `fix/issue-<number>-<topic>`, `docs/<topic>` or `deps/<topic>`.
- **Keep your branch current** with `development` before asking for review.
- **What GitHub enforces:**
  - `development`, the default branch, can't be force-pushed or deleted, and changes reach it through pull requests.
  - GitHub Code Quality checks pull requests into `development`, and error-level findings block the merge.
  - `main` accepts pull requests only from `development` in this repository (the `Main Source Check` workflow), and only once the `quality-gate` check passes.
- **Merged branches** are deleted automatically.

## Making a change

### Code style

- **Formatting:** ruff (`make format`). CI runs `ruff format --check` on `metalog_jax/` and `tests/`.
- **Linting:** ruff with pydocstyle (`D`, Google convention), isort (`I`, `metalog_jax` as first-party) and pyflakes (`F`). Lint covers `tests/` too, so test classes and methods need docstrings.
- **Module header:** every module in `metalog_jax/` and `tests/` has the line `# Copyright: Travis Jefferies 2026` and a module docstring. Package modules list their `Classes:` and `Functions:` and add a `See Also:` section.
- **Docstrings:** [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) for every public module, class, method and function.
  - **Summary line:** one line of 80 characters or fewer, ending with a period. Functions use the imperative ("Return the …", "Fit a …").
  - **Sections:** add `Args:`, `Returns:` and `Raises:` unless the name and signature make a one-line docstring enough. Argument types go in the signature, not the docstring.
  - **References:** algorithms cite their source in a `References:` section (Keelin 2016, or Baucells, Chrisman, Keelin and Xu 2025).
- **Docstring checks:** `make pydoclint` checks that every `Args:`, `Returns:` and `Raises:` section matches its signature and body. Its configuration is under `[tool.pydoclint]` in `pyproject.toml`.
  - **Indirect exceptions:** pydoclint can't see exceptions raised by chex assertions or by functions you call. If you document one of those under `Raises:`, put `# noqa: DOC502` after the docstring's closing quotes. Use `DOC503` instead when the function also raises something directly. The existing cases are marked the same way.
- **Docstring coverage:** `make interrogate` measures docstring coverage of `metalog_jax/` and fails below 95% (`[tool.interrogate]` in `pyproject.toml`). Private helpers count; functions nested inside another function, such as `jax.lax` loop bodies, don't. Give every new function, class and method a docstring, private ones included (a one-line summary is enough).
- **Types:** type hints on every signature. Use `chex.Array`, `chex.Numeric` and `chex.Scalar` for array values.
- **Type checking:** `ty` checks `metalog_jax/` (not `tests/`). A few rules are relaxed in `pyproject.toml` because JAX's type stubs are incomplete. Don't relax more rules or add ignore comments without explaining why in the pull request.
- **Complexity:** CI requires cyclomatic complexity grade B or better for every function, grade A for every module and on average, and maintainability grade A for every file. `make xenon` enforces the complexity grades. `make radon` only prints grades, so check maintainability the way CI does:

  ```bash
  uv run radon mi metalog_jax/ -s | grep -E "^metalog_jax.* - [B-F]"   # should print nothing
  ```

- **Security:** semgrep (registry rulesets plus `.semgrep.yml`) and bandit scan `metalog_jax/`. Don't use any of these:
  - `eval` or `exec`
  - `pickle` loads
  - `os.system` or `subprocess` with `shell=True`
  - `yaml.load` without a safe loader
  - MD5 or SHA-1
  - the `random` module (use `jax.random`)
  - hardcoded secrets
  - leftover `breakpoint()` or `pdb` calls

  Validate inputs with chex assertions or `checkify`, not bare `assert`.

### JAX and numerical conventions

- **Precision:** `metalog_jax/__init__.py` enables float64 (`jax_enable_x64`) on import, because the fits, exact rational tables and feasibility certificates need it. Don't change JAX configuration anywhere else, and don't introduce float32 constants or casts.
- **Data containers:** they're immutable `flax.struct.dataclass` types; use `.replace()` to change a field. `__post_init__` also runs whenever JAX rebuilds the object (`jit`, `vmap`, `jax.tree.map`, `eval_shape`). At that point a field can hold a tracer, `None` or a `ShapeDtypeStruct`, so only transform values whose Python type you can identify, such as a list.
- **Input validation:** use chex assertions and the helpers in `metalog_jax.utils`, such as `assert_float_array` and `assert_probability_range`. Validation that must run inside a transform uses `jax.experimental.checkify`. Build `MetalogInputData` with `MetalogInputData.from_values`; direct construction is blocked.
- **Transforms:**
  - Decorate with `@jax.jit`, and pass arguments that determine shapes through `partial(jax.jit, static_argnames=...)`.
  - Don't branch in Python on traced values or call `float()`, `bool()` or `.item()` on them.
  - Keep shapes static so code stays `vmap`-able. Grid search vmaps over datasets, L1 penalties and term counts, and every function in `metalog_jax.feasibility` must work under `jit` and `vmap`.
- **Term ordering:** the feasibility routines support `TermOrder.KEELIN_2016` (the library's ordering) and `TermOrder.METALOG_2`. Changes there must handle both.
- **Public API:** export new public names through the subpackage `__init__.py` and its `__all__`. A new module needs an `automodule` entry in `docs/source/api/modules.rst`.
- **Correctness:** check new numerical results against an independent reference: the papers above and the reference implementation credited in `NOTICE`, a closed form, or scipy.

### Tests

- **Layout:**
  - `tests/unit/` mirrors the package, with `base/` and `regression/` subdirectories.
  - `tests/integration/` covers end-to-end fitting, grid search and Kolmogorov-Smirnov goodness of fit.
  - `tests/feasibility/` covers `metalog_jax.feasibility`.
- **Style:** test classes subclass `parameterized.TestCase` or `absltest.TestCase`. Use `@parameterized.named_parameters` for cases and `numpy.testing` for comparisons.
- **Fixtures:** keep shared fixtures such as parameters and coefficients as class attributes, not module globals.
- **Docstrings:** keep them to one concise line.
- **Issue references:** refer to issue numbers in docstrings only, never in test or variable names, for example `"""Regression tests for issue #9: ..."""`.
- **Coverage of cases:** test all four boundedness types and every fit method the change affects. Explain any tolerance in a comment.
- **Bug fixes:** add a test that fails on `development` and passes with the fix.

```bash
make test-quick                                                              # whole suite, no coverage
uv run pytest tests/unit/test_base.py -k MetalogQuantileAliasTest            # one test class
make test                                                                    # with coverage, fails under 75%
uv run pytest tests/ --cov=metalog_jax --cov-branch --cov-fail-under=75      # as CI measures it (branch coverage)
```

### Documentation

- **API reference:** generated from docstrings by Sphinx autodoc (Google style through napoleon).
- **User guide:** `docs/source/getting-started.rst` plus three notebooks (`basic_usage`, `fitting_grids`, `feasible_fits`).
- **Notebooks are stored with their outputs.** The docs build doesn't execute them (`nbsphinx_execute = "never"`), so if your change alters behavior or results, re-run the affected notebook and commit its outputs:

  ```bash
  # Once: register a Jupyter kernel named metalog_jax that uses this repo's .venv
  uv run python -m ipykernel install --user --name metalog_jax --display-name metalog_jax

  uv run jupyter nbconvert --to notebook --execute --inplace docs/source/basic_usage.ipynb
  ```

  Re-running `basic_usage` also rewrites `docs/source/basic_usage_{pdf,cdf,sf}.png`. Commit those images as well.

- **Examples:** `examples/` holds marimo apps that mirror the notebooks. Keep them in sync, and run them with `uv run marimo run examples/basic_usage.py`, or `marimo edit` to change one.
- **Building the docs:** `make docs` builds them (pandoc required), and `make docs-live` rebuilds on save.
- **README:** update the usage sections if the public API changes.

### Performance changes

If you change fitting, feasibility, grid search or sampling performance, include before-and-after numbers and your hardware in the pull request. The benchmarks in `benchmarks/feasible_scaling/` document their own usage, for example:

```bash
uv run python benchmarks/feasible_scaling/bench.py MODE N K REPS
```

### Dependencies and licensing

- **New runtime dependencies** need a strong reason, and their licenses must match the allow-list: MIT, BSD, Apache, PSF (Python Software Foundation), ISC, Unlicense, Public Domain, CC0, Zlib, Historical Permission Notice and Disclaimer, Mozilla Public License, Academic Free License or 0BSD. `make license-check` checks runtime dependencies only (the packages `uv export` lists without dependency groups or extras).
- **Development tools** go into the matching dependency group: `uv add --group <group> <package>`.
- **Commit `uv.lock`** with any dependency change. Don't edit it by hand, and resolve lockfile conflicts by re-locking (`uv lock`), not by merging the file.
- **Dependabot** opens update pull requests every Monday. The JAX stack (`jax`, `jaxlib`, `jax-*`, `flax`, `chex`, `optax`, `orbax-*`, `ml-dtypes`) gets its own pull request because it can change numerical results. Tool upgrades sometimes need configuration changes, such as renamed ty rules, and those belong in the same pull request as the upgrade.
- **Third-party code** may be adapted only under a compatible license. Credit it in `NOTICE` with the author, source, license, where it's used and what changed, and keep the original notices.
- **Contributions** are licensed under the project's [MIT License](LICENSE).

## Commit messages

- Write an imperative subject line that makes sense on its own, then explain the why in the body.
- Use the `deps:` and `ci:` prefixes for dependency and workflow changes, matching Dependabot.
- Link issues from the pull request description (`Fixes #123`), so they close on merge.

## Pull requests

1. **Open it against `development`.**
2. **Fill in the template.** A pull request created with `gh pr create --body ...` skips it, so paste it in. Delete sections marked "(if applicable)" that don't apply. Every pull request needs:
   - a summary (problem, approach, and alternatives you rejected)
   - the type of change
   - a test plan with the exact commands you ran and your environment
   - a breaking-change statement and a `release-note` line (or `NONE`)
   - an AI assistance note
3. **Pass CI.** Pull requests into `development` run the Quality Gate and Build workflows:

   | Check | What it runs |
   |---|---|
   | `ruff` | format check and lint |
   | `radon` | maintainability A, complexity B or better |
   | `xenon` | complexity B or better per function, A per module and on average |
   | `interrogate` | docstring coverage at or above the `pyproject.toml` threshold |
   | `pydoclint` | Google-style docstring sections match signatures and bodies |
   | `ty` | type check |
   | `pytest` | full suite with branch coverage of at least 75%, uploaded to Codecov |
   | `license-check` | runtime dependency licenses |
   | `semgrep`, `bandit` | security scans |
   | `sbom` | grype scan of the environment; fails on critical vulnerabilities |
   | `build` | wheel and source distribution |

   `make quality-gate` runs these locally except `radon` (xenon covers its complexity check), `sbom` (`make sbom` needs syft and grype) and `build`. It doesn't check the maintainability grade or branch coverage; use the commands under [Code style](#code-style) and [Tests](#tests) for those. `make ci` adds `make install`, the build and the docs.
4. **Keep compatibility.** A breaking change needs a migration path in the description. Files saved with `dumps()` or `save()` by earlier versions must still `load()`.
5. **Respond to review,** and update the description if the change evolves.

## AI-assisted contributions

AI tools are welcome. Say which ones you used and for what in the pull request. You're responsible for the result: you must be able to explain every line of the diff, and you must have checked the numerical results yourself.

## Releases (maintainers)

A release is a pull request from `development` into `main`. Before merging it, check:

- [ ] The version in `pyproject.toml` is bumped following semantic versioning: patch for fixes, minor for features, major for breaking changes. PyPI rejects a version that already exists.
- [ ] `uv.lock` is refreshed (`uv lock`).
- [ ] `make ci` passes.
- [ ] `development` contains nothing that shouldn't ship.
- [ ] The description lists every included pull request with its release note.

Merging starts this chain on `main`:

1. **Docker** builds the CI image.
2. **Quality Gate** runs.
3. **Build** packages the release.
4. **Publish** (to PyPI, through trusted publishing in the `pypi` environment) and **Documentation** (to GitHub Pages) run.

The Docker workflow runs only when `Dockerfile`, `pyproject.toml` or `uv.lock` changes, so the version bump is also what starts the release.

After the merge, the **Sync development** workflow resets `development` to `main`. It refuses if `development` has commits that aren't in `main`; in that case, merge `main` into `development` instead.

## Citing

If you use metalog-jax in research, see [Citation](README.md#citation) in the README.
