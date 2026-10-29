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
| `lint` | ruff, ty, interrogate, pydoclint, zizmor |
| `test` | pytest, pytest-cov, absl-py, hypothesis, coverage, radon, xenon, plus scipy and scikit-learn as reference implementations |
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
- **Linting:** ruff with pydocstyle (`D`, Google convention), isort (`I`, `metalog_jax` as first-party), pyflakes (`F`), bugbear (`B`), pyupgrade (`UP`, Python 3.11 idioms) and ruff's own rules (`RUF`). Lint covers `tests/` too, so test classes and methods need docstrings.
- **Module header:** every module in `metalog_jax/` and `tests/` has the line `# Copyright: Travis Jefferies 2026` and a module docstring. Package modules list their `Classes:` and `Functions:` and add a `See Also:` section.
- **Docstrings:** [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) for every public module, class, method and function.
  - **Summary line:** one line of 80 characters or fewer, ending with a period. Functions use the imperative ("Return the …", "Fit a …").
  - **Sections:** add `Args:`, `Returns:` and `Raises:` unless the name and signature make a one-line docstring enough. Argument types go in the signature, not the docstring.
  - **References:** algorithms cite their source in a `References:` section (Keelin 2016, or Baucells, Chrisman, Keelin and Xu 2025).
  - **Examples:** `make doctest` runs every `>>>` example, so write examples that run as shown: include the imports and data, print stable values (for example `float(...)` or `.tolist()` rather than raw JAX arrays, whose repr includes the dtype), and show exceptions as a doctest `Traceback`.
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
- **Static fields:** a field that selects a code path or an array shape is declared with `struct.field(pytree_node=False)`. That covers `boundedness`, `method` and `num_terms` in the metalog parameters, and `size` and `max_draws` in `MetalogRandomVariableParameters`. So `jax.jit` compiles once per combination, and stacked models must share them. Values that can be traced, such as bounds, `alpha`, coefficients and PRNG seeds, stay ordinary leaves.
- **Input validation:** use chex assertions and the helpers in `metalog_jax.utils`, such as `assert_float_array` and `assert_probability_range`. For checks on array values, use `assert_condition`. It raises an exception outside transforms and uses `jax.experimental.checkify` inside them, so never branch in Python on array values. Build `MetalogInputData` with `MetalogInputData.from_values`; direct construction is blocked.
- **Transforms:**
  - Decorate with `@jax.jit`, and pass arguments that determine shapes through `partial(jax.jit, static_argnames=...)`.
  - Don't branch in Python on traced values or call `float()`, `bool()` or `.item()` on them.
  - Keep shapes static so code stays `vmap`-able. Grid search vmaps over datasets, L1 penalties and term counts, and every function in `metalog_jax.feasibility` must work under `jit` and `vmap`.
- **Term ordering:** a model's coefficients are in its `MetalogParameters.term_order`: `TermOrder.METALOG_2` (the default, `DEFAULT_TERM_ORDER`) or `TermOrder.KEELIN_2016`. Code that builds or reads the basis must handle both: build it from `term_basis(num_terms, order)`, the one definition, rather than writing an order out. A default argument for an order is `DEFAULT_TERM_ORDER`, never a member.
- **Public API:** export new public names through the subpackage `__init__.py` and its `__all__`. A new module needs an `automodule` entry in `docs/source/api/modules.rst`.
- **Correctness:** check new numerical results against an independent reference: the papers above and the reference implementation credited in `NOTICE`, a closed form, or scipy.

### Tests

- **Layout:**
  - `tests/unit/` mirrors the package, with `base/` and `regression/` subdirectories.
  - `tests/integration/` covers end-to-end fitting, grid search and Kolmogorov-Smirnov goodness of fit.
  - `tests/feasibility/` covers `metalog_jax.feasibility`.
  - `tests/property/` holds the property-based tests: invariants that every feasible metalog satisfies, checked with Hypothesis.
- **Style:** test classes subclass `parameterized.TestCase` or `absltest.TestCase`. Use `@parameterized.named_parameters` for cases and `numpy.testing` for comparisons.
- **Fixtures:** keep shared fixtures such as parameters and coefficients as class attributes, not module globals.
- **Docstrings:** keep them to one concise line.
- **Issue references:** refer to issue numbers in docstrings only, never in test or variable names, for example `"""Regression tests for issue #9: ..."""`.
- **Coverage of cases:** test all four boundedness types and every fit method the change affects. Explain any tolerance in a comment.
- **Bug fixes:** add a test that fails on `development` and passes with the fix.

```bash
make test-quick                                                    # whole suite, no coverage
uv run pytest tests/unit/test_base.py -k MetalogQuantileAliasTest  # one test class
make test                                                          # branch coverage, fails under 95%
make doctest                                                       # docstring examples
make lowest-deps                                                   # lowest allowed versions, Python 3.11
```

Coverage settings live in `[tool.coverage]` in `pyproject.toml`, so `make test` and CI measure the same thing. CI runs the suite on every Python version the `pyproject.toml` classifiers list (3.11 to 3.14), and again with the lowest versions `pyproject.toml` allows for direct dependencies. To support a new Python version, add its classifier: the test matrix and the release's Docker images both read the list from there.

#### Property-based tests

`tests/property/test_properties.py` states invariants, such as "`ppf` is strictly increasing", "an SPT fit reproduces its three quantiles" or "fitting a metalog's own quantiles recovers its coefficients", and [Hypothesis](https://hypothesis.readthedocs.io/) checks each one on generated inputs. Many come straight from Keelin (2016) and Baucells et al. (2025).

- **Strategies** live in `tests/property/strategies.py`:
  - hand-set 3- and 4-term metalogs
  - metalogs fitted to samples from normal, lognormal, gamma, beta and two-component mixture families
  - coefficient vectors on both sides of the feasibility boundary
  - SPT triplets, probability grids and HDR streams

  They draw term counts from a short list and probabilities as fixed-size grids, because each new shape or combination of static fields makes JAX compile again.
- **Profiles** are registered in `tests/property/conftest.py`. Choose one with `HYPOTHESIS_PROFILE`:

  | Profile | Examples per property | Use |
  |---|---|---|
  | `ci` (default) | 50 | CI and `make test`; derandomized, so every run tests the same examples |
  | `dev` | 20 | quick local runs |
  | `nightly` | 500 | a deeper search before a release |

  Two kinds of property run a fifth of these: those that fit several models per example, and exact identities, which fail on any input if they fail at all.

  ```bash
  HYPOTHESIS_PROFILE=dev uv run pytest tests/property
  ```

- **When a property fails,** Hypothesis shrinks the input to a minimal counterexample and prints it. Pin it with `@example(...)` on the test so it always runs, and if it's a real bug, add a regression test in `tests/unit` or `tests/integration` that fails without the fix. Local runs with the `dev` or `nightly` profile save failing examples in `.hypothesis/`, which is git-ignored.

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

  `make notebooks` checks that all three still run, using a scratch copy and a throwaway kernel, so nothing in `docs/source` or your Jupyter setup changes. CI runs the same check.

- **Examples:** `examples/` holds marimo apps that mirror the notebooks. Keep them in sync, and run them with `uv run marimo run examples/basic_usage.py`, or `marimo edit` to change one.
- **Building the docs:** `make docs` builds them from scratch and fails on any Sphinx warning, as CI does (pandoc required). `make docs-live` rebuilds on save.
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
- **Dependabot** opens update pull requests every Monday. The JAX stack (`jax`, `jaxlib`, `jax-*`, `flax`, `chex`, `optax`, `orbax-*`, `ml-dtypes`) gets its own pull request because it can change numerical results. It waits 7 days after a release before proposing it (security updates aren't delayed). Tool upgrades sometimes need configuration changes, such as renamed ty rules, and those belong in the same pull request as the upgrade.
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
   | `pytest` | full suite on every Python version in the `pyproject.toml` classifiers (3.11 to 3.14); branch coverage of at least 95% is measured on 3.12, which uploads to Codecov |
   | `doctest` | every docstring example runs |
   | `lowest-deps` | full suite on Python 3.11 with the lowest allowed dependency versions |
   | `docs` | Sphinx build with warnings as errors |
   | `notebooks` | the three docs notebooks execute |
   | `license-check` | runtime dependency licenses |
   | `semgrep`, `bandit` | security scans |
   | `sbom` | grype scan of the environment; fails on critical vulnerabilities |
   | `build` | wheel and source distribution |

   `make quality-gate` runs the fast checks locally. It skips `radon` (xenon covers its complexity check; see [Code style](#code-style) for the maintainability grade), `sbom` (`make sbom` needs syft and grype), and the slower jobs. `make ci` adds `make install`, the build, the docs, the notebooks and the lowest-version tests. Only CI runs the Python version matrix.
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

Merging starts the **Release** workflow (`publish.yml`) on `main`. It runs each stage as a job on the merged commit:

1. **Quality Gate** runs every check.
2. **Build** packages the wheel and source distribution, and **Docker** builds the CI image once for each tested Python version and pushes each to GHCR, tagged `py<version>` (for example `:py3.14`). `:latest` is the oldest version's image.
3. **Publish** uploads the build to PyPI (trusted publishing, in the `pypi` environment), and **Documentation** deploys the docs to GitHub Pages.

The release runs only when the merge changes `Dockerfile`, `pyproject.toml` or `uv.lock`, so the version bump is what starts it. To re-run a release, for example after a stage failed, start it by hand on `main`: `gh workflow run publish.yml --ref main`. Publishing a version that's already on PyPI fails, so bump the version first if the earlier run got that far.

After the merge, the **Sync development** workflow resets `development` to `main`. It refuses if `development` has commits that aren't in `main`; in that case, merge `main` into `development` instead.

## Citing

If you use metalog-jax in research, see [Citation](README.md#citation) in the README.
