# Automated compatibility releases

The [Compatibility and releases workflow](../.github/workflows/ci.yml) monitors upstream releases of Inspect AI and OpenAI Codex, verifies compatibility against the test suite, and automatically publishes updated packages to PyPI and GitHub Releases.

In addition to running on every push and pull request, upstream dependency checks run automatically every hour (at minute 17 UTC). You can also trigger them manually from the Actions tab or via an `upstream-release` repository dispatch event.

## Candidate selection and validation

When checking for updates, the workflow queries PyPI for the latest stable (non-yanked) Inspect release and npm for the latest stable Codex release. For Codex, the workflow also verifies that the release corresponds to a published stable GitHub release with a valid commit SHA. Version pins are never downgraded, and if neither dependency has changed, the workflow exits without creating a release.

When a newer stable version is found, the workflow constructs a candidate release:

1. Updates the dependency pins and regenerates the lockfile.
2. Re-generates the native JSON-RPC protocol bindings from the updated schemas.
3. Runs the full test suite, including protocol validation, linting (`ruff`), type checking (`ty`), the pytest suite, Docker sandbox acceptance tests, and distribution builds.

All test runs use local mock model responses to avoid paid API inference. If any verification check fails, publication halts immediately, and the workflow attaches the candidate diff and diagnostic logs to the run for review.

## Publication flow

Package versions use the UTC date: `YYYY.M.D`, such as `2026.10.29`. Additional releases on the same date append a counter: `2026.10.29.1`, `2026.10.29.2`. The counter resets on a new date. Tags prefix the version with `v`.

Every GitHub release description names the exact tested Inspect AI and Codex CLI versions and links to their upstream releases, including unchanged dependencies.

The candidate version is assigned before validation and building. Once all checks pass, the workflow executes the following release sequence:

1. **Commit and tag**: Commits the updated lockfile and generated bindings, then tags the commit using an atomic push. If the default branch has received concurrent commits in the meantime, the push is rejected without force-pushing.
2. **Draft GitHub Release**: Packages the built wheel and source distribution (`sdist`) into a draft release on GitHub.
3. **Publish to PyPI**: Uploads the packages to PyPI using GitHub Actions OIDC Trusted Publishing.
4. **Publish GitHub Release**: Converts the draft GitHub release to public once PyPI accepts the upload.

## Repository setup

To enable automated releases in your fork or repository:

- **GitHub Actions permissions**: Ensure GitHub Actions is enabled and that branch protection rules on your default branch permit the Actions bot to push release commits and create tags.
- **PyPI Trusted Publishing**: Set up a [PyPI Trusted Publisher](https://docs.pypi.org/trusted-publishers/using-a-publisher/) for `inspect-codex-app-server` configured with your repository name, workflow filename `ci.yml`, and environment name `pypi`. Because the workflow authenticates via OIDC, no hardcoded API tokens or secrets are required. For unattended releases, configure the GitHub `pypi` environment to allow deployments without manual approval.

## Retries and local inspection

If publication fails after a tag has already been pushed (for example, due to a transient PyPI network error), open the workflow run in GitHub Actions and click **Re-run failed jobs**. The workflow will reuse the previously built artifacts and existing tag, skipping any files that have already been uploaded.

Successful release artifacts are retained for 14 days, while failure diagnostics and diffs are retained for seven days.

You can preview available upstream updates locally at any time without modifying tracked files:

```bash
uv run python scripts/update_dependencies.py --dry-run
```
