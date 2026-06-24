# Changelog

## v0.1.0 @yyb - 2026-06-24

### Added
- Released the clean public pe2sink demo artifact without private development history.
- Added the FLUX.1-dev comma-padding prompt demo with dry-run and GPU modes.
- Added the compact public paper-stats pack for no-GPU figure reproduction.
- Added smoke tests covering prompt construction, dry-run metadata, and public figure generation.
- Added README links to the project page, code, reproduction notes, prompt demo, and stats pack.
- Added a composite README teaser showing architecture-dependent sink identity and FLUX token-type attention redistribution.

### Changed
- Structured the README in a paper-project style with News, Installation, Usage, TODO, Citation, and License sections.
- Set the GitHub repository homepage to the project page.

### Verification
- `python3 scripts/smoke_public.py --skip-figures`
- `scripts/smoke_public.py` in a Linux GPU development environment
- `tools/reproduce_public_stats.py` for all public stats figures
- Secret, absolute-path, private-remote, and large-file scans before the initial public push
