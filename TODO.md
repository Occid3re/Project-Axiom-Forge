# TODO

## Done

- Split the prod transport into high-rate `entities` packets and throttled `fields` packets to reduce per-client bandwidth pressure.
- Wire kin selection into actual behavior:
  - cooperation bonuses now apply only to kin neighbors
  - attacks now skip kin and target non-kin only
  - process order buffer is reused instead of allocating a new typed array every tick

## Next

- Reduce field bandwidth further with tile deltas and/or downsampled field frames instead of full-grid refreshes.
- Align scoring/UI language with the current mechanics:
  - `communication` currently measures glyph activity, not chemical signals
  - the emergence ladder and related copy still describe it as signaling
- Fix display-world scoring to use the laws of the currently displayed world instead of `bestLaws`, so overlays cannot drift from what viewers are seeing.
- Make `carryingCapacity` actually affect the world or remove it from the evolvable law space.
- Add proper gzip or brotli for static assets in nginx.
- Restore a working lint setup for ESLint 9 by adding a flat `eslint.config.*`.

## Review Backlog

- Revisit social scoring after the kin behavior fix to ensure `socialDifferentiation` is measuring the intended phenomenon.
- Decide whether signal mechanics should be made perceptible to entities again or fully repositioned as visual-only output.
