# TODO

## Done

- Split the prod transport into high-rate `entities` packets and throttled `fields` packets to reduce per-client bandwidth pressure.
- Wire kin selection into actual behavior:
  - cooperation bonuses now apply only to kin neighbors
  - attacks now skip kin and target non-kin only
  - process order buffer is reused instead of allocating a new typed array every tick
- Fix display-world overlay drift:
  - display scores now use the displayed world's laws
  - the client reads and shows the displayed world's laws instead of the all-time best
- Activate `carryingCapacity` in the population-pressure curve and expose it in the laws UI.
- Update the emergence/UI copy so the communication metric is described as glyph-based rather than signal-based.
- Reduce field bandwidth further by downsampling field frames on the wire and expanding them client-side before render.
- Add gzip for static assets in nginx.
- Restore a working ESLint 9 setup with a flat config and clean `npm run lint`.
- Reduce field bandwidth further with tile-delta field packets plus periodic keyframes.

## Next

- Revisit score and stage thresholds after the transport, kin-selection, and carrying-capacity changes to ensure the UI still reflects real emergence progress.
- If brotli is still desired later, install an nginx build/module that includes brotli support on the VPS first.

## Review Backlog

- Revisit social scoring after the kin behavior fix to ensure `socialDifferentiation` is measuring the intended phenomenon.
- Decide whether signal mechanics should be made perceptible to entities again or fully repositioned as visual-only output.
