Vendored from NVIDIA's Isaac Sim 5.1 asset library (Nucleus/Omniverse content server):

- `table_instanceable.usd` ← `Isaac/Props/Mounts/ThorlabsTable/table_instanceable.usd`
- `Props/instaceable_meshes.usd` ← `Isaac/Props/Mounts/ThorlabsTable/Props/instaceable_meshes.usd`
  (filename typo is NVIDIA's own; preserved verbatim since the first file references it by that
  literal relative path)

Downloaded from `https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/...`.
Chosen over the default `SeattleLabTable` asset for its smaller, better-centered footprint
(~0.72m x 0.76m, flat top at local Z=0, identity xformOps) — see `compose_isaac_scene.py`'s
`--table-asset-path`.
