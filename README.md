# Elite Hunter v1
Elite Hunter v1 -> 4.23

Elite Hunter v2 - > 3.65

Dual-mode Mech Royale bot (pure Python, standard library only).

- `elite_hunter_v1.py`: the bot. Squad mode (practice: the others hunt us) and free-for-all mode (the others fight each other); it switches when mechs die before the zone shrinks without our kill.
- `mapmodel.py` + `mapmodel.bin`: the campus map model (buildings, trees, walkable grid, openness regions). `mapmodel.bin` is precomputed for the official `v2` map; the bot loads the map itself from `/arena` at match time.
- `mechfight.py`: the kit's protocol wrapper (unchanged).
- `Dockerfile`: builds the image; entry point `elite_hunter_v1.py`.

Check: `python3 check/check.py --map-dir arena -- python3 -u elite_hunter_v1.py` (from the kit).
