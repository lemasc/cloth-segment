"""Run management (D7): resolved-config hashing, run naming, locks, registry.

Run folder name: `<YYYYMMDD-HHMM>-<shortcommit>[-dirty]-<slug>`, set at start.
Cache key: `config_hash` (+ data version) in `meta.json`. Task T5.
"""
