Configuration for this project lives in `backend/app/config.py` (a single
dataclass with environment-variable overrides) rather than external YAML/
JSON files here, since the number of tunables is small enough that a
typed Python module is simpler to read, get autocomplete on, and keep in
sync with the code that consumes it.

If the project grows to multiple environments (dev/staging/prod) with
meaningfully different configs, this directory is the natural place to
add e.g. `configs/dev.yaml`, `configs/prod.yaml`, loaded by
`app/config.py` based on an `ENV` variable.
