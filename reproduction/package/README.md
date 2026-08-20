# Self-contained result check

Run `docker build --network=none --pull=false -t ggsjob-release-repro .`, followed by `docker run --rm --network=none --read-only --cap-drop=ALL --security-opt=no-new-privileges ggsjob-release-repro`.

The package recalculates the published summary from the included result files. It does not rerun model inference.
