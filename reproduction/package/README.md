# Isolated minimum reproduction

Run `docker build --network=none --pull=false -t ggsjob-release-repro .` and then `docker run --rm --network=none --read-only --cap-drop=ALL --security-opt=no-new-privileges ggsjob-release-repro`.

The package reproduces frozen aggregation and acceptance gates. It does not claim to rerun model inference.
