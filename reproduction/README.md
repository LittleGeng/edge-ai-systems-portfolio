# Minimum offline reproduction

The package replays frozen result aggregation and acceptance gates. It performs no model inference and requires no credentials or network access.

```bash
cd package
docker build --network=none --pull=false -t edge-ai-portfolio-repro .
docker run --rm --network=none --read-only --cap-drop=ALL \
  --security-opt=no-new-privileges edge-ai-portfolio-repro
```
