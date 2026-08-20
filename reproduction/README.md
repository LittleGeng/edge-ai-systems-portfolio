# Check the published results offline

This package recalculates the summary values from the included result files and checks the expected conditions. It does not rerun model inference and needs neither credentials nor network access.

```bash
cd package
docker build --network=none --pull=false -t edge-ai-portfolio-repro .
docker run --rm --network=none --read-only --cap-drop=ALL \
  --security-opt=no-new-privileges edge-ai-portfolio-repro
```
