#!/bin/sh
# Re-downloads the kitchen-room assets (~300MB, gitignored) listed in urls.txt, mirroring NVIDIA's
# bucket layout under this directory so USD-relative references resolve.
cd "$(dirname "$0")" || exit 1
P="https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
xargs -P 8 -I{} sh -c 'mkdir -p "$(dirname "{}")" && curl -sf -o "{}" "'$P'{}" || echo "FAIL {}"' < urls.txt
