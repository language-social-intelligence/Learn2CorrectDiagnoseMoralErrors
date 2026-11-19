#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
  printf '%s\n' 'This installer supports Linux x86_64. Use the Cloudflare download page for other systems.' >&2
  exit 1
fi
mkdir -p bin
curl --fail --location --retry 3 https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 --output bin/cloudflared.download
chmod 755 bin/cloudflared.download
mv bin/cloudflared.download bin/cloudflared
bin/cloudflared --version
