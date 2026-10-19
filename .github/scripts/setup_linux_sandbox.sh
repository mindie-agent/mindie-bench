#!/usr/bin/env bash
set -euo pipefail

# GitHub's disposable Ubuntu runners restrict unprivileged user namespaces.
# Permit only bwrap to create them; keep the benchmark's network/filesystem
# namespaces intact instead of weakening its sandbox command.
sudo apt-get update -qq
sudo apt-get install -y bubblewrap apparmor
sudo tee /etc/apparmor.d/bwrap >/dev/null <<'PROFILE'
abi <abi/4.0>,
include <tunables/global>

profile bwrap /usr/bin/bwrap flags=(unconfined) {
  userns,
}
PROFILE
sudo apparmor_parser -r /etc/apparmor.d/bwrap
bwrap --unshare-all --ro-bind / / --proc /proc --dev /dev -- /usr/bin/true
