#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:?Set CONDA_ENV to your Python environment root}"
DEST="${VIDEOX_LOCAL_SITE_PACKAGES:?Set VIDEOX_LOCAL_SITE_PACKAGES to a writable cache directory}"
SOURCE="${CONDA_ENV}/lib/python3.10/site-packages"
READY_MARKER="${DEST}/.cap_local_site_ready"
LOCK_FILE="${DEST}.lock"

if [[ ! -d "$SOURCE" ]]; then
    echo "ERROR: runtime site-packages does not exist: $SOURCE" >&2
    exit 1
fi

source_key="$(stat -c '%Y:%s' "$SOURCE/torch/__init__.py")"

validate_mirror() {
    local root="$1"
    local package
    for package in torch diffusers accelerate safetensors nvidia transformers; do
        if [[ ! -e "$root/$package" ]]; then
            echo "ERROR: local site-packages mirror is incomplete: missing $root/$package" >&2
            return 1
        fi
    done
}

mkdir -p "$(dirname "$DEST")"
exec 9>"$LOCK_FILE"
flock 9

if [[ "${CAP_LOCAL_SITE_FORCE_REFRESH:-0}" != "1" && -f "$READY_MARKER" ]] \
    && [[ "$(<"$READY_MARKER")" == "$source_key" ]]; then
    validate_mirror "$DEST"
    echo "CAP local site-packages already ready: $DEST"
    exit 0
fi

temp_dest="${DEST}.tmp.$$"
rm -rf "$temp_dest"
mkdir -p "$temp_dest"

echo "Staging CAP runtime site-packages: $SOURCE -> $DEST"
if command -v rsync >/dev/null 2>&1; then
    rsync -a --delete "$SOURCE/" "$temp_dest/"
else
    cp -a "$SOURCE/." "$temp_dest/"
fi

validate_mirror "$temp_dest"
printf '%s\n' "$source_key" >"$temp_dest/.cap_local_site_ready"
rm -rf "$DEST"
mv "$temp_dest" "$DEST"
echo "CAP local site-packages ready: $DEST"
