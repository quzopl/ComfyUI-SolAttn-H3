#!/usr/bin/env bash
# Sol-Attn for MiniMax-H3 — one-shot installer.
#
# Drop this into your ComfyUI directory and run it. It fetches the node, fetches
# the Sol-Attn kernel from NVIDIA, installs every dependency into ComfyUI's own
# Python environment, and tells you which kernel backend you ended up with.
#
#   cd /path/to/ComfyUI
#   curl -sSLO https://raw.githubusercontent.com/quzopl/ComfyUI-SolAttn-H3/master/install.sh
#   bash install.sh
#
# Options:
#   --comfyui PATH     ComfyUI root (default: autodetected)
#   --no-cute          skip the CuTe DSL runtime, use the Triton kernel
#   --skip-selftest    do not run the benchmark at the end
#   --ref REF          node branch/tag to install (default: master)
set -euo pipefail

NODE_REPO="https://github.com/quzopl/ComfyUI-SolAttn-H3.git"
NODE_NAME="ComfyUI-SolAttn-H3"
SANA_REPO="https://github.com/NVlabs/Sana.git"
SANA_BRANCH="sol-engine"

COMFYUI=""
NODE_REF="master"
WITH_CUTE=1
RUN_SELFTEST=1

die()  { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }
say()  { printf '\033[36m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[33mwarning:\033[0m %s\n' "$*" >&2; }

while [ $# -gt 0 ]; do
  case "$1" in
    --comfyui) COMFYUI="${2:?--comfyui needs a path}"; shift 2 ;;
    --ref) NODE_REF="${2:?--ref needs a value}"; shift 2 ;;
    --no-cute) WITH_CUTE=0; shift ;;
    --skip-selftest) RUN_SELFTEST=0; shift ;;
    -h|--help) sed -n '2,19p' "$0" | sed 's/^# \?//'; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac
done

for tool in git curl; do
  command -v "$tool" >/dev/null 2>&1 || die "$tool is required but not installed"
done

# --- 1. locate ComfyUI -----------------------------------------------------
is_comfyui() { [ -f "$1/main.py" ] && [ -d "$1/comfy" ]; }

if [ -z "$COMFYUI" ]; then
  here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
  for guess in "$PWD" "$here" "$here/../.." "$HOME/ComfyUI"; do
    cand="$(cd "$guess" 2>/dev/null && pwd -P || true)"
    [ -n "$cand" ] && is_comfyui "$cand" && { COMFYUI="$cand"; break; }
  done
fi
[ -n "$COMFYUI" ] || die "could not find ComfyUI. Run this from the ComfyUI directory, or pass --comfyui /path/to/ComfyUI"
is_comfyui "$COMFYUI" || die "$COMFYUI does not look like ComfyUI (needs main.py and comfy/)"
COMFYUI="$(cd "$COMFYUI" && pwd -P)"
say "ComfyUI: $COMFYUI"

# --- 2. locate its Python --------------------------------------------------
PY=""
for p in "$COMFYUI/venv/bin/python" "$COMFYUI/.venv/bin/python" \
         "$COMFYUI/venv/Scripts/python.exe" "$COMFYUI/.venv/Scripts/python.exe"; do
  [ -x "$p" ] && { PY="$p"; break; }
done
if [ -z "$PY" ]; then
  PY="$(command -v python3 || command -v python || true)"
  [ -n "$PY" ] && warn "no virtualenv under $COMFYUI; falling back to $PY — make sure it is the interpreter ComfyUI runs with"
fi
[ -n "$PY" ] || die "no Python interpreter found"
say "Python:  $PY"

if command -v uv >/dev/null 2>&1; then
  PIP=(uv pip install --python "$PY")
else
  PIP=("$PY" -m pip install)
fi

# --- 3. check the environment before changing anything ---------------------
say "checking requirements"
"$PY" -W ignore - <<'PYEOF' || die "requirements not met; nothing was installed"
import sys
try:
    import torch
except ImportError:
    sys.exit("    torch is not installed in this environment")

def ver(text):
    out = []
    for part in text.split("."):
        digits = "".join(c for c in part if c.isdigit())
        if not digits:
            break
        out.append(int(digits))
    return tuple(out[:2])

problems = []
print(f"    torch {torch.__version__}, CUDA {torch.version.cuda or 'none'}")
if ver(torch.__version__) < (2, 10):
    problems.append(f"torch {torch.__version__} < 2.10")
if not torch.cuda.is_available():
    problems.append("CUDA is not available to torch")
else:
    cap = torch.cuda.get_device_capability(0)
    print(f"    GPU {torch.cuda.get_device_name(0)}, SM{cap[0]}{cap[1]}")
    if cap[0] < 8:
        problems.append(f"SM{cap[0]}{cap[1]} < 8.0 — Sol-Attn cannot run on this GPU")
if ver(torch.version.cuda or "0") < (12, 8):
    problems.append(f"CUDA {torch.version.cuda} < 12.8")
try:
    import triton
    print(f"    triton {triton.__version__}")
    if ver(triton.__version__) < (3, 6):
        problems.append(f"triton {triton.__version__} < 3.6")
except ImportError:
    problems.append("triton is not installed")

for p in problems:
    print(f"    UNMET: {p}")
sys.exit(1 if problems else 0)
PYEOF

# --- 4. fetch the node -----------------------------------------------------
NODE_DIR="$COMFYUI/custom_nodes/$NODE_NAME"
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
mkdir -p "$COMFYUI/custom_nodes"

if [ "$SELF" = "$NODE_DIR" ] || [ -e "$NODE_DIR/kernel.py" ] && [ ! -d "$NODE_DIR/.git" ]; then
  say "node already present at $NODE_DIR (leaving it alone)"
elif [ -d "$NODE_DIR/.git" ]; then
  say "updating $NODE_NAME"
  git -C "$NODE_DIR" fetch -q origin "$NODE_REF"
  git -C "$NODE_DIR" checkout -q FETCH_HEAD
else
  say "cloning $NODE_NAME into custom_nodes"
  git clone -q --branch "$NODE_REF" --depth 1 "$NODE_REPO" "$NODE_DIR"
fi

# --- 5. fetch the kernel from NVIDIA ---------------------------------------
SANA="$NODE_DIR/vendor/sana-sol-engine"
if [ -d "$SANA/.git" ]; then
  say "updating the NVIDIA kernel source"
  git -C "$SANA" fetch -q --depth 50 origin "$SANA_BRANCH"
  git -C "$SANA" checkout -q FETCH_HEAD
else
  say "cloning NVlabs/Sana ($SANA_BRANCH) — this is the kernel, ~60 MB"
  mkdir -p "$(dirname "$SANA")"
  git clone -q --depth 1 --branch "$SANA_BRANCH" "$SANA_REPO" "$SANA"
fi
say "kernel source: $(git -C "$SANA" log -1 --format='%h %ad' --date=short)"

BACKENDS="$SANA/techniques/sparse_backends"
[ -f "$BACKENDS/pyproject.toml" ] || die "$BACKENDS is not the sparse_backends package — upstream layout changed"

# --- 6. install ------------------------------------------------------------
say "installing sol-attn"
"${PIP[@]}" -e "$BACKENDS"

if [ "$WITH_CUTE" -eq 1 ]; then
  # Without all three, every architecture falls back to the Triton reference
  # kernel. apache-tvm-ffi is needed too, which upstream does not document.
  say "installing the CuTe DSL runtime"
  "${PIP[@]}" "nvidia-cutlass-dsl>=4.5" cuda-python apache-tvm-ffi
else
  warn "--no-cute: you will get the Triton backend, which is slower on every architecture that has a CuTe kernel"
fi

# --- 7. report what you actually got ---------------------------------------
say "resolving backend"
"$PY" -W ignore - "$NODE_DIR" <<'PYEOF' || die "the kernel did not load; see the message above"
import pathlib, sys, types
root = pathlib.Path(sys.argv[1])
pkg = types.ModuleType("solattn_h3"); pkg.__path__ = [str(root)]
sys.modules["solattn_h3"] = pkg
from solattn_h3.kernel import probe
found = probe()
print(f"    {found.describe()}")
if not found.available:
    sys.exit(1)
if found.backend == "triton" and (found.arch or (0, 0)) >= (8, 9):
    print("    NOTE: this GPU has a CuTe kernel, but the runtime is missing.")
    print("          Install nvidia-cutlass-dsl, cuda-python and apache-tvm-ffi.")
PYEOF

if [ "$RUN_SELFTEST" -eq 1 ]; then
  say "running the selftest (first call per shape compiles kernels)"
  "$PY" -W ignore "$NODE_DIR/selftest.py" --tokens 8192 16384 || warn "selftest reported a problem"
fi

say "done — restart ComfyUI and look for 'Sol-Attn (MiniMax-H3)' under model_patches/attention"
