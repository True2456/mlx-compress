#!/bin/zsh
# Re-add SpecPrefill drafter-on-ANE to oMLX 0.6.2.
#
# 0.6.2 dropped `specprefill_ane_enabled`: enable_qwen35_ane_prefill is still
# called for the TARGET model (engine/batched.py) but no longer for the
# SpecPrefill drafter, so token scoring always runs on the GPU. This restores
# the setting, wires the drafter compile, and exposes a toggle in the model
# settings modal (which is both the web dashboard and the desktop app, since
# the app is a WebView over the same templates).
#
# Requires the terminal to have App Management permission:
#   System Settings > Privacy & Security > App Management
# Bundle edits are wiped by oMLX updates -- re-run after each update.
set -e
O=/Applications/oMLX.app/Contents/Resources
P="$(cd "$(dirname "$0")" && pwd)/specprefill_ane_0.6.2.patch"

echo "== dry run =="
patch -p1 -d "$O" --dry-run < "$P"

echo "== applying =="
patch -p1 -d "$O" < "$P"

echo "== i18n keys =="
python3 - "$O" <<'PY'
import json, pathlib, sys
i18n = pathlib.Path(sys.argv[1]) / "omlx/admin/i18n"
label = "Score SpecPrefill on the Neural Engine"
hint = ("Run the draft model's token scoring on the ANE instead of the GPU, freeing "
        "the GPU for target prefill. Needs a 4/5-bit affine gs64/128 drafter - bf16 "
        "drafts are ineligible and stay on GPU.")
for f in sorted(i18n.glob("*.json")):
    d = json.loads(f.read_text())
    d["modal.model_settings.specprefill_ane"] = label
    d["modal.model_settings.specprefill_ane_hint"] = hint
    f.write_text(json.dumps(d, ensure_ascii=False, indent=2) + "\n")
    print("  +", f.name)
PY
echo "== done. restart oMLX. =="
