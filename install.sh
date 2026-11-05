#!/usr/bin/env bash
# Sets up tapzones for this user, from wherever this folder lives: the repo,
# or the Omarchy plugin folder (~/.config/omarchy/plugins/meshwa.tapzones).
#
#   ./install.sh              the service, its Python env, the `tapzones` command
#   ./install.sh --full       also torch + matplotlib (overlay ensemble, research)
#   ./install.sh --no-start   install, but don't start anything
#   ./install.sh --uninstall  stop and remove the units and the command
#                             (keeps your recordings, models and config)
#
# The Python env lives outside this folder, in ~/.local/share/tapzones/venv:
# Omarchy plugin folders may not contain symlinks, and a plugin update then
# never touches it. On Omarchy the plugin draws the feedback in the shell; on
# other desktops a small GTK client (tapzones-flash.service) does.
set -euo pipefail

APP=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DATA=${XDG_DATA_HOME:-$HOME/.local/share}/tapzones
VENV=$DATA/venv
UNITS=${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user
BIN=$HOME/.local/bin
OMARCHY=$([[ -d $HOME/.config/omarchy ]] && command -v omarchy >/dev/null && echo yes || echo no)

full=no start=yes
for arg in "$@"; do
  case $arg in
    --full) full=yes ;;
    --no-start) start=no ;;
    --uninstall)
      systemctl --user disable --now tapzones.service tapzones-flash.service 2>/dev/null || true
      rm -f "$UNITS/tapzones.service" "$UNITS/tapzones-flash.service" "$BIN/tapzones"
      systemctl --user daemon-reload
      echo "Removed the tapzones units and command. Kept $DATA and ~/.config/tapzones."
      exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

say() { printf '\033[1m%s\033[0m\n' "$*"; }

# ------------------------------------------------------------ requirements
need=()
command -v parec >/dev/null || need+=("parec (PulseAudio tools; on Arch: libpulse)")
python3 -c 'import sys; sys.exit(sys.version_info < (3, 11))' 2>/dev/null || need+=("python 3.11 or newer")
if [[ $OMARCHY == no ]]; then
  python3 -c 'import gi; gi.require_version("Gtk", "4.0")' 2>/dev/null || need+=("PyGObject with GTK 4 (on Arch: python-gobject gtk4)")
  [[ -e /usr/lib/libgtk4-layer-shell.so ]] || need+=("gtk4-layer-shell")
fi
if ((${#need[@]})); then
  echo "tapzones needs:" >&2
  printf '  - %s\n' "${need[@]}" >&2
  exit 1
fi

# --------------------------------------------------------------- python env
say "Python env: $VENV"
mkdir -p "$DATA"
reqs=$APP/requirements.txt
[[ $full == yes ]] && reqs=$APP/requirements-research.txt
# System site packages: the GTK overlay and flash use the system's PyGObject.
if command -v uv >/dev/null; then
  [[ -x $VENV/bin/python ]] || uv venv --system-site-packages --python python3 "$VENV"
  uv pip install --python "$VENV/bin/python" -r "$reqs"
else
  [[ -x $VENV/bin/python ]] || python3 -m venv --system-site-packages "$VENV"
  "$VENV/bin/python" -m pip install --upgrade -r "$reqs"
fi

# ------------------------------------------------------------- the command
mkdir -p "$BIN"
cat > "$BIN/tapzones" <<EOF
#!/usr/bin/env bash
# tapzones: see $APP/docs/usage.md
exec "$VENV/bin/python" "$APP/taps.py" "\$@"
EOF
chmod +x "$BIN/tapzones"

# ------------------------------------------------------------ systemd units
mkdir -p "$UNITS"
unit() {  # template in this folder → $UNITS, with this install's paths
  sed -e "s|@APP@|$APP|g" -e "s|@PYTHON@|$VENV/bin/python|g" "$APP/$1" > "$UNITS/$1"
}
unit tapzones.service
units=(tapzones.service)
if [[ $OMARCHY == no ]]; then
  unit tapzones-flash.service
  units+=(tapzones-flash.service)
fi
systemctl --user daemon-reload

if [[ $start == yes ]]; then
  systemctl --user enable --now "${units[@]}"
  systemctl --user restart "${units[@]}"     # picks up a new version on reinstall
fi

say "Installed."
echo "  service:  systemctl --user status tapzones     log: journalctl --user -u tapzones -f"
echo "  command:  tapzones status | pause | resume | layout | collect | noise | test"
if [[ $OMARCHY == yes ]]; then
  echo "  Omarchy:  the bar's tap icon opens the layout editor; right click pauses."
  omarchy plugin list 2>/dev/null | grep -q '^meshwa.tapzones .*enabled' ||
    echo "            enable the plugin: omarchy plugin enable meshwa.tapzones"
fi
ls "$DATA"/collect-*.npz >/dev/null 2>&1 ||
  echo "  first:    the mic has to learn this laptop. Record taps: tapzones collect   (docs/training.md)"
