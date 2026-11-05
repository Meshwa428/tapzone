# tapzones

Tap the screen of a laptop that has no touchscreen, and something happens. The
built-in microphone hears the tap, a small model works out which part of the screen
you tapped, and tapzones runs the command you gave that part: open the browser, take a
screenshot, pause the music.

It works from sound alone. The lid moves differently depending on where it's hit, and
that difference is enough to tell top from bottom about 95% of the time, and the four
quadrants apart about 85% of the time. The research behind it is in
[docs/research.md](docs/research.md).

- **Your own layout.** Split the screen into 1 to 5 sections (or more) per gesture,
  leave dead zones between them, and give each a shell command. The editor shows how
  reliably the mic can tell your sections apart before you save.
- **Ignores what isn't a tap.** A gate turns away typing, knocks on the table and other
  noise, and taps it isn't sure of are refused instead of guessed.
- **Fast:** it answers 45 ms after the tap.
- **Omarchy native.** On [Omarchy](https://omarchy.org/) it's a shell plugin: a bar
  icon, feedback drawn in your theme, and the layout editor inside the shell. It also
  runs on any other Linux desktop with PipeWire or PulseAudio, using a small GTK front
  end.

> [!IMPORTANT]
> The models learn **one laptop**: its lid, its hinge, its mic. On a new machine you
> record a few hundred taps first (about 15 minutes), then train. See
> [docs/training.md](docs/training.md). Single taps work today. Double and triple taps
> and 2- and 3-finger taps are designed but not built yet
> ([roadmap](docs/Gestures/Roadmap.md)).

## Install on Omarchy

```bash
omarchy plugin add https://github.com/Meshwa428/tapzone --enable
~/.config/omarchy/plugins/meshwa.tapzones/install.sh
```

The first command installs the shell plugin. The second sets up the listening service:
1. a Python env in `~/.local/share/tapzones/venv`;
2. the `tapzones` command in `~/.local/bin`;
3. the systemd user unit `tapzones.service`, which it also starts.

You can skip the second command and click the tap icon in the bar instead: while the
service isn't running, the click runs the installer in a terminal.

Then:

| Do | How |
|---|---|
| Record taps so it learns your laptop | `tapzones collect`, then `tapzones noise` ([training](docs/training.md)) |
| Choose sections and commands | Click the bar icon: the layout editor. **S** saves and trains |
| Pause / resume | Right-click the bar icon, or `tapzones toggle` |
| Open the editor from a keybinding | `omarchy-shell shell summon meshwa.tapzones '{}'` |
| See what it hears | `journalctl --user -u tapzones -f` |

**Update:**

```bash
omarchy plugin update meshwa.tapzones
~/.config/omarchy/plugins/meshwa.tapzones/install.sh   # restarts the service on the new code
omarchy restart shell
```

**Uninstall:**

```bash
~/.config/omarchy/plugins/meshwa.tapzones/install.sh --uninstall
omarchy plugin remove meshwa.tapzones
```

Your recordings, models and config stay in `~/.local/share/tapzones` and
`~/.config/tapzones`. Delete those too if you want everything gone.

## Install on other Linux desktops

You need Python 3.11+, `parec` (PulseAudio tools, which also work on PipeWire), GTK 4,
PyGObject and gtk4-layer-shell (a Wayland compositor with layer-shell support, such as
Hyprland or Sway). On Arch:

```bash
sudo pacman -S --needed python python-gobject gtk4 gtk4-layer-shell libpulse
git clone https://github.com/Meshwa428/tapzone ~/.local/share/tapzones/app
~/.local/share/tapzones/app/install.sh
```

This installs and starts both `tapzones.service` and `tapzones-flash.service` (the GTK
flash feedback). The layout editor is `tapzones layout`.

`install.sh --full` also installs torch and matplotlib, for the calibration overlay's
full ensemble and the research scripts.

## Documentation

The docs are an [Obsidian](https://obsidian.md/) vault in [docs/](docs/README.md); they
also read fine on GitHub.

| | |
|---|---|
| [Omarchy plugin](docs/Omarchy%20plugin.md) | How the service, the socket and the front ends fit together; developing the plugin |
| [usage](docs/usage.md) | Every command, key and file |
| [training](docs/training.md) | Recording taps and noise, training and checking a model |
| [Gesture system](docs/Gestures/Gesture%20system.md) | Sections, dead zones, the planned gestures |
| [research](docs/research.md) | What the mic can and can't hear, and why |

## Repository layout

```
tapzones/            the Python package: `tapzones` is tapzones/__main__.py
  audio.py           the mic and the tap detector
  features.py        what the models hear in a tap
  layout.py          sections, presets, the editor's live estimate
  config.py          ~/.config/tapzones/gestures.toml
  models.py          the gate and section models the service runs
  train.py           tapzones train …
  service/           the headless service and its socket client
  gtk/               the GTK overlay, layout editor and flash (other desktops)
omarchy/             the Omarchy shell plugin (manifest.json is at the root)
systemd/             unit templates, filled in by install.sh
research/            experiments, plots, and their logged results
tests/               python -m unittest discover -s tests
docs/                the Obsidian vault
```

For development: `uv venv --system-site-packages .venv && uv pip install --python
.venv/bin/python -e '.[full]'`, then `.venv/bin/tapzones …`
([usage](docs/usage.md#Environment)).

## Publishing changes

```bash
omarchy plugin validate <a clean clone>   # a working tree with .venv fails: no symlinks allowed
git push                                  # users get it with: omarchy plugin update meshwa.tapzones
```

For a release, bump `version` in `manifest.json`, then tag it and create the release:

```bash
git tag v0.2.0 && git push --tags
gh release create v0.2.0 --generate-notes
```

## License

MIT. See [LICENSE](LICENSE). Copyright (c) 2026 Meshwa.

The sounds in `tapzones/sounds/` aren't covered by the MIT licence. They're
from SND01 "sine" by Yasuhiro Tsuchiya, part of [SND](https://snd.dev), and
are used under SND's terms: see [their credits](tapzones/sounds/CREDITS.md).
