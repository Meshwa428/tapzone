# TODO

What tapzones still needs before someone other than its developer can use it.
Most of it follows from one fact: **every device trains its own model from
scratch.** The 868 recorded taps and the trained models fit one laptop's mic
and lid, and are worth nothing on another machine.

Priority: **P0** blocks a first user · **P1** needed for a good first release ·
**P2** later.

---

## 1. First run and calibration (per device)

- [x] **P0 — No model on a fresh install.** Built 2026-11-06: with no recorded
      taps the service says "not calibrated yet" (and notifies once), and
      the bar panel shows only "Start calibration".
- [x] **P0 — Calibration in the GUI.** Built 2026-11-06: `tapzones setup`, a
      short collect session (104 taps, ~2 min), opened by the service
      (socket `calibrate`) from the panel. When the overlay closes with new
      sessions, the service trains the single-tap model, then the gate.
      A lone sitting is scored by its halves (`train.sittings`) and shown as
      an estimate.
- [x] **P0 — Record the device's noise too.** Built 2026-11-06: setup's second
      step, 45 s of everyday sounds → `noise-setup-*.npz`.
- [x] **P0 — "That wasn't a tap", in everyday use.** Built 2026-11-06: the
      service keeps the sound of the most recent tap that ran a command
      (one 175 ms window, nothing older); the bar panel's "That wasn't a
      tap", `tapzones mistake` (bind it to a key) and the socket command
      `mistake` save it to `noise-marked.npz` and clear it. "Record what
      isn't a tap" in the panel keeps every sound the detector catches
      until Stop (`noise-rec-*.npz`). The gate retrains by itself 20 s
      after new noise, and the service now reloads `gate.joblib` when it
      changes (it used to need a restart).
- [ ] **P2 — Undo the command** a mistaken tap ran, where that's possible.
- [ ] **P1 — Review what was marked.** A list of saved false alarms (with
      time, and play/delete), because a real tap marked by mistake teaches
      the gate to reject real taps. Show how many noises the gate has
      learned from, and the measured "real taps let through" after each
      retrain, so a bad batch is visible.
- [ ] **P1 — "Right tap, wrong section".** A different correction: the tap
      was real but landed in the wrong section. Saving it with the section
      the user names adds a labelled tap for the section model, not noise.
- [ ] **P0 — How many taps does a new device need?** Unknown: all the research
      used many sittings on one laptop. Run a from-scratch learning curve on
      the existing data (accuracy against taps per area, against number of
      sittings) and set a minimum and a recommended amount. Decide how long
      people are asked to tap (open question, see the end).
      *First result (`research/experiments.py minimum`): quadrants level off
      near 77% at ~190 taps from 3+ sittings; top/bottom reaches 90% at
      32–64 taps. Still to do: the shipped ensemble, and seconds per tap.*
- [ ] **P0 — Several sittings.** The accuracy we report is from sittings the
      model never saw. With one sitting there's no honest number. Either ask
      for 2+ short sittings, or report a within-sitting number marked as
      optimistic, and ask for another sitting later.
- [ ] **P1 — Areas with too few taps.** A layout section with fewer than
      `MIN_TAPS` recorded taps can't be trained. The editor should say which
      area is short and offer to record taps just there.
- [ ] **P1 — "Improve" instead of "fine-tune".** The models (SVM, MLP, logistic
      regression, CNN) can't be nudged with a few new taps. Improving
      means retraining from scratch on all of the device's taps, with the new
      ones added. Word the UI that way.
- [ ] **P1 — Recalibration.** Lid angle, a different table or a laptop on a lap
      change the sound. Detect a drop in confidence over time and suggest
      recording a few more taps.
- [ ] **P1 — Reset.** Delete this device's recordings and models and start over.

## 2. Training: in the background, visible

- [ ] **P0 — Progress reporting.** Training runs as a subprocess with no
      progress. Report stages (gate, sections, cutoff, the CNN), a fraction
      done and a time estimate as JSON lines. The service forwards them as
      socket events.
- [ ] **P0 — Show progress.** In the settings menu, and on hover over the bar
      icon (tooltip: "training… 40%, about 3 min left"). The bar icon shows a
      training state.
- [ ] **P1 — Cancel, and survive restarts.** A cancel button. A service restart
      during training either resumes or reports "training was interrupted".
- [ ] **P1 — Be a good neighbour.** Train at low priority (`nice`, `ionice`).
      Maybe wait until the laptop is on power.
- [ ] **P1 — One queue.** Calibration, noise, layout saves and 2/3-finger
      training all start training runs. Queue them (or merge them), and never
      run two at once. (The service already stops a run a newer save made
      pointless.)
- [ ] **P1 — Model versions.** Store the feature-pipeline version and the
      scikit-learn/torch versions with each model. Pickles break across
      scikit-learn versions: retrain automatically, from the stored taps,
      after an upgrade.
- [ ] **P2 — Without torch.** The CNN needs the `full` extra. Training and the
      service must work, a little less accurately, without it.

## 3. Gestures

- [ ] **P0 — Double and triple taps need no training.** They reuse the
      single-tap model: group taps that come within a gap window and land in
      the same section, and combine their probabilities (a double tap is
      surer than a single one). Detector changes:
  - [ ] the 300 ms lockout after a tap (`count - last_tap > 60` in
        `audio.py`) loses fast second taps;
  - [ ] a single tap must wait for the gap window to close, but only when a
        double/triple gesture is configured;
  - [ ] tap 2 can ring inside tap 1's 150 ms window.
- [ ] **P0 — Don't make single/double/triple separate parts.** The editor treats
      them as separate gestures with separate layouts and training. They are
      one model plus a count. Decide whether a double tap gets its own layout
      or shares the single-tap one (open question), and restructure the
      editor and the config to match.
- [ ] **P1 — 2- and 3-finger taps.** These do need their own recordings and
      training: a calibration step per kind, a model that tells 1, 2 and 3
      fingers apart, and a noise filter that doesn't turn them away (it has
      never heard them). Whether the mic can tell them apart at all is
      unknown: research first (docs/Gestures/Research plan.md #5).
- [ ] **P1 — The editor shows what's real.** Gestures the service can't hear yet
      are greyed out. Say clearly what they need ("record 2-finger taps
      first").

## 4. Settings menu (new)

Today every value below is a constant in the code. Each becomes a config key
that the service applies live, without a restart.

- [ ] **P0 — Calibration**: status (taps recorded per area, sittings, when),
      record more, record noise, reset.
- [ ] **P0 — Training**: what's trained, accuracy, progress, cancel.
- [ ] **P0 — Microphone**: pick the input. `SOURCE` defaults to *this*
      laptop's ALSA name (`alsa_input.pci-0000_00_1f.3.analog-stereo`), so
      it fails on any other machine. Default to the built-in mic, found
      automatically.
- [ ] **P1 — Sensitivity**: how light a tap still counts (the trigger: 6× the
      room's background noise and 4× just before). Show a live level meter
      with the threshold line, so it can be tuned by tapping.
- [ ] **P1 — Noise filter strictness**: the gate keeps 97% of real taps; stricter
      rejects more knocks but misses more taps.
- [ ] **P1 — Only act when sure**: the confidence cutoff (answer more taps, or be
      right more often), with the measured trade-off shown.
- [ ] **P1 — Typing pause**: taps are ignored for 0.4 s after a key press.
- [ ] **P1 — Double/triple timing**: the gap window.
- [x] **Settings page** in the bar panel (the gear), built 2026-11-06:
      calibration status and "Calibrate again", the noise filter, feedback
      and sound (socket `set`: saved without retraining).
- [ ] **P1 — Feedback** (flash / notify / none): now in settings too; remove it
      from the layout editor.
- [ ] **P1 — Start at login** (enable the systemd unit), and pause/resume.
- [ ] **P2 — Data**: where recordings are, how much space, export, delete.

## 5. Theme and UI

- [ ] **P0 — Follow the theme's colours.** Hard-coded colours don't follow
      Omarchy themes: `zoneColors`, `good`/`mid`/`bad` in `Editor.qml` (the
      flash in `Service.qml` already uses the accent). On a light theme they become pastel and
      low-contrast. The theme's `colors.toml`
      (`~/.local/state/omarchy/current/theme/`) has `mode = "dark" | "light"`
      and named colours (`red`, `yellow`, `green`, `blue`, `magenta`, `cyan`)
      that `Commons.Color` doesn't pass on. Read them, and re-read them when
      the theme changes. Pick tile opacity per mode, and check text contrast
      in both modes.
- [ ] **P1 — GTK front-ends**: the same for `tapzones/gtk/` (fixed `ZONE_RGB`,
      `ACCENT`, `FG`, `BG`). Follow the GTK theme or prefer-dark.
- [ ] **P1 — Small screens**: at 1366×768 the header covers the top sections'
      labels. Make the header collapsible, or place labels to avoid it.
- [ ] **P1 — Bar icon**: states (off, paused, needs calibration, training,
      ready), and a tooltip with what's going on.
- [ ] **P2 — Keyboard and accessibility**: every control reachable by keyboard,
      with visible focus.
- [ ] **P2 — The flash follows a section's real shape** (the L-shaped corners
      of corners + centre).

## 6. Devices and the real world

- [ ] **P0 — Which screen.** Tap positions are fractions of the laptop's own
      screen. Handle external monitors, a closed lid (stop listening), and
      different resolutions and scales.
- [ ] **P1 — Suspend/resume, headphones, mic changes.** The audio device can
      vanish or change. Reconnect, or pause and say why.
- [ ] **P1 — Sound from the speakers.** Media playback can sound like taps.
      Check the gate against it, and consider pausing during calls.
- [ ] **P1 — Battery and CPU.** Always-on listening and inference: measure the
      cost, and offer to pause on battery.
- [ ] **P1 — Privacy.** The mic is always open. Say so clearly. Audio is kept
      only for calibration recordings, stored locally, and deletable. Show
      when it's listening.

## 7. Packaging and operations

- [ ] **P0 — Install.** `install.sh` has never been run on a clean machine, and
      the systemd unit isn't enabled. Test a fresh install end to end (Omarchy
      and plain Linux), including the venv, the CLI and the plugin.
- [ ] **P1 — Uninstall** script.
- [ ] **P1 — Config migrations.** `gestures.toml` will change (settings,
      double/triple restructure). Add a version key and migrate old files.
- [ ] **P1 — Dependencies**: pin the Python, scikit-learn and numpy versions
      the models are trained with (see model versions above).
- [ ] **P2 — Publish** as an Omarchy plugin: version, changelog, screenshots.

## 8. Quality

- [ ] **P1 — Tests for the service** (the socket protocol, save → train → reload,
      stopping training, progress events) and for config migrations.
- [ ] **P1 — CI** running the unittest suite (with node for the Layout.js parity tests).
- [ ] **P1 — Manual checklist** before a release: dark and light theme, small
      screen, a fresh device with no model, a real mouse drag in the editor.

## 9. Docs

- [ ] **P1 — User guide** for first run and calibration.
- [ ] **P1 — Mark the numbers in docs/** as one laptop's results, not what a new
      device will get.
- [ ] **P1 — Update Roadmap.md** to this list.

---

## Open questions (for the owner)

- How long can a new user be asked to tap before it works: a couple of
  minutes, or closer to ten? (Depends on the learning-curve result.)
- Does a double or triple tap get its own layout, or share the single-tap one?
- Calibration in its own window, or full-screen like the editor?
