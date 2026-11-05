import QtQuick
import Quickshell
import Quickshell.Io
import Quickshell.Wayland
import qs.Commons
import qs.Commons as Commons

// The shell side of tapzones: a client of the service (tapzones/service/engine.py, run by
// systemd as tapzones.service) over its socket. Keeps the service's state
// for the bar widget and the editor, and draws the flash feedback in the
// theme's colours. The service keeps listening if the shell restarts.
Item {
  id: root

  property var shell: null
  property var manifest: null

  readonly property string pluginId: (root.manifest && root.manifest.id) || "meshwa.tapzones"
  // No fallback to a shared directory like /tmp: another user could put a
  // socket there first (the service refuses to run without it too).
  readonly property string runtimeDir: Quickshell.env("XDG_RUNTIME_DIR") || ""
  readonly property string socketPath: runtimeDir ? runtimeDir + "/tapzones.sock" : ""

  // The last {"event": "state"}: paused, config, model, training, estimate…
  property var state: null
  // The editor's live-estimate data, once asked for (Layout.quickEstimate).
  property var estimate: null
  property var lastTap: null
  // The newest {"event": "marked"} and the live count of a noise recording.
  // What the last "That wasn't a tap" did: { ok, name, total, at } when it
  // was kept, { ok: false, message, at } when there was nothing to mark.
  property var markResult: null
  // What the last "That was a tap" did: { ok, name, at }, or { ok: false, message, at }.
  property var confirmResult: null
  property int noiseCount: 0
  readonly property bool online: sock.connected && state !== null
  readonly property bool paused: online && state.paused
  readonly property bool training: online && state.training !== null
  readonly property bool recording: online && state.recording !== null

  // Every event from the service, for the editor (saved, error, trained…).
  signal event(var msg)

  function send(obj) {
    if (!sock.connected) return false
    sock.write(JSON.stringify(obj) + "\n")
    sock.flush()
    return true
  }

  function toggle() { send({ cmd: "toggle" }) }
  // The last tap that ran a command wasn't a tap: the service keeps its
  // sound as noise and retrains the noise filter.
  function mistake() { send({ cmd: "mistake" }) }
  // The last sound that ran nothing was a tap in section i: the service
  // runs it and keeps it as a tap.
  function confirm(i) { send({ cmd: "confirm", section: i }) }
  function recordNoise(on) {
    if (on) root.noiseCount = 0
    send({ cmd: "noise", on: on })
  }

  function openEditor() {
    if (root.shell && typeof root.shell.summon === "function") root.shell.summon(root.pluginId, "{}")
  }

  function handle(line) {
    var msg
    try { msg = JSON.parse(line) } catch (e) { return }
    if (msg.event === "state") root.state = msg
    else if (msg.event === "estimate") root.estimate = msg
    else if (msg.event === "marked")
      root.markResult = { ok: true, name: msg.name, total: msg.total, at: Date.now() / 1000 }
    else if (msg.event === "error" && msg.cmd === "mistake")
      root.markResult = { ok: false, message: msg.message, at: Date.now() / 1000 }
    else if (msg.event === "confirmed")
      root.confirmResult = { ok: true, name: msg.name, at: Date.now() / 1000 }
    else if (msg.event === "error" && msg.cmd === "confirm")
      root.confirmResult = { ok: false, message: msg.message, at: Date.now() / 1000 }
    else if (msg.event === "noise") root.noiseCount = msg.count
    else if (msg.event === "tap" || msg.event === "unsure") {
      root.lastTap = msg
      if (msg.feedback === "flash") flash.play(msg.event === "tap" ? msg.rect : null, msg.event === "tap" ? msg.name : "not sure: tap again")
    }
    root.event(msg)
  }

  Socket {
    id: sock
    path: root.socketPath
    connected: root.socketPath !== ""
    onConnectedChanged: {
      if (connected) {
        root.send({ cmd: "hello", ui: true })
      } else {
        root.state = null
      }
    }
    parser: SplitParser { onRead: function(line) { root.handle(line) } }
  }

  // The service may start after the shell, or restart: keep knocking.
  Timer {
    interval: 2000
    repeat: true
    running: !sock.connected && root.socketPath !== ""
    onTriggered: { sock.connected = false; sock.connected = true }
  }

  // ------------------------------------------------------------ flash

  // Taps are heard by the laptop's own mic, so sections are fractions of
  // the built-in screen.
  readonly property var laptopScreen: Quickshell.screens.filter(function(s) { return /^eDP/.test(s.name) })[0] || Quickshell.screens[0]

  PanelWindow {
    id: flash
    property var rect: null
    property string label: ""
    // How long it stays fully shown before fading: the section flash is a
    // glance; "tap again" is something to read.
    property int hold: 0

    function play(r, text) {
      flash.rect = r
      flash.label = text
      flash.hold = r === null ? 1600 : 0
      fade.restart()
    }

    visible: fade.running
    screen: root.laptopScreen
    color: "transparent"
    anchors { top: true; bottom: true; left: true; right: true }
    exclusionMode: ExclusionMode.Ignore
    WlrLayershell.namespace: "omarchy-tapzones-flash"
    WlrLayershell.layer: WlrLayer.Overlay
    WlrLayershell.keyboardFocus: WlrKeyboardFocus.None
    mask: Region {}            // clicks go straight through

    Item {
      id: content
      anchors.fill: parent
      opacity: 0

      SequentialAnimation {
        id: fade
        PropertyAction { target: content; property: "opacity"; value: 1 }
        PauseAnimation { duration: flash.hold }
        NumberAnimation { target: content; property: "opacity"; from: 1; to: 0; duration: 450; easing.type: Easing.InQuad }
      }

      Rectangle {
        visible: flash.rect !== null
        x: flash.rect ? flash.rect[0] * content.width + Style.space(6) : 0
        y: flash.rect ? flash.rect[1] * content.height + Style.space(6) : 0
        width: flash.rect ? flash.rect[2] * content.width - Style.space(12) : 0
        height: flash.rect ? flash.rect[3] * content.height - Style.space(12) : 0
        radius: Style.cornerRadius
        color: Util.alpha(Commons.Color.accent, 0.16)
        border.color: Util.alpha(Commons.Color.accent, 0.85)
        border.width: Math.max(2, Style.space(3))

        Text {
          anchors.centerIn: parent
          text: flash.label
          color: Commons.Color.foreground
          font.family: Style.font.menuFamily
          font.pixelSize: Style.font.display
        }
      }

      Rectangle {
        visible: flash.rect === null
        anchors.centerIn: parent
        width: unsure.implicitWidth + Style.spacing.xl * 2
        height: unsure.implicitHeight + Style.spacing.md * 2
        radius: Style.cornerRadius
        color: Commons.Color.tooltip.background
        border.color: Commons.Color.tooltip.border

        Text {
          id: unsure
          anchors.centerIn: parent
          text: flash.label
          color: Commons.Color.tooltip.text
          font.family: Style.font.menuFamily
          font.pixelSize: Style.font.body
        }
      }
    }
  }
}
