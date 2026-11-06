import QtQuick
import Quickshell
import Quickshell.Io
import qs.Ui
import qs.Commons

// tapzones in the bar: a tap glyph that dims while paused or while the
// service isn't running, and lights up while a model trains or noise is
// being recorded. Left click opens the panel (Panel.qml); right click
// pauses or resumes.
BarWidget {
  id: root
  moduleName: "meshwa.tapzones"

  readonly property var svc: bar && bar.shell ? bar.shell.serviceFor("meshwa.tapzones") : null
  readonly property var st: svc ? svc.state : null
  readonly property string installer: decodeURIComponent(String(Qt.resolvedUrl("../install.sh")).replace(/^file:\/\//, ""))

  // With the service down, a click runs the installer in a terminal: it
  // sets up the Python env and the systemd unit, or restarts them.
  function setUp() {
    Quickshell.execDetached(["omarchy", "launch", "terminal", "bash", "-c",
      "'" + root.installer + "'; echo; read -rp 'Press Enter to close.'"])
  }

  readonly property string tooltip: {
    if (!svc || !svc.online) return "tapzones isn't running\nclick: set it up (runs install.sh in a terminal)"
    var lines = ["tapzones: " + (svc.recording ? "recording what isn't a tap" : svc.paused ? "paused" : "listening")]
    var m = st.model
    if (svc.recording) lines.push(svc.noiseCount + " sounds caught")
    else if (svc.training) {
      // The tooltip is text: the bar is drawn with blocks (TrainingBar.qml in the panel).
      var t = st.training, p = t.progress || 0, n = Math.round(p * 16)
      var spent = Date.now() / 1000 - t.since, rest = p >= 0.08 ? spent * (1 - p) / p : -1
      lines.push(t.job === "gate" ? "retraining the noise filter" : "training the " + t.job + "-tap model")
      lines.push("█".repeat(n) + "░".repeat(16 - n) + "  " + Math.round(p * 100) + "%"
                 + (rest < 0 ? "" : rest < 50 ? " · under a minute left" : " · about " + Math.round(rest / 60) + " min left"))
    }
    else if (m.ok) lines.push("single tap: " + Math.round(m.answered_accuracy * 100) + "% right on the "
                              + Math.round(m.answered * 100) + "% of taps it answers")
    else lines.push(m.why)
    var t = svc.lastTap
    if (t) lines.push("last: " + (t.event === "tap" ? t.name : "not sure") + " (" + Math.round(t.conf * 100) + "%)")
    lines.push("click: menu · right click: " + (svc.paused ? "resume" : "pause"))
    return lines.join("\n")
  }

  // ---- The popup. Bar.findPanelWidget routes shell.summon/hide/toggle to
  //      open/close/opened on the bar-widget root.
  readonly property bool opened: panelLoader.item ? panelLoader.item.opened === true : false
  readonly property bool popoutSwitchClosing: panelLoader.item ? panelLoader.item.popoutSwitchClosing === true : false
  function open() { if (panelLoader.item) panelLoader.item.open() }
  function close() { if (panelLoader.item) panelLoader.item.close() }
  function closeForPopoutSwitch() { if (panelLoader.item) panelLoader.item.closeForPopoutSwitch() }
  function togglePanel() { if (panelLoader.item) panelLoader.item.toggle() }

  function injectPanel() {
    var target = panelLoader.item
    if (!target) return
    target.bar = root.bar
    target.anchorItem = button
    target.hostWidget = root
    target.svc = Qt.binding(function() { return root.svc })
  }

  onBarChanged: injectPanel()

  implicitWidth: button.implicitWidth
  implicitHeight: barSize

  Loader {
    id: panelLoader
    active: true
    source: Qt.resolvedUrl("Panel.qml")
    visible: false
    onLoaded: {
      root.injectPanel()
      Qt.callLater(root.injectPanel)
    }
  }

  // omarchy-shell meshwa.tapzones toggle — e.g. from a Hyprland binding.
  IpcHandler {
    target: "meshwa.tapzones"
    function open(): void { root.open() }
    function close(): void { root.close() }
    function toggle(): void { root.togglePanel() }
  }

  WidgetButton {
    id: button
    anchors.fill: parent
    bar: root.bar
    text: String.fromCodePoint(0xF0741)     // md-gesture_tap
    dimmed: !root.svc || !root.svc.online || root.svc.paused
    active: root.svc ? root.svc.training || root.svc.recording : false
    tooltipText: root.opened ? "" : root.tooltip
    onPressed: function(b) {
      if (!root.svc || !root.svc.online) root.setUp()
      else if (b === Qt.RightButton) root.svc.toggle()
      else root.togglePanel()
    }
  }
}
