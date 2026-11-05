import QtQuick
import Quickshell
import qs.Ui
import qs.Commons

// tapzones in the bar: a tap glyph that dims while paused or while the
// service isn't running, and lights up while a section model trains.
// Left click opens the layout editor; right click pauses or resumes.
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
    var lines = ["tapzones: " + (svc.paused ? "paused" : "listening")]
    var m = st.model
    if (svc.training) lines.push("training the " + st.training.gesture + "-tap model…")
    else if (m.ok) lines.push("single tap: " + Math.round(m.answered_accuracy * 100) + "% right on the "
                              + Math.round(m.answered * 100) + "% of taps it answers")
    else lines.push(m.why)
    var t = svc.lastTap
    if (t) lines.push("last: " + (t.event === "tap" ? t.name : "not sure") + " (" + Math.round(t.conf * 100) + "%)")
    lines.push("click: edit layout · right click: " + (svc.paused ? "resume" : "pause"))
    return lines.join("\n")
  }

  implicitWidth: button.implicitWidth
  implicitHeight: barSize

  WidgetButton {
    id: button
    anchors.fill: parent
    bar: root.bar
    text: String.fromCodePoint(0xF0741)     // md-gesture_tap
    dimmed: !root.svc || !root.svc.online || root.svc.paused
    active: root.svc ? root.svc.training : false
    tooltipText: root.tooltip
    onPressed: function(b) {
      if (!root.svc || !root.svc.online) root.setUp()
      else if (b === Qt.RightButton) root.svc.toggle()
      else root.svc.openEditor()
    }
  }
}
