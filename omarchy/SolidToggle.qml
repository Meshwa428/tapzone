import QtQuick
import Quickshell
import qs.Commons
import qs.Commons as Commons
import qs.Ui
import "Sfx.js" as Sfx

// "On"/"Off" in words, then the shell's switch on a solid track: accent when
// on, grey when off. The shared switch only tints its track, which on its
// own is hard to tell apart. Flipping it plays toggle on / toggle off.
Row {
  id: root
  property bool checked: false
  property color foreground: Commons.Color.foreground
  signal toggled()

  spacing: Style.spacing.lg

  Text {
    anchors.verticalCenter: parent.verticalCenter
    text: root.checked ? "On" : "Off"
    color: root.foreground
    font.family: Style.font.family
    font.pixelSize: Style.font.body
    font.bold: root.checked
  }
  Item {
    anchors.verticalCenter: parent.verticalCenter
    implicitWidth: toggle.implicitWidth
    implicitHeight: toggle.implicitHeight
    Rectangle {
      anchors.centerIn: parent
      width: toggle.trackWidth
      height: toggle.trackHeight
      radius: toggle.rounded ? height / 2 : 0
      color: root.checked ? Commons.Color.accent : Qt.tint(Commons.Color.popups.background, Util.alpha(root.foreground, 0.25))
      Behavior on color { ColorAnimation { duration: 120 } }
    }
    ToggleSwitch {
      id: toggle
      anchors.fill: parent
      checked: root.checked
      foreground: root.foreground
      // After the change: switching sound itself off is silent, on is heard.
      onToggled: {
        var was = root.checked
        root.toggled()
        var c = Sfx.command(Qt.resolvedUrl("."), was ? "toggle_off" : "toggle_on")
        if (c) Quickshell.execDetached(c)
      }
    }
  }
}
