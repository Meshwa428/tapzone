import QtQuick
import Quickshell
import qs.Commons
import qs.Commons as Commons
import qs.Ui
import "Sfx.js" as Sfx

// The shell's Button, but opaque: its state fills (hover, selected) are
// translucent tints, so over a picture the button turns see-through. Here
// the same tints are laid onto `base` instead, so the button keeps a solid
// face and only its colour changes. A press plays the tap sound, unless the
// service answers it with its own (`tapSound: false`).
Button {
  id: root
  property color base: Commons.Color.popups.background
  property bool tapSound: true

  color: Qt.tint(root.base, root.hot ? Style.hoverFillFor(root.foreground, root.accent)
                          : root.selected || root.active ? Style.selectedFillFor(root.foreground, root.accent)
                          : "transparent")

  onClicked: {
    var c = root.tapSound ? Sfx.command(Qt.resolvedUrl("."), "tap") : null
    if (c) Quickshell.execDetached(c)
  }
}
