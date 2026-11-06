import QtQuick
import qs.Commons
import qs.Commons as Commons
import qs.Ui

// A training run's real progress (the service's state.training: progress
// 0–1 and stage, read from the run's log, train.progress): what it's doing,
// a bar, and the time left, worked out from how fast it has gone so far.
Column {
  id: root
  property var training: null          // state.training, or null
  property real now: Date.now() / 1000
  property color foreground: Commons.Color.foreground
  property string fontFamily: Style.font.family
  property string title: "Training"

  readonly property real progress: root.training && root.training.progress !== undefined ? root.training.progress : 0
  readonly property real elapsed: root.training ? Math.max(0, root.now - root.training.since) : 0
  readonly property string timeLeft: {
    if (root.progress < 0.08 || root.elapsed < 3) return ""
    var s = root.elapsed * (1 - root.progress) / root.progress
    return s < 50 ? "less than a minute left" : "about " + Math.round(s / 60) + " min left"
  }

  spacing: Style.space(6)

  Text {
    width: parent.width
    elide: Text.ElideRight
    text: root.title + " · " + Math.round(root.progress * 100) + "%" + (root.timeLeft ? " · " + root.timeLeft : "")
    color: root.foreground
    font.family: root.fontFamily
    font.pixelSize: Style.font.body
  }

  Rectangle {
    id: track
    width: parent.width
    height: Math.max(4, Style.space(6))
    radius: Style.cornerRadius > 0 ? height / 2 : 0
    color: Qt.tint(Commons.Color.popups.background, Util.alpha(root.foreground, 0.15))

    Rectangle {
      width: Math.max(track.height, track.width * root.progress)
      height: parent.height
      radius: parent.radius
      color: Commons.Color.accent
      Behavior on width { NumberAnimation { duration: 400; easing.type: Easing.OutCubic } }
    }
  }

  Text {
    width: parent.width
    elide: Text.ElideRight
    text: root.training && root.training.stage ? root.training.stage.charAt(0).toUpperCase() + root.training.stage.slice(1) : ""
    color: Util.alpha(root.foreground, 0.7)
    font.family: root.fontFamily
    font.pixelSize: Style.font.bodySmall
  }
}
