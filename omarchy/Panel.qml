import QtQuick
import Quickshell
import qs.Commons
import qs.Commons as Commons
import qs.Ui
import "Sfx.js" as Sfx

// The bar icon's popup: whether tapzones is listening, and what to do when
// it got a sound wrong. "That wasn't a tap" keeps the last tap's sound as
// noise; "That was a tap" takes the last sound that ran nothing, asks which
// section it was meant for, runs it and keeps it as a tap; "Record what
// isn't a tap" keeps every sound the mic catches until Stop. All of them
// teach the noise filter, which the service retrains by itself a little later.
//
// BarWidget.qml owns the bar button and hands this panel the button to
// anchor against, and the service client.
//
// Keys: P pause / resume · M that wasn't a tap · T that was a tap (then 1–9 the section)
//       R record / stop · E edit layout · Esc close
Panel {
  id: root
  moduleName: "meshwa.tapzones"
  manageIpc: false

  property var anchorItem: null
  property var hostWidget: null
  property var svc: null
  readonly property var barIdentity: hostWidget || root

  readonly property var st: svc && svc.online ? svc.state : null
  // The buttons' sounds follow the saved Sound setting.
  onStChanged: if (st) Sfx.on = st.config.sound !== false
  readonly property color foreground: bar ? bar.foreground : Commons.Color.foreground
  readonly property color dim: Qt.darker(foreground, 1.55)
  readonly property string fontFamily: bar ? bar.fontFamily : Style.font.family
  property real now: Date.now() / 1000
  // "That was a tap" pressed: the section map is open to pick from.
  property bool picking: false
  readonly property var sections: root.st ? root.st.config.single.sections : []

  function open() { root.controller.show() }
  function close() { root.controller.hide() }
  function toggle() { root.opened ? root.close() : root.open() }

  function ago(t) {
    var s = Math.max(0, Math.round(root.now - t))
    return s < 60 ? s + " s ago" : Math.floor(s / 60) + " min ago"
  }

  function clock(since) {
    var s = Math.max(0, Math.round(root.now - since))
    return Math.floor(s / 60) + ":" + ("0" + s % 60).slice(-2)
  }

  readonly property string status: {
    if (!root.st) return "the service isn't running"
    if (root.svc.recording) return "recording what isn't a tap"
    if (root.st.paused) return "paused: taps run nothing"
    if (root.st.training) return root.st.training.job === "gate" ? "retraining the noise filter…" : "training the " + root.st.training.job + "-tap model…"
    if (!root.st.model.ok) return root.st.model.why
    return "listening"
  }

  readonly property string filterLine: {
    if (!root.st) return ""
    var n = root.st.noise
    var parts = [root.st.gate ? "learned from " + n.learned + " sounds" : "not trained yet"]
    if (n.marked > 0) parts.push(n.marked + " marked by you")
    if (n.confirmed > 0) parts.push(n.confirmed + " confirmed taps")
    if (root.st.training && root.st.training.job === "gate") parts.push("retraining now")
    else if (n.retrain_due) parts.push("retrains in a moment")
    return parts.join(" · ")
  }

  function pick(i) {
    root.picking = false
    root.svc.confirm(i)
  }

  function act(t) {
    if (!root.st) return
    var n = parseInt(t)
    if (root.picking && n >= 1 && n <= root.sections.length) root.pick(n - 1)
    else if ((t === "t" || t === "T") && root.st.missed) root.picking = !root.picking
    else if (t === "p" || t === "P") root.svc.toggle()
    else if ((t === "m" || t === "M") && root.st.last) root.svc.mistake()
    else if (t === "r" || t === "R") root.svc.recordNoise(!root.svc.recording)
    else if (t === "e" || t === "E") { root.close(); root.svc.openEditor() }
  }

  // The answer to the last "That wasn't a tap", for a few seconds.
  readonly property var result: svc && svc.markResult && root.now - svc.markResult.at < 6 ? svc.markResult : null
  // …and to the last "That was a tap".
  readonly property var confirmed: svc && svc.confirmResult && root.now - svc.confirmResult.at < 6 ? svc.confirmResult : null

  Timer { interval: 1000; repeat: true; running: root.opened; onTriggered: root.now = Date.now() / 1000 }
  Connections {
    target: root.svc
    function onMarkResultChanged() { root.now = Date.now() / 1000 }
    function onConfirmResultChanged() { root.now = Date.now() / 1000 }
  }
  onOpenedChanged: {
    root.picking = false
    if (opened) { root.now = Date.now() / 1000; Qt.callLater(function() { keyCatcher.forceActiveFocus() }) }
  }

  KeyboardPanel {
    id: panel
    anchorItem: root.anchorItem
    owner: root.barIdentity
    bar: root.bar
    open: root.opened
    focusTarget: keyCatcher
    contentWidth: panel.fittedContentWidth(Style.space(420))
    contentHeight: panel.fittedContentHeight(column.implicitHeight)

    PanelKeyCatcher {
      id: keyCatcher
      anchors.fill: parent
      onCloseRequested: root.close()
      onTabRequested: function(direction) { root.switchPanel(direction) }
      onTextKey: function(t) { root.act(t) }

      Column {
        id: column
        width: parent.width
        spacing: Style.space(12)

        PanelHero {
          id: hero
          width: parent.width
          title: "tapzones"
          meta: root.status
          foreground: root.foreground
          fontFamily: root.fontFamily
          iconOpacity: root.st && !root.st.paused ? 1.0 : 0.5
          iconComponent: Component {
            Text {
              text: String.fromCodePoint(0xF0741)     // md-gesture_tap
              color: root.foreground
              font.family: root.fontFamily
              font.pixelSize: Style.font.display
            }
          }
          trailingControl: Component {
            ToggleSwitch {
              id: listenSwitch
              visible: root.st !== null
              checked: root.st !== null && !root.st.paused
              foreground: hero.foreground
              onToggled: root.svc.toggle()

              PanelToolTip {
                visible: listenSwitch.containsMouse
                text: root.st && root.st.paused ? "Resume: taps run their commands again (P)" : "Pause: taps run nothing (P)"
                fontFamily: hero.fontFamily
              }
            }
          }
        }

        PanelSeparator { visible: root.st !== null; width: parent.width }

        // ---------------------------------------------------- last tap
        PanelSectionHeader { visible: root.st !== null; text: "LAST TAP"; foreground: root.foreground; fontFamily: root.fontFamily }

        Item {
          visible: root.st !== null
          width: parent.width
          implicitHeight: Math.max(lastText.implicitHeight, mistakeButton.implicitHeight)

          Text {
            id: lastText
            anchors.left: parent.left
            anchors.right: mistakeButton.left
            anchors.rightMargin: Style.space(12)
            anchors.verticalCenter: parent.verticalCenter
            wrapMode: Text.Wrap
            text: root.st && root.st.last ? root.st.last.name + ", " + root.ago(root.st.last.at) : "No tap to mark."
            color: root.st && root.st.last ? root.foreground : root.dim
            font.family: root.fontFamily
            font.pixelSize: Style.font.body
          }

          // Says what the press did: a check for a few seconds once it's kept.
          // The service answers it with a sound of its own (marked / nothing).
          SolidButton {
            id: mistakeButton
            tapSound: false
            readonly property bool justMarked: root.result !== null && root.result.ok
            anchors.right: parent.right
            anchors.verticalCenter: parent.verticalCenter
            text: justMarked ? "Marked" : "That wasn’t a tap"
            iconText: justMarked ? String.fromCodePoint(0xF012C) : ""     // md-check
            enabled: !justMarked && root.st !== null && root.st.last !== null
            bordered: true
            foreground: enabled || justMarked ? root.foreground : root.dim
            fontFamily: root.fontFamily
            tooltipText: enabled ? "Keeps its sound as noise, so the filter learns to ignore it (M)"
                                 : "Nothing to mark: no tap has run a command since the last one you marked"
            onClicked: root.svc.mistake()
          }
        }

        // The answer to the press, in words.
        Row {
          visible: root.result !== null
          width: parent.width
          spacing: Style.space(8)
          Text {
            id: resultIcon
            text: String.fromCodePoint(root.result && root.result.ok ? 0xF012C : 0xF02FC)   // md-check / md-information
            color: root.result && root.result.ok ? Commons.Color.accent : root.dim
            font.family: root.fontFamily
            font.pixelSize: Style.font.body
          }
          Text {
            width: parent.width - resultIcon.width - parent.spacing
            wrapMode: Text.Wrap
            text: !root.result ? "" : root.result.ok ? "Marked the last tap as not a tap." : "Nothing to mark."
            color: root.foreground
            font.family: root.fontFamily
            font.pixelSize: Style.font.bodySmall
          }
        }

        PanelSeparator { visible: root.st !== null; width: parent.width }

        // ---------------------------------------------------- missed tap
        // The newest sound that ran nothing: turned away as noise, or a tap
        // it wasn't sure where. "That was a tap" asks which section.
        PanelSectionHeader { visible: root.st !== null; text: "MISSED TAP"; foreground: root.foreground; fontFamily: root.fontFamily }

        Item {
          visible: root.st !== null
          width: parent.width
          implicitHeight: Math.max(missedText.implicitHeight, confirmButton.implicitHeight)

          Text {
            id: missedText
            anchors.left: parent.left
            anchors.right: confirmButton.left
            anchors.rightMargin: Style.space(12)
            anchors.verticalCenter: parent.verticalCenter
            wrapMode: Text.Wrap
            text: root.st && root.st.missed
                  ? (root.st.missed.why === "unsure" ? "Not sure where, " : "Taken for noise, ") + root.ago(root.st.missed.at)
                  : "No missed tap."
            color: root.st && root.st.missed ? root.foreground : root.dim
            font.family: root.fontFamily
            font.pixelSize: Style.font.body
          }

          SolidButton {
            id: confirmButton
            readonly property bool justConfirmed: root.confirmed !== null && root.confirmed.ok
            anchors.right: parent.right
            anchors.verticalCenter: parent.verticalCenter
            text: justConfirmed ? "Done" : root.picking ? "Cancel" : "That was a tap"
            iconText: justConfirmed ? String.fromCodePoint(0xF012C) : ""     // md-check
            enabled: !justConfirmed && root.st !== null && root.st.missed !== null
            selected: root.picking
            bordered: true
            foreground: enabled || justConfirmed ? root.foreground : root.dim
            fontFamily: root.fontFamily
            tooltipText: !enabled ? "Nothing to confirm: no sound has been missed since the last one you confirmed"
                         : root.picking ? "Close the map without confirming (T)"
                         : "Pick the section you tapped: it runs, and the filter learns from it (T)"
            onClicked: root.picking = !root.picking
          }
        }

        // Where was it? The layout in use, small; a click picks a section.
        Column {
          visible: root.picking && root.st !== null && root.st.missed !== null
          width: parent.width
          spacing: Style.space(6)

          Text {
            text: "Where did you tap?"
            color: root.foreground
            font.family: root.fontFamily
            font.pixelSize: Style.font.bodySmall
          }

          Rectangle {
            id: map
            readonly property var screen: root.svc ? root.svc.laptopScreen : null
            width: parent.width
            height: Math.round(width * (screen ? screen.height / screen.width : 0.5625))
            radius: Style.cornerRadius
            color: Commons.Color.popups.background
            border.width: 1
            border.color: Qt.tint(Commons.Color.popups.background, Util.alpha(root.foreground, 0.35))

            Repeater {
              model: root.sections
              delegate: Rectangle {
                id: zone
                required property var modelData
                required property int index
                readonly property real inset: Math.max(2, Style.space(3))
                x: inset + modelData.rect[0] * (map.width - 2 * inset) + 1
                y: inset + modelData.rect[1] * (map.height - 2 * inset) + 1
                width: modelData.rect[2] * (map.width - 2 * inset) - 2
                height: modelData.rect[3] * (map.height - 2 * inset) - 2
                z: modelData.over ? 1 : 0            // drawn over the others, as on the screen
                radius: Style.cornerRadius
                color: Qt.tint(Commons.Color.popups.background,
                               zoneMouse.containsMouse ? Util.alpha(Commons.Color.accent, 0.35) : Util.alpha(root.foreground, 0.1))
                border.width: zoneMouse.containsMouse ? Math.max(2, Style.space(2)) : 1
                border.color: zoneMouse.containsMouse ? Commons.Color.accent
                                                      : Qt.tint(Commons.Color.popups.background, Util.alpha(root.foreground, 0.3))

                Text {
                  anchors.centerIn: parent
                  width: parent.width - Style.space(6)
                  horizontalAlignment: Text.AlignHCenter
                  elide: Text.ElideRight
                  text: (zone.index < 9 ? (zone.index + 1) + "  " : "") + zone.modelData.name
                  color: root.foreground
                  font.family: root.fontFamily
                  font.pixelSize: Style.font.caption
                }

                MouseArea {
                  id: zoneMouse
                  anchors.fill: parent
                  hoverEnabled: true
                  cursorShape: Qt.PointingHandCursor
                  onClicked: root.pick(zone.index)
                }
              }
            }
          }
        }

        // The answer to "That was a tap", in words.
        Row {
          visible: root.confirmed !== null
          width: parent.width
          spacing: Style.space(8)
          Text {
            id: confirmIcon
            text: String.fromCodePoint(root.confirmed && root.confirmed.ok ? 0xF012C : 0xF02FC)   // md-check / md-information
            color: root.confirmed && root.confirmed.ok ? Commons.Color.accent : root.dim
            font.family: root.fontFamily
            font.pixelSize: Style.font.body
          }
          Text {
            width: parent.width - confirmIcon.width - parent.spacing
            wrapMode: Text.Wrap
            text: !root.confirmed ? "" : root.confirmed.ok ? "Ran " + root.confirmed.name + ". Kept as a tap." : "Nothing to confirm."
            color: root.foreground
            font.family: root.fontFamily
            font.pixelSize: Style.font.bodySmall
          }
        }

        PanelSeparator { visible: root.st !== null; width: parent.width }

        // ------------------------------------------------ noise filter
        PanelSectionHeader { visible: root.st !== null; text: "NOISE FILTER"; foreground: root.foreground; fontFamily: root.fontFamily }

        Text {
          visible: root.st !== null && !root.svc.recording
          width: parent.width
          wrapMode: Text.Wrap
          text: root.filterLine
          color: root.dim
          font.family: root.fontFamily
          font.pixelSize: Style.font.body
        }

        Column {
          visible: root.st !== null && root.svc.recording
          width: parent.width
          spacing: Style.space(6)

          Text {
            text: root.svc ? root.svc.noiseCount + (root.svc.noiseCount === 1 ? " sound" : " sounds") + " caught · "
                  + (root.st && root.st.recording ? root.clock(root.st.recording.since) : "") : ""
            color: root.foreground
            font.family: root.fontFamily
            font.pixelSize: Style.font.display
          }
          Text {
            width: parent.width
            wrapMode: Text.Wrap
            text: "Do what normally happens around the laptop: type, click, set a mug down, knock the desk, "
                  + "move the lid. Don’t tap the screen. Nothing runs while this records."
            color: root.dim
            font.family: root.fontFamily
            font.pixelSize: Style.font.body
          }
        }

        SolidButton {
          visible: root.st !== null
          text: root.svc && root.svc.recording ? "Stop and learn" : "Record what isn’t a tap"
          bordered: true
          active: root.svc ? root.svc.recording : false
          foreground: root.foreground
          fontFamily: root.fontFamily
          tooltipText: root.svc && root.svc.recording ? "Save the sounds; the filter retrains with them (R)"
                       : "Every sound the mic catches is kept as not a tap, until you stop (R)"
          onClicked: root.svc.recordNoise(!root.svc.recording)
        }

        PanelSeparator { visible: root.st !== null; width: parent.width }

        // A link, not a button: it leaves the panel for the editor.
        Item {
          id: editLink
          visible: root.st !== null
          implicitWidth: editRow.implicitWidth
          implicitHeight: editRow.implicitHeight
          readonly property color ink: editMouse.containsMouse ? Commons.Color.accent : root.foreground

          Row {
            id: editRow
            spacing: Style.space(6)

            Text {
              anchors.verticalCenter: parent.verticalCenter
              text: "Edit tap layout"
              font.underline: true
              color: editLink.ink
              font.family: root.fontFamily
              font.pixelSize: Style.font.body
            }
            Text {
              anchors.verticalCenter: parent.verticalCenter
              text: String.fromCodePoint(0xF0054)     // md-arrow_right
              color: editLink.ink
              font.family: root.fontFamily
              font.pixelSize: Style.font.body
            }
          }

          MouseArea {
            id: editMouse
            anchors.fill: parent
            hoverEnabled: true
            cursorShape: Qt.PointingHandCursor
            onClicked: { root.close(); root.svc.openEditor() }
          }

          PanelToolTip {
            visible: editMouse.containsMouse
            text: "Sections and what each one runs (E)"
            fontFamily: root.fontFamily
          }
        }
      }
    }
  }
}
