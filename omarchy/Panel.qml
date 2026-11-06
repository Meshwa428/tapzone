import QtQuick
import Quickshell
import qs.Commons
import qs.Commons as Commons
import qs.Ui
import "Sfx.js" as Sfx
import "Layout.js" as Layout

// The bar icon's popup, in three pages:
//   first run: a laptop that was never calibrated sees only "Start
//     calibration" (tapzones setup: tap the dots, then everyday noise; the
//     service trains when it closes).
//   main: the newest sound (a tap that ran, or a miss) on a small map of
//     the layout, the section that ran lit. Clicking the section you really
//     tapped runs it instead and keeps the sound as a tap there (a wrong
//     section, or a miss); "Not a tap" keeps it as noise.
//   settings (the gear): calibration, the noise filter ("Record what isn't
//     a tap" keeps every sound the mic catches until Stop), feedback, sound.
// What teaches the noise filter retrains it by itself a little later.
//
// BarWidget.qml owns the bar button and hands this panel the button to
// anchor against, and the service client.
//
// Keys: P pause / resume · M not a tap · 1–9 it was that section
//       S settings · C calibrate · R record / stop · E edit layout · Esc close
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
  readonly property var sections: root.st ? root.st.config.single.sections : []
  // The sound a correction is about: a tap that ran ({name, index}) or a
  // miss ({why}), whichever is newer.
  readonly property var newest: root.st && root.st.newest ? root.st.newest : null

  // "main" or "settings"; a laptop with no recorded taps gets the first-run page instead.
  property string page: "main"
  readonly property var cal: root.st && root.st.calibration ? root.st.calibration : null
  readonly property bool calibrated: root.st !== null && (root.cal === null || root.cal.taps > 0)
  readonly property bool calibrating: root.cal !== null && root.cal.running === true
  readonly property bool showWelcome: root.st !== null && !root.calibrated
  readonly property bool showMain: root.calibrated && root.page === "main"
  readonly property bool showSettings: root.calibrated && root.page === "settings"
  // The last calibration didn't open or crashed, in words.
  readonly property string calibrateError: root.svc && root.svc.calibrateError
    ? "Calibration didn't open: " + root.svc.calibrateError.message : ""

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

  // "today 14:20", or "3 Nov" for an older day.
  function day(t) {
    var d = new Date(t * 1000)
    return d.toDateString() === new Date().toDateString() ? "today " + Qt.formatTime(d, "hh:mm") : Qt.formatDate(d, "d MMM")
  }

  readonly property string status: {
    if (!root.st) return "the service isn't running"
    if (root.calibrating) return "calibrating"
    if (!root.calibrated) return "not calibrated yet"
    if (root.svc.recording) return "recording what isn't a tap"
    if (root.st.paused) return "paused: taps run nothing"
    if (root.st.training) return "training"          // the bar below says what and how far
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

  // Settings: how much this laptop has been recorded, and what came of it.
  readonly property string calibrationLine: !root.cal ? ""
    : root.cal.taps + " taps · " + root.cal.sittings + (root.cal.sittings === 1 ? " sitting" : " sittings")
      + (root.cal.last ? " · " + root.day(root.cal.last) : "")
  readonly property string modelLine: {
    if (!root.st) return ""
    if (root.st.training && root.st.training.job !== "gate") return ""     // the bar says it
    var m = root.st.model
    if (!m.ok) return m.why
    return "Single tap: " + Math.round(m.accuracy * 100) + "% right" + (m.sittings < 2 ? " (estimate from one sitting)" : "")
  }

  // It was section i: runs it (unless it's the one that ran).
  function pick(i) {
    if (root.newest && root.newest.index !== i) root.svc.confirm(i)
  }

  function calibrate() {
    root.close()                // the calibration covers the screen
    root.svc.calibrate()
  }

  function act(t) {
    if (!root.st) return
    var n = parseInt(t)
    if ((t === "c" || t === "C") && !root.calibrating) root.calibrate()
    else if (!root.calibrated) return
    else if (n >= 1 && n <= root.sections.length) root.pick(n - 1)
    else if (t === "s" || t === "S") root.page = root.page === "settings" ? "main" : "settings"
    else if (t === "p" || t === "P") root.svc.toggle()
    else if ((t === "m" || t === "M") && root.newest) root.svc.mistake()
    else if (t === "r" || t === "R") root.svc.recordNoise(!root.svc.recording)
    else if (t === "e" || t === "E") { root.close(); root.svc.openEditor() }
  }

  // What the last correction did, in a few words: for a few seconds, or
  // until a newer sound arrives (that one can be corrected).
  readonly property var result: {
    if (!root.svc) return null
    var m = root.svc.markResult, c = root.svc.confirmResult
    var r = m && (!c || m.at > c.at) ? m : c
    if (!r || root.now - r.at >= 6 || (root.newest && root.newest.at > r.at)) return null
    if (!r.ok) return { ok: false, text: r.message.charAt(0).toUpperCase() + r.message.split(":")[0].slice(1) + "." }
    return { ok: true, text: r === m ? "Kept as noise." : r.was ? "Ran " + r.name + " instead." : "Ran " + r.name + "." }
  }

  Timer { interval: 1000; repeat: true; running: root.opened; onTriggered: root.now = Date.now() / 1000 }
  Connections {
    target: root.svc
    function onMarkResultChanged() { root.now = Date.now() / 1000 }
    function onConfirmResultChanged() { root.now = Date.now() / 1000 }
  }
  onOpenedChanged: {
    // A recording in progress lives on the settings page: open there.
    if (opened) root.page = root.svc && root.svc.recording ? "settings" : "main"
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
          title: root.showSettings ? "Settings" : "tapzones"
          meta: root.status
          foreground: root.foreground
          fontFamily: root.fontFamily
          iconOpacity: root.st && !root.st.paused ? 1.0 : 0.5
          iconComponent: Component {
            Text {
              text: String.fromCodePoint(root.showSettings ? 0xF0493 : 0xF0741)     // md-cog / md-gesture_tap
              color: root.foreground
              font.family: root.fontFamily
              font.pixelSize: Style.font.display
            }
          }
          trailingControl: Component {
            Row {
              visible: root.calibrated
              spacing: Style.space(12)

              SolidButton {
                anchors.verticalCenter: parent.verticalCenter
                iconText: String.fromCodePoint(0xF0493)     // md-cog
                selected: root.page === "settings"
                bordered: true
                foreground: hero.foreground
                fontFamily: hero.fontFamily
                tooltipText: root.page === "settings" ? "Back (S)" : "Settings: calibration, noise filter, feedback (S)"
                onClicked: root.page = root.page === "settings" ? "main" : "settings"
              }

              ToggleSwitch {
                id: listenSwitch
                anchors.verticalCenter: parent.verticalCenter
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
        }

        // Whatever is training, on every page: its real progress.
        TrainingBar {
          visible: root.st !== null && root.st.training !== null
          width: parent.width
          training: root.st ? root.st.training : null
          now: root.now
          foreground: root.foreground
          fontFamily: root.fontFamily
          title: root.st && root.st.training && root.st.training.job === "gate" ? "Retraining the noise filter"
                 : "Training the single-tap model"
        }

        // ================================================== first run
        PanelSeparator { visible: root.showWelcome; width: parent.width }

        Column {
          visible: root.showWelcome
          width: parent.width
          spacing: Style.space(12)

          PanelSectionHeader { text: "CALIBRATE THIS LAPTOP"; foreground: root.foreground; fontFamily: root.fontFamily }

          Text {
            width: parent.width
            wrapMode: Text.Wrap
            text: root.calibrating ? "Follow the steps on the screen."
                  : "tapzones learns how your taps sound on this laptop. Tap the dots that appear, "
                    + "then make everyday noise. About 3 minutes."
            color: root.foreground
            font.family: root.fontFamily
            font.pixelSize: Style.font.body
          }

          Text {
            visible: root.calibrateError !== ""
            width: parent.width
            wrapMode: Text.Wrap
            text: root.calibrateError
            color: Commons.Color.accent
            font.family: root.fontFamily
            font.pixelSize: Style.font.bodySmall
          }

          SolidButton {
            text: root.calibrating ? "Calibrating…" : "Start calibration"
            enabled: !root.calibrating
            active: !root.calibrating
            bordered: true
            foreground: root.foreground
            fontFamily: root.fontFamily
            tooltipText: "Opens full screen. Esc stops it (C)"
            onClicked: root.calibrate()
          }
        }

        // ======================================================= main
        PanelSeparator { visible: root.showMain; width: parent.width }

        Column {
          visible: root.showMain
          width: parent.width
          spacing: Style.space(12)

          // The newest sound on a small map of the layout, the section that
          // ran lit. Click the one you really tapped to run it instead.
          Item {
            width: parent.width
            implicitHeight: Math.max(map.height, info.implicitHeight)

            Rectangle {
              id: map
              readonly property var screen: root.svc ? root.svc.laptopScreen : null
              anchors.verticalCenter: parent.verticalCenter
              width: Math.round(parent.width * 0.5)
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
                  readonly property bool ran: root.newest !== null && root.newest.index === zone.index
                  readonly property bool hot: zoneMouse.containsMouse && zoneMouse.enabled
                  x: inset + modelData.rect[0] * (map.width - 2 * inset) + 1
                  y: inset + modelData.rect[1] * (map.height - 2 * inset) + 1
                  width: modelData.rect[2] * (map.width - 2 * inset) - 2
                  height: modelData.rect[3] * (map.height - 2 * inset) - 2
                  z: modelData.over ? 1 : 0            // drawn over the others, as on the screen
                  radius: Style.cornerRadius
                  color: Qt.tint(Commons.Color.popups.background,
                                 zone.ran ? Util.alpha(Commons.Color.accent, 0.5)
                                 : zone.hot ? Util.alpha(Commons.Color.accent, 0.25) : Util.alpha(root.foreground, 0.08))
                  border.width: zone.ran || zone.hot ? Math.max(2, Style.space(2)) : 1
                  border.color: zone.ran || zone.hot ? Commons.Color.accent
                                                     : Qt.tint(Commons.Color.popups.background, Util.alpha(root.foreground, 0.3))

                  Text {
                    anchors.centerIn: parent
                    width: parent.width - Style.space(4)
                    horizontalAlignment: Text.AlignHCenter
                    elide: Text.ElideRight
                    text: zone.index + 1               // its key; the name is beside the map and on hover
                    color: zone.ran ? root.foreground : Util.alpha(root.foreground, 0.75)
                    font.family: root.fontFamily
                    font.pixelSize: Style.font.caption
                  }

                  MouseArea {
                    id: zoneMouse
                    anchors.fill: parent
                    enabled: root.newest !== null && !zone.ran && root.result === null
                    hoverEnabled: true
                    cursorShape: enabled ? Qt.PointingHandCursor : Qt.ArrowCursor
                    onClicked: root.pick(zone.index)
                  }

                  PanelToolTip {
                    visible: zone.hot
                    text: "It was " + zone.modelData.name + ": run it" + (zone.index < 9 ? " (" + (zone.index + 1) + ")" : "")
                    fontFamily: root.fontFamily
                  }
                }
              }
            }

            Column {
              id: info
              anchors.left: map.right
              anchors.leftMargin: Style.space(16)
              anchors.right: parent.right
              anchors.verticalCenter: parent.verticalCenter
              spacing: Style.space(4)

              Text {
                width: parent.width
                elide: Text.ElideRight
                text: !root.newest ? "No taps yet" : root.newest.name !== undefined ? root.newest.name : "Missed"
                color: root.newest ? root.foreground : root.dim
                font.family: root.fontFamily
                font.pixelSize: Style.font.body
                font.bold: root.newest !== null
              }
              Text {
                visible: root.newest !== null
                width: parent.width
                wrapMode: Text.Wrap
                text: !root.newest ? ""
                      : (root.newest.why === "unsure" ? "Not sure where · " : root.newest.why === "ignored" ? "Taken for noise · " : "")
                        + root.ago(root.newest.at)
                color: root.dim
                font.family: root.fontFamily
                font.pixelSize: Style.font.bodySmall
              }
              Item { width: 1; height: Style.space(4) }

              // What the correction did, for a moment; else what can be done.
              Row {
                visible: root.result !== null
                width: parent.width
                spacing: Style.space(6)
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
                  text: root.result ? root.result.text : ""
                  color: root.foreground
                  font.family: root.fontFamily
                  font.pixelSize: Style.font.bodySmall
                }
              }
              Text {
                visible: root.newest !== null && root.result === null
                width: parent.width
                wrapMode: Text.Wrap
                text: root.newest && root.newest.index !== undefined ? "Wrong? Click the right spot." : "Click where you tapped."
                color: root.dim
                font.family: root.fontFamily
                font.pixelSize: Style.font.caption
              }
              // The service answers it with a sound of its own (marked / nothing).
              SolidButton {
                visible: root.result === null
                tapSound: false
                text: "Not a tap"
                enabled: root.newest !== null
                bordered: true
                foreground: enabled ? root.foreground : root.dim
                fontFamily: root.fontFamily
                tooltipText: "Keep it as noise, so it's ignored next time (M)"
                onClicked: root.svc.mistake()
              }
            }
          }
        }

        // =================================================== settings
        PanelSeparator { visible: root.showSettings; width: parent.width }

        Column {
          visible: root.showSettings
          width: parent.width
          spacing: Style.space(12)

          // ----------------------------------------------- calibration
          PanelSectionHeader { text: "CALIBRATION"; foreground: root.foreground; fontFamily: root.fontFamily }

          Item {
            width: parent.width
            implicitHeight: Math.max(calibrationText.implicitHeight, calibrateButton.implicitHeight)

            Column {
              id: calibrationText
              anchors.left: parent.left
              anchors.right: calibrateButton.left
              anchors.rightMargin: Style.space(12)
              anchors.verticalCenter: parent.verticalCenter
              spacing: Style.space(4)

              Text {
                width: parent.width
                wrapMode: Text.Wrap
                text: root.calibrationLine
                color: root.foreground
                font.family: root.fontFamily
                font.pixelSize: Style.font.body
              }
              Text {
                width: parent.width
                wrapMode: Text.Wrap
                text: root.modelLine
                color: root.dim
                font.family: root.fontFamily
                font.pixelSize: Style.font.bodySmall
              }
              Text {
                visible: root.calibrateError !== ""
                width: parent.width
                wrapMode: Text.Wrap
                text: root.calibrateError
                color: Commons.Color.accent
                font.family: root.fontFamily
                font.pixelSize: Style.font.bodySmall
              }
            }

            SolidButton {
              id: calibrateButton
              anchors.right: parent.right
              anchors.verticalCenter: parent.verticalCenter
              text: root.calibrating ? "Calibrating…" : "Calibrate again"
              enabled: !root.calibrating
              bordered: true
              foreground: enabled ? root.foreground : root.dim
              fontFamily: root.fontFamily
              tooltipText: "Another sitting: more taps make it more accurate, best at another time or lid angle (C)"
              onClicked: root.calibrate()
            }
          }

          PanelSeparator { width: parent.width }

          // ---------------------------------------------- noise filter
          PanelSectionHeader { text: "NOISE FILTER"; foreground: root.foreground; fontFamily: root.fontFamily }

          Text {
            visible: root.svc !== null && !root.svc.recording
            width: parent.width
            wrapMode: Text.Wrap
            text: root.filterLine
            color: root.dim
            font.family: root.fontFamily
            font.pixelSize: Style.font.body
          }

          Column {
            visible: root.svc !== null && root.svc.recording
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
            text: root.svc && root.svc.recording ? "Stop and learn" : "Record what isn’t a tap"
            bordered: true
            active: root.svc ? root.svc.recording : false
            foreground: root.foreground
            fontFamily: root.fontFamily
            tooltipText: root.svc && root.svc.recording ? "Save the sounds; the filter retrains with them (R)"
                         : "Every sound the mic catches is kept as not a tap, until you stop (R)"
            onClicked: root.svc.recordNoise(!root.svc.recording)
          }

          PanelSeparator { width: parent.width }

          // -------------------------------------------- feedback, sound
          Item {
            width: parent.width
            implicitHeight: feedbackChoices.implicitHeight

            PanelSectionHeader {
              anchors.left: parent.left
              anchors.verticalCenter: parent.verticalCenter
              text: "FEEDBACK"
              foreground: root.foreground
              fontFamily: root.fontFamily
            }
            SolidChoices {
              id: feedbackChoices
              anchors.right: parent.right
              readonly property var tips: ({ flash: "The section lights up",
                                             notify: "A desktop notification with the section's name and command",
                                             none: "No feedback" })
              options: Layout.FEEDBACK.map(f => ({ value: f, label: f, tooltip: tips[f] }))
              value: root.st ? root.st.config.feedback : ""
              foreground: root.foreground
              fontSize: Style.font.bodySmall
              onChanged: v => root.svc.setOptions({ feedback: v })
            }
          }

          Item {
            width: parent.width
            implicitHeight: soundToggle.implicitHeight

            PanelSectionHeader {
              anchors.left: parent.left
              anchors.verticalCenter: parent.verticalCenter
              text: "SOUND"
              foreground: root.foreground
              fontFamily: root.fontFamily
            }
            SolidToggle {
              id: soundToggle
              anchors.right: parent.right
              checked: root.st !== null && root.st.config.sound !== false
              foreground: root.foreground
              onToggled: root.svc.setOptions({ sound: !soundToggle.checked })
            }
          }
        }

        PanelSeparator { visible: root.calibrated; width: parent.width }

        // A link, not a button: it leaves the panel for the editor.
        Item {
          id: editLink
          visible: root.calibrated
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
