import QtQuick
import Quickshell
import Quickshell.Wayland
import qs.Commons
import qs.Commons as Commons
import qs.Ui
import "Layout.js" as Layout

// The layout editor (docs/Gestures/Sections.md), in the shell: the same
// editor as `taps.py layout`, drawn in the theme. Pick a gesture, start from
// a preset or drag out sections, name them and give each a command. Every
// section shows how well the mic tells it apart, estimated live from the
// recorded taps. Save hands the config to the service, which writes
// gestures.toml and trains the section model.
//
// Mouse: drag on empty screen draws a section · drag inside one moves it ·
//        drag its edge resizes it · right-click deletes it
// Keys:  1–5 gesture · P / Shift+P preset · E on/off · F feedback ·
//        Delete remove · U undo · S save · Esc close
Item {
  id: root

  property var shell: null
  property var manifest: null

  readonly property string pluginId: (root.manifest && root.manifest.id) || "meshwa.tapzones"
  readonly property var svc: root.shell && typeof root.shell.serviceFor === "function" ? root.shell.serviceFor(root.pluginId) : null

  property bool opened: false
  property var cfg: null              // the config being edited
  property string savedJson: ""       // the config as last saved
  property string sendingJson: ""     // the config of a save on its way
  property string gesture: "single"
  property int selected: -1
  property var history: []            // earlier configs, for undo
  property var drag: null
  property string warn: ""
  property bool quitArmed: false
  property real now: Date.now() / 1000
  // Edits change cfg in place; bumping rev re-evaluates what depends on it.
  property int rev: 0

  readonly property var zoneColors: ["#e06b75", "#e6bf7a", "#99c278", "#61b0f0"]
  readonly property color good: "#8cd980"
  readonly property color mid: "#ebbf6b"
  readonly property color bad: "#f26161"
  readonly property real edgePx: Style.space(10)

  readonly property var sections: (root.rev, root.cfg ? root.cfg[root.gesture].sections : [])
  readonly property bool dirty: (root.rev, root.cfg !== null && JSON.stringify(root.cfg) !== root.savedJson)
  readonly property var estimate: (root.rev, root.svc && root.svc.estimate && root.cfg)
    ? Layout.quickEstimate(root.svc.estimate, Layout.layoutOf(root.sections)) : null

  function touch() { root.rev++ }
  function grade(a) { return a >= 0.85 ? root.good : a >= 0.7 ? root.mid : root.bad }
  function say(msg) { root.warn = msg; warnTimer.restart() }

  // ------------------------------------------------------------ lifecycle

  function open(payloadJson) {
    var st = root.svc && root.svc.online ? root.svc.state : null
    root.cfg = st ? Layout.clone(st.config) : null
    root.savedJson = root.cfg ? JSON.stringify(root.cfg) : ""
    root.gesture = "single"
    root.history = []
    root.quitArmed = false
    root.warn = ""
    root.select(-1)
    root.opened = true
    if (st) {
      root.svc.send({ cmd: "editing", on: true })
      if (!root.svc.estimate) root.svc.send({ cmd: "estimate" })
    }
    root.touch()
    keyCatcher.forceActiveFocus()
  }

  function close() {
    if (root.svc) root.svc.send({ cmd: "editing", on: false })
    root.opened = false
    root.drag = null
  }

  function dismiss() {
    root.close()
    if (root.shell && typeof root.shell.hide === "function") root.shell.hide(root.pluginId)
  }

  function toggle() {
    if (root.opened) root.dismiss()
    else root.open("{}")
  }

  Connections {
    target: root.svc
    function onEvent(msg) {
      if (!root.opened) return
      if (msg.event === "saved") {
        root.savedJson = root.sendingJson
        root.say("saved " + msg.path)
        root.touch()
      } else if (msg.event === "error") {
        root.say(msg.message)
      } else if (msg.event === "trained") {
        root.say(msg.ok ? "trained: the service uses it now" : "training failed: " + msg.message)
      }
    }
  }

  // ---------------------------------------------------------------- edits

  function remember() {
    var h = root.history.slice(-49)
    h.push({ cfg: JSON.stringify(root.cfg), gesture: root.gesture, selected: root.selected })
    root.history = h
    root.quitArmed = false
  }

  function undo() {
    if (!root.history.length) return
    var h = root.history.slice()
    var last = h.pop()
    root.history = h
    root.cfg = JSON.parse(last.cfg)
    root.gesture = last.gesture
    root.touch()
    root.select(last.selected < root.sections.length ? last.selected : -1)
  }

  function select(i) {
    root.selected = i
    if (i >= 0) {
      nameField.text = root.sections[i].name
      commandField.text = root.sections[i].command
    }
  }

  function edit(field, value) {
    if (root.selected < 0) return
    root.sections[root.selected][field] = value
    root.quitArmed = false
    root.touch()
  }

  function remove(i) {
    if (i < 0) return
    root.remember()
    root.cfg[root.gesture].sections.splice(i, 1)
    root.select(-1)
    root.touch()
  }

  function setGesture(g) {
    root.gesture = g
    root.select(-1)
    root.touch()
  }

  function stepPreset(step) {
    root.remember()
    root.cfg[root.gesture].sections = Layout.stepPreset(root.sections, step)
    root.select(-1)
    root.touch()
  }

  function toggleEnabled() {
    root.remember()
    root.cfg[root.gesture].enabled = !root.cfg[root.gesture].enabled
    root.touch()
  }

  function cycleFeedback() {
    root.remember()
    var f = Layout.FEEDBACK
    root.cfg.feedback = f[(f.indexOf(root.cfg.feedback) + 1) % f.length]
    root.touch()
  }

  function save() {
    for (var k = 0; k < Layout.GESTURES.length; k++) {
      var g = Layout.GESTURES[k], bad = Layout.problems(root.cfg, g)
      if (bad.length) {
        root.setGesture(g)
        root.say("not saved: " + Layout.TITLES[g] + ": " + bad[0])
        return
      }
    }
    root.sendingJson = JSON.stringify(root.cfg)
    if (!root.svc.send({ cmd: "save", config: root.cfg })) root.say("not saved: the service isn't running")
    root.quitArmed = false
  }

  function requestClose() {
    if (root.dirty && !root.quitArmed) {
      root.quitArmed = true
      root.say("unsaved changes: S saves, Esc again closes without saving")
    } else {
      root.dismiss()
    }
  }

  // ---------------------------------------------------------------- mouse

  // [section index, edges]: edges holds l / r / t / b on a border (resize),
  // nothing inside (move). The selected section wins where they touch.
  function hit(x, y) {
    var w = surface.width, h = surface.height, e = root.edgePx
    var order = root.selected >= 0 ? [root.selected] : []
    for (var j = 0; j < root.sections.length; j++) if (j !== root.selected) order.push(j)
    for (var k = 0; k < order.length; k++) {
      var i = order[k], r = root.sections[i].rect
      var x0 = r[0] * w, y0 = r[1] * h, x1 = (r[0] + r[2]) * w, y1 = (r[1] + r[3]) * h
      if (x < x0 - e || x > x1 + e || y < y0 - e || y > y1 + e) continue
      var edges = {}
      if (Math.abs(x - x0) < e) edges.l = true
      if (Math.abs(x - x1) < e) edges.r = true
      if (Math.abs(y - y0) < e) edges.t = true
      if (Math.abs(y - y1) < e) edges.b = true
      var any = Object.keys(edges).length > 0
      if (any || (x >= x0 && x <= x1 && y >= y0 && y <= y1)) return [i, edges, any]
    }
    return [-1, {}, false]
  }

  function dragBegin(x, y) {
    var hitAt = root.hit(x, y)
    root.remember()
    var start = [x / surface.width, y / surface.height]
    if (hitAt[0] < 0) {
      root.drag = { mode: "draw", start: start, px: [x, y], index: -1 }
      root.select(-1)
    } else {
      root.select(hitAt[0])
      root.drag = { mode: hitAt[2] ? "resize" : "move", edges: hitAt[1], index: hitAt[0], start: start,
                    orig: root.sections[hitAt[0]].rect.slice() }
    }
    keyCatcher.forceActiveFocus()
  }

  function dragUpdate(x, y) {
    var d = root.drag
    if (!d) return
    var fx = x / surface.width, fy = y / surface.height, s = root.sections, i = d.index
    var snap = function(v, axis) { return Layout.snap(v, axis, s, i) }
    if (d.mode === "draw") {
      if (i < 0) {
        if (Math.abs(x - d.px[0]) < 8 && Math.abs(y - d.px[1]) < 8) return
        s.push({ name: "Section " + (s.length + 1), rect: [0, 0, 0, 0], command: "" })
        i = d.index = s.length - 1
        root.select(i)
      }
      var ax = [snap(d.start[0], 0), snap(fx, 0)].sort(function(a, b) { return a - b })
      var ay = [snap(d.start[1], 1), snap(fy, 1)].sort(function(a, b) { return a - b })
      s[i].rect = [ax[0], ay[0], ax[1] - ax[0], ay[1] - ay[0]]
    } else if (d.mode === "move") {
      var w = d.orig[2], h = d.orig[3]
      var nx = Math.min(Math.max(d.orig[0] + fx - d.start[0], 0), 1 - w)
      var ny = Math.min(Math.max(d.orig[1] + fy - d.start[1], 0), 1 - h)
      // Snap whichever side of the section is closer to something.
      var sx = snap(nx, 0), ex = snap(nx + w, 0) - w
      var sy = snap(ny, 1), ey = snap(ny + h, 1) - h
      nx = Math.abs(sx - nx) <= Math.abs(ex - nx) ? sx : ex
      ny = Math.abs(sy - ny) <= Math.abs(ey - ny) ? sy : ey
      s[i].rect = [Math.min(Math.max(nx, 0), 1 - w), Math.min(Math.max(ny, 0), 1 - h), w, h]
    } else {
      var x0 = d.orig[0], y0 = d.orig[1], x1 = x0 + d.orig[2], y1 = y0 + d.orig[3], m = Layout.MIN_SIDE
      if (d.edges.l) x0 = Math.min(snap(fx, 0), x1 - m)
      if (d.edges.r) x1 = Math.max(snap(fx, 0), x0 + m)
      if (d.edges.t) y0 = Math.min(snap(fy, 1), y1 - m)
      if (d.edges.b) y1 = Math.max(snap(fy, 1), y0 + m)
      x0 = Math.max(x0, 0); y0 = Math.max(y0, 0)
      s[i].rect = [x0, y0, Math.min(x1, 1) - x0, Math.min(y1, 1) - y0]
    }
    root.touch()
  }

  function dragEnd() {
    var d = root.drag
    root.drag = null
    if (!d) return
    if (d.index < 0) {                 // a click on empty screen: nothing changed
      root.history = root.history.slice(0, -1)
      return
    }
    var bad = Layout.problems(root.cfg, root.gesture).filter(function(p) {
      return p.indexOf("overlap") >= 0 || p.indexOf("smaller") >= 0
    })
    if (bad.length) {
      root.undo()
      root.say(bad[0])
      return
    }
    if (d.mode !== "draw" && Layout.sameRect(d.orig, root.sections[d.index].rect))
      root.history = root.history.slice(0, -1)    // just selected it
    if (root.sections.length > root.cfg.max_sections)
      root.say(root.sections.length + " sections: more than the recommended " + root.cfg.max_sections
               + ". Raise max_sections in gestures.toml to save it")
    root.touch()
  }

  // --------------------------------------------------------------- status

  readonly property string stateLine: {
    root.rev; root.now
    var st = root.svc && root.svc.online ? root.svc.state : null
    if (!st || !root.cfg) return ""
    if (st.detected.indexOf(root.gesture) < 0)
      return "the service can't hear this gesture yet: the layout is saved for later"
    if (!root.cfg[root.gesture].enabled) return "off: E turns it on"
    if (st.training) return "training the section model… " + Math.round(root.now - st.training.since) + " s (about 8 min)"
    if (root.dirty) return "unsaved: S saves and trains"
    var m = st.model
    if (m.ok) return "trained: " + Layout.pct(m.accuracy) + " right across sessions; answers "
                     + Layout.pct(m.answered) + " of taps, " + Layout.pct(m.answered_accuracy) + " of those right"
    return "saved, but not trained for this layout: S trains it"
  }

  readonly property string estimateLine: {
    var st = root.svc && root.svc.online ? root.svc.state : null
    if (!st) return ""
    if (root.svc.estimate) return "rough estimates from " + root.svc.estimate.where.length
                                  + " recorded taps (usually a few points low; Save shows the real number)"
    if (st.estimate === "preparing") return "preparing estimates from the recorded taps (once, a few minutes)…"
    return "estimates: waiting for the service"
  }

  Timer { id: warnTimer; interval: 6000; onTriggered: root.warn = "" }
  Timer { interval: 1000; repeat: true; running: root.opened; onTriggered: root.now = Date.now() / 1000 }

  // -------------------------------------------------------------- surface

  readonly property var laptopScreen: Quickshell.screens.filter(function(s) { return /^eDP/.test(s.name) })[0] || Quickshell.screens[0]

  PanelWindow {
    id: panel
    visible: root.opened
    screen: root.laptopScreen
    color: Util.alpha(Commons.Color.background, 0.97)     // dead zones stay dark
    anchors { top: true; bottom: true; left: true; right: true }
    exclusionMode: ExclusionMode.Ignore
    WlrLayershell.namespace: "omarchy-tapzones-editor"
    WlrLayershell.layer: WlrLayer.Overlay
    WlrLayershell.keyboardFocus: WlrKeyboardFocus.Exclusive

    Item {
      id: surface
      anchors.fill: parent

      Item {
        id: keyCatcher
        anchors.fill: parent
        focus: true
        Keys.onPressed: function(event) {
          var k = event.key
          event.accepted = true
          if (k === Qt.Key_Escape) return root.requestClose()
          if (!root.cfg) return
          if (k >= Qt.Key_1 && k <= Qt.Key_5) root.setGesture(Layout.GESTURES[k - Qt.Key_1])
          else if (k === Qt.Key_P) root.stepPreset(event.modifiers & Qt.ShiftModifier ? -1 : 1)
          else if (k === Qt.Key_E) root.toggleEnabled()
          else if (k === Qt.Key_F) root.cycleFeedback()
          else if (k === Qt.Key_Delete || k === Qt.Key_Backspace) root.remove(root.selected)
          else if (k === Qt.Key_U) root.undo()
          else if (k === Qt.Key_S) root.save()
          else event.accepted = false
        }
      }

      MouseArea {
        anchors.fill: parent
        enabled: root.cfg !== null
        acceptedButtons: Qt.LeftButton | Qt.RightButton
        hoverEnabled: true
        cursorShape: {
          if (root.drag) return root.drag.mode === "move" ? Qt.ClosedHandCursor : Qt.CrossCursor
          var h = root.hit(mouseX, mouseY)
          if (h[0] < 0) return Qt.CrossCursor
          var e = h[1]
          if ((e.l && e.t) || (e.r && e.b)) return Qt.SizeFDiagCursor
          if ((e.r && e.t) || (e.l && e.b)) return Qt.SizeBDiagCursor
          if (e.l || e.r) return Qt.SizeHorCursor
          if (e.t || e.b) return Qt.SizeVerCursor
          return Qt.OpenHandCursor
        }
        onPressed: function(mouse) {
          if (mouse.button === Qt.RightButton) root.remove(root.hit(mouse.x, mouse.y)[0])
          else root.dragBegin(mouse.x, mouse.y)
        }
        onPositionChanged: function(mouse) { if (root.drag) root.dragUpdate(mouse.x, mouse.y) }
        onReleased: function(mouse) { if (mouse.button === Qt.LeftButton) root.dragEnd() }
      }

      // ------------------------------------------------------- sections

      Repeater {
        model: root.sections.length
        delegate: Rectangle {
          id: zone
          required property int index
          readonly property var s: (root.rev, root.sections[index])
          readonly property var est: root.estimate && index < root.estimate.per.length ? root.estimate.per[index] : null
          readonly property bool on: root.cfg[root.gesture].enabled
          readonly property color tint: root.zoneColors[index % root.zoneColors.length]
          readonly property bool isSelected: index === root.selected

          visible: s !== undefined
          x: s ? s.rect[0] * surface.width + 3 : 0
          y: s ? s.rect[1] * surface.height + 3 : 0
          width: s ? Math.max(0, s.rect[2] * surface.width - 6) : 0
          height: s ? Math.max(0, s.rect[3] * surface.height - 6) : 0
          radius: Style.space(10)
          color: Util.alpha(tint, on ? 0.14 : 0.05)
          border.color: isSelected ? Commons.Color.accent : Util.alpha(tint, 0.5)
          border.width: isSelected ? 3 : 1.5

          Column {
            anchors.centerIn: parent
            width: parent.width - Style.spacing.xl * 2
            spacing: Style.spacing.xs

            Text {
              width: parent.width
              horizontalAlignment: Text.AlignHCenter
              elide: Text.ElideRight
              text: zone.s ? zone.s.name : ""
              color: Util.alpha(Commons.Color.foreground, zone.on ? 0.95 : 0.4)
              font.family: Style.font.menuFamily
              font.pixelSize: Style.font.title
            }
            Text {
              width: parent.width
              horizontalAlignment: Text.AlignHCenter
              elide: Text.ElideMiddle
              text: zone.s && zone.s.command ? zone.s.command : "no command"
              color: Util.alpha(Commons.Color.foreground, zone.s && zone.s.command ? 0.55 : 0.3)
              font.family: Style.font.menuFamily
              font.pixelSize: Style.font.caption
            }
            Text {
              width: parent.width
              horizontalAlignment: Text.AlignHCenter
              visible: zone.est !== null
              text: !zone.est ? "" : isNaN(zone.est.acc) ? "only " + zone.est.n + " recorded taps here" : "≈ " + Layout.pct(zone.est.acc)
              color: !zone.est || isNaN(zone.est.acc) ? root.bad : root.grade(zone.est.acc)
              font.family: Style.font.menuFamily
              font.pixelSize: !zone.est || isNaN(zone.est.acc) ? Style.font.bodySmall : Style.font.display
            }
            Text {
              width: parent.width
              horizontalAlignment: Text.AlignHCenter
              visible: zone.est !== null && !isNaN(zone.est.acc)
              text: zone.est ? zone.est.n + " taps" : ""
              color: Util.alpha(Commons.Color.foreground, 0.4)
              font.family: Style.font.menuFamily
              font.pixelSize: Style.font.caption
            }
          }
        }
      }

      // --------------------------------------------------------- header

      Rectangle {
        id: header
        anchors.horizontalCenter: parent.horizontalCenter
        anchors.top: parent.top
        anchors.topMargin: Style.space(24)
        width: Math.min(parent.width - Style.space(32), headerColumn.implicitWidth + Style.spacing.xl * 4)
        height: headerColumn.implicitHeight + Style.spacing.xl * 2
        radius: Style.space(12)
        color: Commons.Color.menu.background
        border.color: Util.alpha(Commons.Color.menu.border, 0.25)

        MouseArea { anchors.fill: parent; acceptedButtons: Qt.AllButtons }   // not a place to draw

        Column {
          id: headerColumn
          anchors.centerIn: parent
          spacing: Style.spacing.md

          Row {
            spacing: Style.spacing.md
            visible: root.cfg !== null
            Repeater {
              model: Layout.GESTURES
              delegate: Chip {
                required property string modelData
                required property int index
                label: (index + 1) + " " + Layout.TITLES[modelData] + ((root.rev, root.cfg && root.cfg[modelData].enabled) ? "" : " (off)")
                current: root.gesture === modelData
                onClicked: root.setGesture(modelData)
              }
            }
          }

          Row {
            spacing: Style.spacing.md
            visible: root.cfg !== null
            Chip { label: "‹"; onClicked: root.stepPreset(-1) }
            Text {
              anchors.verticalCenter: parent.verticalCenter
              text: (root.rev, root.cfg) ? Layout.presetOf(root.sections) + " · " + root.sections.length + "/"
                    + root.cfg.max_sections + " sections" : ""
              color: Commons.Color.menu.text
              font.family: Style.font.menuFamily
              font.pixelSize: Style.font.body
            }
            Chip { label: "›"; onClicked: root.stepPreset(1) }
            Chip { label: (root.rev, root.cfg && root.cfg[root.gesture].enabled) ? "on" : "off"; onClicked: root.toggleEnabled() }
            Chip { label: "feedback: " + ((root.rev, root.cfg) ? root.cfg.feedback : ""); onClicked: root.cycleFeedback() }
            Chip { label: "undo"; onClicked: root.undo() }
            Chip { label: "save"; current: root.dirty; onClicked: root.save() }
            Text {
              anchors.verticalCenter: parent.verticalCenter
              visible: root.estimate !== null && !isNaN(root.estimate.overall)
              text: root.estimate ? "estimate ≈ " + Layout.pct(root.estimate.overall) : ""
              color: root.estimate ? root.grade(root.estimate.overall) : root.mid
              font.family: Style.font.menuFamily
              font.pixelSize: Style.font.body
            }
          }

          Text {
            text: root.cfg === null
              ? "The tapzones service isn't running, so there's nothing to edit.\nStart it: systemctl --user start tapzones   (Esc closes)"
              : root.stateLine
            color: root.cfg === null ? root.bad : root.stateLine.indexOf("trained:") === 0 ? root.good : root.mid
            font.family: Style.font.menuFamily
            font.pixelSize: Style.font.bodySmall
          }

          Text {
            text: root.warn || root.estimateLine
            visible: text !== ""
            color: root.warn && /not |fail|more than|overlap|smaller/.test(root.warn) ? root.bad
                 : Util.alpha(Commons.Color.menu.text, root.warn ? 0.95 : 0.5)
            font.family: Style.font.menuFamily
            font.pixelSize: Style.font.caption
          }
        }
      }

      // --------------------------------------------- name, command, picks

      Rectangle {
        id: details
        visible: root.selected >= 0
        anchors.horizontalCenter: parent.horizontalCenter
        anchors.bottom: hints.top
        anchors.bottomMargin: Style.space(14)
        width: Math.min(parent.width - Style.space(32), Style.space(820))
        height: detailsColumn.implicitHeight + Style.spacing.xl * 2
        radius: Style.space(12)
        color: Commons.Color.menu.background
        border.color: Util.alpha(Commons.Color.menu.border, 0.25)

        MouseArea { anchors.fill: parent; acceptedButtons: Qt.AllButtons }   // not a place to draw

        Column {
          id: detailsColumn
          anchors.fill: parent
          anchors.margins: Style.spacing.xl
          spacing: Style.spacing.md

          Row {
            width: parent.width
            spacing: Style.spacing.lg
            TextField {
              id: nameField
              width: Style.space(200)
              placeholderText: "name"
              font.family: Style.font.menuFamily
              font.pixelSize: Style.font.body
              onTextEdited: root.edit("name", text)
              Keys.onEscapePressed: keyCatcher.forceActiveFocus()
              onAccepted: keyCatcher.forceActiveFocus()
            }
            TextField {
              id: commandField
              width: parent.width - nameField.width - parent.spacing
              placeholderText: "shell command, e.g. omarchy launch browser"
              font.family: Style.font.menuFamily
              font.pixelSize: Style.font.body
              onTextEdited: root.edit("command", text)
              Keys.onEscapePressed: keyCatcher.forceActiveFocus()
              onAccepted: keyCatcher.forceActiveFocus()
            }
          }

          Flow {
            width: parent.width
            spacing: Style.spacing.sm
            Repeater {
              model: Layout.OMARCHY_ACTIONS
              delegate: Chip {
                required property var modelData
                label: modelData.label
                small: true
                current: (root.rev, root.selected >= 0 && root.sections[root.selected] && root.sections[root.selected].command === modelData.command)
                onClicked: {
                  root.remember()
                  commandField.text = modelData.command
                  root.edit("command", modelData.command)
                }
              }
            }
          }
        }
      }

      Text {
        id: hints
        anchors.horizontalCenter: parent.horizontalCenter
        anchors.bottom: parent.bottom
        anchors.bottomMargin: Style.space(14)
        text: "drag: draw / move / resize · right-click: delete · 1–5 gesture · P preset · E on/off · F feedback · U undo · S save · Esc close"
        color: Util.alpha(Commons.Color.foreground, 0.7)
        font.family: Style.font.menuFamily
        font.pixelSize: Style.font.caption
      }
    }
  }

  // A small clickable label in the theme: tabs, header buttons, picks.
  component Chip: Rectangle {
    id: chip
    property string label: ""
    property bool current: false
    property bool small: false
    signal clicked()

    implicitWidth: chipText.implicitWidth + Style.spacing.lg * 2
    implicitHeight: chipText.implicitHeight + (small ? Style.spacing.xs : Style.spacing.sm) * 2
    radius: Style.space(6)
    color: current ? Util.alpha(Commons.Color.accent, 0.25)
         : chipMouse.containsMouse ? Commons.Color.menu.selectedBackground : "transparent"
    border.color: Util.alpha(Commons.Color.menu.text, small ? 0.2 : 0)

    Text {
      id: chipText
      anchors.centerIn: parent
      text: chip.label
      color: chip.current ? Commons.Color.menu.selectedText : Util.alpha(Commons.Color.menu.text, 0.8)
      font.family: Style.font.menuFamily
      font.pixelSize: chip.small ? Style.font.caption : Style.font.bodySmall
    }
    MouseArea {
      id: chipMouse
      anchors.fill: parent
      hoverEnabled: true
      cursorShape: Qt.PointingHandCursor
      onClicked: chip.clicked()
    }
  }
}
