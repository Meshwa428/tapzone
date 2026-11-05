import QtQuick
import Quickshell
import Quickshell.Wayland
import qs.Commons
import qs.Commons as Commons
import qs.Ui
import "Layout.js" as Layout

// The layout editor (docs/Gestures/Sections.md), in the shell: the same
// editor as `tapzones layout`, drawn in the theme. Pick a gesture, how many
// sections, and one of the preset layouts of that many (best estimate first),
// adjust its sections, name them and give each a command. New sections
// can't be drawn: layouts come only from the presets. Every
// section shows how well the mic tells it apart, estimated live from the
// recorded taps. Save hands the config to the service, which writes
// gestures.toml and trains the section model.
//
// Mouse: drag inside a section moves it · drag its edge resizes it ·
//        right-click deletes it
// Keys:  1–5 gesture · N / Shift+N sections · P / Shift+P layout · E on/off · F feedback ·
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

  readonly property var zoneColors: ["#e06b75", "#e6bf7a", "#99c278", "#61b0f0", "#c49be8", "#6fd1c4"]
  readonly property color good: "#8cd980"
  readonly property color mid: "#ebbf6b"
  readonly property color bad: "#f26161"
  readonly property real edgePx: Style.space(10)

  readonly property var sections: (root.rev, root.cfg ? root.cfg[root.gesture].sections : [])
  readonly property bool dirty: (root.rev, root.cfg !== null && JSON.stringify(root.cfg) !== root.savedJson)
  readonly property var estimate: (root.rev, root.svc && root.svc.estimate && root.cfg)
    ? Layout.quickEstimate(root.svc.estimate, Layout.layoutOf(root.sections)) : null
  readonly property int count: (root.rev, root.cfg ? root.sections.length : 0)
  readonly property string preset: (root.rev, root.cfg ? Layout.presetOf(root.sections) : "")
  // The layouts with as many sections as now, the usual one first, then the simplest.
  readonly property var choices: Layout.choicesFor(root.svc ? root.svc.estimate : null, root.count)

  function touch() { root.rev++ }
  function grade(a) { return a >= 0.85 ? root.good : a >= 0.7 ? root.mid : root.bad }
  function say(msg) { root.warn = msg; warnTimer.restart() }

  // ------------------------------------------------------------ lifecycle

  function open(payloadJson) {
    var st = root.svc && root.svc.online ? root.svc.state : null
    root.cfg = st ? Layout.clone(st.config) : null
    root.savedJson = root.cfg ? JSON.stringify(root.cfg) : ""
    root.serviceConfig = st ? root.canon(st.config) : ""
    root.sendingJson = ""
    root.drag = null
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

  // The config as JSON with sorted keys: the service may send the same config
  // back with its keys in another order.
  function canon(v) {
    return JSON.stringify(v, function(k, x) {
      if (!x || typeof x !== "object" || Array.isArray(x)) return x
      var o = {}
      Object.keys(x).sort().forEach(function(key) { o[key] = x[key] })
      return o
    })
  }

  property string serviceConfig: ""    // canon() of the service's config as last seen

  Connections {
    target: root.svc
    // The config changed somewhere else while the editor is open (the CLI,
    // the GTK editor, a socket save): take it, or at least don't save over
    // it without saying so.
    function onStateChanged() {
      var st = root.svc.state
      if (!root.opened || !st || !st.config) return
      var now = root.canon(st.config)
      if (now === root.serviceConfig) return
      root.serviceConfig = now
      if (root.sendingJson && now === root.canon(JSON.parse(root.sendingJson))) return   // our own save
      if (!root.cfg || !root.dirty) {
        root.cfg = Layout.clone(st.config)
        root.savedJson = JSON.stringify(root.cfg)
        root.history = []
        root.select(-1)
        root.touch()
        root.say("the config changed outside the editor: reloaded")
      } else {
        root.say("the config changed outside the editor: Save will replace that change with yours")
      }
    }
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

  function choose(name) {
    if (name === root.preset) return
    root.remember()
    root.cfg[root.gesture].sections = Layout.applyPreset(root.sections, name)
    root.select(-1)
    root.touch()
  }

  // The next (1) or previous (-1) layout with as many sections.
  function stepPreset(step) {
    var names = root.choices.map(function(c) { return c.name })
    var cur = names.indexOf(root.preset)
    root.choose(names[cur < 0 ? 0 : (cur + step + names.length) % names.length])
  }

  // n sections: the usual layout of n.
  function setCount(n) {
    if (n !== root.count) root.choose(Layout.BY_COUNT[n][0])
  }

  function stepCount(step) {
    var c = Layout.COUNTS, k = c.indexOf(root.count)
    root.setCount(c[k < 0 ? 0 : (k + step + c.length) % c.length])
  }

  function toggleEnabled() {
    root.remember()
    root.cfg[root.gesture].enabled = !root.cfg[root.gesture].enabled
    root.touch()
  }

  function cycleFeedback() {
    var f = Layout.FEEDBACK
    root.setFeedback(f[(f.indexOf(root.cfg.feedback) + 1) % f.length])
  }

  function setFeedback(v) {
    if (!root.cfg || root.cfg.feedback === v) return
    root.remember()
    root.cfg.feedback = v
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
    keyCatcher.forceActiveFocus()
    if (hitAt[0] < 0) {                // empty screen: sections come from the presets, not drawn
      root.select(-1)
      return
    }
    root.remember()
    root.select(hitAt[0])
    root.drag = { mode: hitAt[2] ? "resize" : "move", edges: hitAt[1], index: hitAt[0],
                  start: [x / surface.width, y / surface.height], orig: root.sections[hitAt[0]].rect.slice() }
  }

  function dragUpdate(x, y) {
    var d = root.drag
    if (!d) return
    var fx = x / surface.width, fy = y / surface.height, s = root.sections, i = d.index
    var snap = function(v, axis) { return Layout.snap(v, axis, s, i) }
    if (d.mode === "move") {
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
    var bad = Layout.problems(root.cfg, root.gesture).filter(function(p) {
      return p.indexOf("overlap") >= 0 || p.indexOf("smaller") >= 0
    })
    if (bad.length) {
      root.undo()
      root.say(bad[0])
      return
    }
    if (Layout.sameRect(d.orig, root.sections[d.index].rect))
      root.history = root.history.slice(0, -1)    // just selected it
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

  // stateLine in a word or two, for the header's caption.
  readonly property string stateWord: {
    root.rev
    var st = root.svc && root.svc.online ? root.svc.state : null
    if (!st || !root.cfg) return ""
    if (st.detected.indexOf(root.gesture) < 0) return "not heard yet"
    if (!root.cfg[root.gesture].enabled) return "off"
    if (st.training) return "training"
    if (root.dirty) return "unsaved"
    return st.model.ok ? "trained" : "not trained"
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
          else if (k === Qt.Key_N) root.stepCount(event.modifiers & Qt.ShiftModifier ? -1 : 1)
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
          if (h[0] < 0) return Qt.ArrowCursor
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
        // Only while the button is down: a drag whose release never came must
        // not keep moving the section with the bare pointer.
        onPositionChanged: function(mouse) {
          if (!root.drag) return
          if (mouse.buttons & Qt.LeftButton) root.dragUpdate(mouse.x, mouse.y)
          else root.dragEnd()
        }
        onReleased: function(mouse) { if (mouse.button === Qt.LeftButton) root.dragEnd() }
        onCanceled: root.dragEnd()     // Hyprland can take the pointer mid-drag
      }

      // ------------------------------------------------------- sections

      // The tiles get a layer of their own, so a section raised over the
      // others (z) still stays under the header and the details panel.
      Item {
        anchors.fill: parent

        Repeater {
          model: root.sections.length
          delegate: Rectangle {
            id: zone
            required property int index
            // Edits change the section in place, and a var property that gets the
            // same object back doesn't notify, so copy it: the tile then follows
            // a drag as it happens.
            readonly property var s: (root.rev, root.sections[index] ? Object.assign({}, root.sections[index]) : undefined)
            readonly property var est: root.estimate && index < root.estimate.per.length ? root.estimate.per[index] : null
            readonly property bool on: root.cfg[root.gesture].enabled
            readonly property color tint: root.zoneColors[index % root.zoneColors.length]
            readonly property bool isSelected: index === root.selected
  
            readonly property bool over: s !== undefined && s.over === true
  
            visible: s !== undefined
            // A section that sits over the others is drawn on top, solid, so
            // the ones under it read as the shapes they really are.
            z: over ? 1 : 0
            x: s ? s.rect[0] * surface.width + 3 : 0
            y: s ? s.rect[1] * surface.height + 3 : 0
            width: s ? Math.max(0, s.rect[2] * surface.width - 6) : 0
            height: s ? Math.max(0, s.rect[3] * surface.height - 6) : 0
            radius: Style.cornerRadius
            color: over ? Qt.tint(panel.color, Util.alpha(tint, on ? 0.14 : 0.05)) : Util.alpha(tint, on ? 0.14 : 0.05)
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
      }

      // --------------------------------------------------------- header

      // Framed like the shell's own popups. Top to bottom, most important
      // first: what you're editing and how well it will work, then which
      // gesture, then its settings, then status and the actions.
      BorderSurface {
        id: header
        readonly property color fg: Commons.Color.popups.text
        readonly property real pad: Style.spacing.popupPadding

        anchors.horizontalCenter: parent.horizontalCenter
        anchors.top: parent.top
        anchors.topMargin: Style.space(24)
        width: Math.min(parent.width - Style.space(32), headerColumn.width + pad * 2)
        height: headerColumn.implicitHeight + pad * 2
        radius: Style.cornerRadius
        color: Commons.Color.popups.background
        borderSpec: Border.localOrSurfaceSpec("popups", "border", Commons.Color.popups.border,
                                              Commons.Color.popups.border, Math.max(1, Style.space(2)))

        MouseArea { anchors.fill: parent; acceptedButtons: Qt.AllButtons }   // not a place to draw

        Column {
          id: headerColumn
          anchors.centerIn: parent
          width: Math.max(tabs.implicitWidth, controls.implicitWidth, Style.space(560))
          spacing: Style.space(12)

          Text {
            visible: root.cfg === null
            width: parent.width
            wrapMode: Text.Wrap
            text: "The tapzones service isn't running, so there's nothing to edit.\nStart it: systemctl --user start tapzones   (Esc closes)"
            color: root.bad
            font.family: Style.font.family
            font.pixelSize: Style.font.body
          }

          // what you're editing, and how well the mic will tell it apart
          Item {
            visible: root.cfg !== null
            width: parent.width
            height: Math.max(titleColumn.implicitHeight, estimateColumn.implicitHeight)

            Column {
              id: titleColumn
              anchors.left: parent.left
              anchors.verticalCenter: parent.verticalCenter
              spacing: Style.spacing.xs
              Text {
                text: "Tap layout"
                color: header.fg
                font.family: Style.font.family
                font.pixelSize: Style.font.heading
                font.bold: true
              }
              Text {
                text: ((root.rev, root.cfg) ? Layout.TITLES[root.gesture] : "")
                      + (root.stateWord ? "  ·  " + root.stateWord : "")
                font.capitalization: Font.AllUppercase
                font.letterSpacing: 1.2
                color: root.stateWord === "trained" ? root.good
                     : root.stateWord === "" ? Util.alpha(header.fg, 0.55) : root.mid
                font.family: Style.font.family
                font.pixelSize: Style.font.caption
              }
            }

            Column {
              id: estimateColumn
              readonly property bool known: root.estimate !== null && !isNaN(root.estimate.overall)
              anchors.right: parent.right
              anchors.verticalCenter: parent.verticalCenter
              Text {
                anchors.right: parent.right
                text: estimateColumn.known ? "≈ " + Layout.pct(root.estimate.overall) : "—"
                color: estimateColumn.known ? root.grade(root.estimate.overall) : Util.alpha(header.fg, 0.4)
                font.family: Style.font.family
                font.pixelSize: Style.font.display
                font.bold: true
              }
              Text {
                anchors.right: parent.right
                text: "estimated accuracy"
                font.capitalization: Font.AllUppercase
                font.letterSpacing: 1.2
                color: Util.alpha(header.fg, 0.55)
                font.family: Style.font.family
                font.pixelSize: Style.font.caption
              }
            }
          }

          PanelSeparator { visible: root.cfg !== null; width: parent.width }

          PanelSectionHeader { visible: root.cfg !== null; text: "GESTURE"; foreground: header.fg }

          Row {
            id: tabs
            visible: root.cfg !== null
            spacing: Style.spacing.controlGap
            Repeater {
              model: Layout.GESTURES
              delegate: Button {
                required property string modelData
                required property int index
                readonly property bool on: (root.rev, root.cfg !== null && root.cfg[modelData].enabled)
                text: (index + 1) + "   " + Layout.TITLES[modelData]
                bordered: true
                selected: root.gesture === modelData
                foreground: on ? header.fg : Util.alpha(header.fg, 0.45)
                fontSize: Style.font.bodySmall
                tooltipText: on ? "" : "off"
                onClicked: root.setGesture(modelData)
              }
            }
          }

          PanelSeparator { visible: root.cfg !== null; width: parent.width }

          // how many sections, then which layout of that many
          Item {
            visible: root.cfg !== null
            width: parent.width
            height: countRow.implicitHeight

            Row {
              id: countRow
              spacing: Style.spacing.lg
              PanelSectionHeader { anchors.verticalCenter: parent.verticalCenter; text: "SECTIONS"; foreground: header.fg }
              ButtonGroup {
                options: Layout.COUNTS.map(String)
                value: String(root.count)
                focusable: false
                foreground: header.fg
                background: "transparent"     // the default is opaque: hover would fade through a bright grey
                fontSize: Style.font.bodySmall
                onChanged: v => root.setCount(Number(v))
              }
            }

            Text {
              readonly property bool over: (root.rev, root.cfg !== null && root.count > root.cfg.max_sections)
              anchors.right: parent.right
              anchors.verticalCenter: parent.verticalCenter
              text: (root.rev, root.cfg) ? root.choices.length + (root.choices.length === 1 ? " layout" : " layouts")

                    + (over ? " · over " : " · up to ") + root.cfg.max_sections + " sections recommended" : ""
              color: over ? root.mid : Util.alpha(header.fg, 0.55)
              font.family: Style.font.family
              font.pixelSize: Style.font.caption
            }
          }

          // The layouts with that many sections: the chosen one big in the
          // middle, its neighbours peeking in at the sides.
          Item {
            id: carousel
            visible: root.cfg !== null
            width: parent.width
            height: slides.height
            readonly property real aspect: root.laptopScreen ? root.laptopScreen.height / root.laptopScreen.width : 0.625
            readonly property real slideWidth: Math.round(slides.width * 0.5)   // the neighbours show in part
            readonly property int current: {
              root.preset
              for (var i = 0; i < root.choices.length; i++) if (root.choices[i].name === root.preset) return i
              return -1
            }
            function sync() { if (current >= 0 && slides.currentIndex !== current) slides.currentIndex = current }
            onCurrentChanged: Qt.callLater(sync)

            Button {
              id: prevButton
              anchors.left: parent.left
              y: (carousel.slideWidth * carousel.aspect - height) / 2
              text: "‹"
              bordered: true
              enabled: root.choices.length > 1
              tooltipText: "previous layout (Shift+P)"
              onClicked: root.stepPreset(-1)
            }
            Button {
              id: nextButton
              anchors.right: parent.right
              y: prevButton.y
              text: "›"
              bordered: true
              enabled: root.choices.length > 1
              tooltipText: "next layout (P)"
              onClicked: root.stepPreset(1)
            }

            ListView {
              id: slides
              anchors.left: prevButton.right
              anchors.right: nextButton.left
              anchors.leftMargin: Style.spacing.lg
              anchors.rightMargin: Style.spacing.lg
              height: carousel.slideWidth * carousel.aspect + Style.spacing.md + slideCaption.height * 2
              orientation: ListView.Horizontal
              spacing: Style.spacing.xl
              clip: true
              model: root.choices
              snapMode: ListView.SnapOneItem
              highlightRangeMode: ListView.StrictlyEnforceRange
              preferredHighlightBegin: (width - carousel.slideWidth) / 2
              preferredHighlightEnd: (width + carousel.slideWidth) / 2
              highlightMoveDuration: 160
              onModelChanged: Qt.callLater(carousel.sync)
              // Dragged or flicked to another layout: use it.
              onMovementEnded: if (currentIndex >= 0 && currentIndex < root.choices.length)
                                 root.choose(root.choices[currentIndex].name)

              Text { id: slideCaption; visible: false; text: "X"; font.pixelSize: Style.font.body }

              delegate: Item {
                id: slide
                required property var modelData
                required property int index
                readonly property bool on: index === carousel.current
                width: carousel.slideWidth
                height: slides.height
                opacity: on ? 1 : 0.4
                scale: on ? 1 : 0.86
                Behavior on opacity { NumberAnimation { duration: 140 } }
                Behavior on scale { NumberAnimation { duration: 140 } }

                Rectangle {
                  id: frame
                  width: parent.width
                  height: Math.round(width * carousel.aspect)
                  radius: Style.cornerRadius
                  color: "transparent"
                  border.width: slide.on ? Math.max(2, Style.space(2)) : 1
                  border.color: slide.on ? Commons.Color.accent : Util.alpha(header.fg, 0.35)

                  Repeater {
                    model: Layout.PRESETS[slide.modelData.name]
                    delegate: Rectangle {
                      required property var modelData
                      required property int index
                      readonly property real inset: Math.max(2, Style.space(4))
                      x: inset + modelData[0] * (frame.width - 2 * inset) + 1
                      y: inset + modelData[1] * (frame.height - 2 * inset) + 1
                      width: modelData[2] * (frame.width - 2 * inset) - 2
                      height: modelData[3] * (frame.height - 2 * inset) - 2
                      readonly property bool over: index < Layout.PRESET_OVER[slide.modelData.name]
                      readonly property color tint: Util.alpha(root.zoneColors[index % root.zoneColors.length], 0.6)
                      z: over ? 1 : 0
                      radius: Style.cornerRadius
                      color: over ? Qt.tint(Commons.Color.popups.background, tint) : tint
                      border.width: over ? Math.max(2, Style.space(3)) : 0      // a gap round it, as drawn
                      border.color: Commons.Color.popups.background
                    }
                  }
                }

                Column {
                  visible: slide.on                     // a neighbour's would be cut off
                  anchors.top: frame.bottom
                  anchors.topMargin: Style.spacing.md
                  anchors.horizontalCenter: parent.horizontalCenter
                  Text {
                    anchors.horizontalCenter: parent.horizontalCenter
                    text: slide.modelData.name
                    color: header.fg
                    font.family: Style.font.family
                    font.pixelSize: Style.font.body
                    font.bold: true
                  }
                  Text {
                    anchors.horizontalCenter: parent.horizontalCenter
                    text: (slide.modelData.acc >= 0 ? "≈ " + Layout.pct(slide.modelData.acc) + "  ·  " : "")
                          + (slide.index + 1) + " of " + root.choices.length
                    color: slide.modelData.acc >= 0 ? root.grade(slide.modelData.acc) : Util.alpha(header.fg, 0.55)
                    font.family: Style.font.family
                    font.pixelSize: Style.font.caption
                  }
                }

                HoverHandler { cursorShape: slide.on ? Qt.ArrowCursor : Qt.PointingHandCursor }
                TapHandler { onTapped: root.choose(slide.modelData.name) }
              }
            }
          }

          PanelSeparator { visible: root.cfg !== null; width: parent.width }

          // the chosen gesture's settings
          Row {
            id: controls
            visible: root.cfg !== null
            spacing: Style.space(28)

            Column {
              spacing: Style.spacing.lg
              PanelSectionHeader { text: "ENABLED"; foreground: header.fg }
              ToggleSwitch {
                checked: (root.rev, root.cfg !== null && root.cfg[root.gesture].enabled)
                foreground: header.fg
                onToggled: root.toggleEnabled()
              }
            }

            Column {
              spacing: Style.spacing.lg
              PanelSectionHeader { text: "FEEDBACK"; foreground: header.fg }
              ButtonGroup {
                options: Layout.FEEDBACK
                value: (root.rev, root.cfg) ? root.cfg.feedback : ""
                focusable: false
                foreground: header.fg
                background: "transparent"
                fontSize: Style.font.bodySmall
                onChanged: v => root.setFeedback(v)
              }
            }
          }

          PanelSeparator { visible: root.cfg !== null; width: parent.width }

          // what happens now, and the actions
          Item {
            visible: root.cfg !== null
            width: parent.width
            height: Math.max(statusColumn.implicitHeight, actions.implicitHeight)

            Column {
              id: statusColumn
              anchors.left: parent.left
              anchors.verticalCenter: parent.verticalCenter
              width: parent.width - actions.width - Style.spacing.xl
              spacing: Style.spacing.xs
              Text {
                width: parent.width
                wrapMode: Text.Wrap
                text: root.stateLine
                visible: text !== ""
                color: root.stateWord === "trained" ? root.good : root.mid
                font.family: Style.font.family
                font.pixelSize: Style.font.bodySmall
              }
              Text {
                width: parent.width
                wrapMode: Text.Wrap
                text: root.warn || root.estimateLine
                visible: text !== ""
                color: root.warn && /not |fail|more than|overlap|smaller/.test(root.warn) ? root.bad
                     : Util.alpha(header.fg, root.warn ? 0.95 : 0.5)
                font.family: Style.font.family
                font.pixelSize: Style.font.caption
              }
            }

            Row {
              id: actions
              anchors.right: parent.right
              anchors.verticalCenter: parent.verticalCenter
              spacing: Style.spacing.controlGap
              Button { text: "Undo"; foreground: header.fg; tooltipText: "U"; onClicked: root.undo() }
              Button {
                text: root.dirty ? "Save and train" : "Saved"
                bordered: true
                selected: root.dirty
                foreground: header.fg
                tooltipText: "S"
                onClicked: root.save()
              }
            }
          }
        }
      }

      // --------------------------------------------- name, command, picks

      BorderSurface {
        id: details
        readonly property real pad: Style.spacing.popupPadding

        visible: root.selected >= 0
        anchors.horizontalCenter: parent.horizontalCenter
        anchors.bottom: hints.top
        anchors.bottomMargin: Style.space(14)
        width: Math.min(parent.width - Style.space(32), Style.space(820))
        height: detailsColumn.implicitHeight + pad * 2
        radius: Style.cornerRadius
        color: Commons.Color.popups.background
        borderSpec: Border.localOrSurfaceSpec("popups", "border", Commons.Color.popups.border,
                                              Commons.Color.popups.border, Math.max(1, Style.space(2)))

        MouseArea { anchors.fill: parent; acceptedButtons: Qt.AllButtons }   // not a place to draw

        Column {
          id: detailsColumn
          anchors.fill: parent
          anchors.margins: details.pad
          spacing: Style.space(12)

          Row {
            width: parent.width
            spacing: Style.spacing.xl
            Column {
              spacing: Style.spacing.lg
              PanelSectionHeader { text: "NAME"; foreground: header.fg }
              TextField {
                id: nameField
                width: Style.space(200)
                placeholderText: "name"
                foreground: header.fg
                onTextEdited: root.edit("name", text)
                Keys.onEscapePressed: keyCatcher.forceActiveFocus()
                onAccepted: keyCatcher.forceActiveFocus()
              }
            }
            Column {
              spacing: Style.spacing.lg
              PanelSectionHeader { text: "COMMAND"; foreground: header.fg }
              TextField {
                id: commandField
                width: detailsColumn.width - nameField.width - Style.spacing.xl
                placeholderText: "shell command, e.g. omarchy launch browser"
                foreground: header.fg
                onTextEdited: root.edit("command", text)
                Keys.onEscapePressed: keyCatcher.forceActiveFocus()
                onAccepted: keyCatcher.forceActiveFocus()
              }
            }
          }

          PanelSeparator { width: parent.width }

          PanelSectionHeader { text: "OMARCHY ACTIONS"; foreground: header.fg }

          Flow {
            width: parent.width
            spacing: Style.spacing.controlGap
            Repeater {
              model: Layout.OMARCHY_ACTIONS
              delegate: Button {
                required property var modelData
                text: modelData.label
                bordered: true
                foreground: header.fg
                fontSize: Style.font.bodySmall
                selected: (root.rev, root.selected >= 0 && root.sections[root.selected] && root.sections[root.selected].command === modelData.command)
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
        text: "drag: move / resize · right-click: delete · 1–5 gesture · N sections · P layout · E on/off · F feedback · U undo · S save · Esc close"
        color: Util.alpha(Commons.Color.foreground, 0.7)
        font.family: Style.font.menuFamily
        font.pixelSize: Style.font.caption
      }
    }
  }
}
