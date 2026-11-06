import QtQuick
import Quickshell
import Quickshell.Wayland
import qs.Commons
import qs.Commons as Commons
import qs.Ui
import "Layout.js" as Layout
import "Sfx.js" as Sfx

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
  readonly property var savedCfg: root.savedJson ? JSON.parse(root.savedJson) : null

  // What a gesture's model depends on: whether it's on and where its
  // sections are. Names, commands and feedback don't need training.
  function shapeOf(c, g) {
    return c && c[g] ? JSON.stringify([c[g].enabled, c[g].sections.map(s => [s.rect, s.over])]) : ""
  }
  function reshaped(g) { root.rev; return root.cfg !== null && root.shapeOf(root.cfg, g) !== root.shapeOf(root.savedCfg, g) }

  // Saving trains (engine.save) when single taps are on and their saved
  // layout has no model: the shape changed, or it was never trained.
  readonly property bool saveTrains: {
    root.rev
    var st = root.svc && root.svc.online ? root.svc.state : null
    return root.cfg !== null && root.cfg.single.enabled && (root.reshaped("single") || !(st && st.model.ok))
  }
  readonly property var estimate: (root.rev, root.svc && root.svc.estimate && root.cfg)
    ? Layout.quickEstimate(root.svc.estimate, Layout.layoutOf(root.sections)) : null
  readonly property int count: (root.rev, root.cfg ? root.sections.length : 0)
  readonly property string preset: (root.rev, root.cfg ? Layout.presetOf(root.sections) : "")
  // Browsing is only looking: the section count and layout on show in the
  // carousel, until "Use this layout" puts it in place of the one in use.
  property int browseCount: 0
  property int browse: 0
  // The layouts with browseCount sections, the usual one first, then the simplest.
  readonly property var choices: Layout.choicesFor(root.svc ? root.svc.estimate : null, root.browseCount)
  // Where the layout in use sits among them, or -1 (another count, or moved by hand).
  readonly property int inUse: {
    root.preset
    if (root.browseCount !== root.count) return -1
    for (var i = 0; i < root.choices.length; i++) if (root.choices[i].name === root.preset) return i
    return -1
  }
  property bool showDetails: false

  function touch() { root.rev++ }
  // The buttons' sounds follow the Sound switch as it is here, saved or not;
  // on close, the saved setting again.
  onRevChanged: Sfx.on = root.cfg === null || root.cfg.sound !== false
  onCfgChanged: Sfx.on = root.cfg === null || root.cfg.sound !== false
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
    root.showDetails = false
    root.select(-1)
    root.lookAtInUse()
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
    var st = root.svc && root.svc.online ? root.svc.state : null
    Sfx.on = !st || st.config.sound !== false
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
        root.lookAtInUse()
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
    root.lookAtInUse()
  }

  function select(i) {
    root.selected = i
    if (i >= 0) commandField.text = root.sections[i].command
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
    root.rename()
    root.select(-1)
    root.touch()
  }

  function setGesture(g) {
    root.gesture = g
    root.select(-1)
    root.touch()
    root.lookAtInUse()
  }

  // Show the layout in use in the carousel (the usual one of its count when
  // it was moved by hand and matches no preset).
  function lookAtInUse() {
    root.browseCount = Math.max(1, Math.min(root.count, Layout.COUNTS[Layout.COUNTS.length - 1]))
    root.browse = Math.max(0, root.inUse)
  }

  // Put preset `name` in place of the layout in use.
  function choose(name) {
    if (name === root.preset) return
    root.remember()
    root.cfg[root.gesture].sections = Layout.applyPreset(root.sections, name)
    root.select(-1)
    root.touch()
    root.say("using “" + name + "”: Save to train it")
  }

  function useBrowsed() {
    if (root.browse >= 0 && root.browse < root.choices.length) root.choose(root.choices[root.browse].name)
  }

  // Look at the next (1) or previous (-1) layout with as many sections.
  function stepPreset(step) {
    var n = root.choices.length
    if (n) root.browse = (root.browse + step + n) % n
  }

  // Look at the layouts with n sections, the usual one first.
  function setCount(n) {
    if (n === root.browseCount) return
    root.browseCount = n
    root.browse = n === root.count && root.inUse >= 0 ? root.inUse : 0
  }

  function stepCount(step) {
    var c = Layout.COUNTS, k = c.indexOf(root.browseCount)
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

  function toggleSound() {
    if (!root.cfg) return
    root.remember()
    root.cfg.sound = root.cfg.sound === false
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
    root.rename()
    root.touch()
  }

  // Sections are named by where they are ("Top left"), never by hand: after
  // a move, a resize or a delete the names follow the new places.
  function rename() {
    var s = root.cfg[root.gesture].sections
    var names = Layout.sectionNames(s.map(x => x.rect))
    for (var i = 0; i < s.length; i++) s[i].name = names[i]
  }

  // --------------------------------------------------------------- status

  function heard(g) {
    var st = root.svc && root.svc.online ? root.svc.state : null
    return st !== null && st.detected.indexOf(g) >= 0
  }

  // A gesture's state in a word: what its tab and the header say.
  function gestureState(g) {
    root.rev
    var st = root.svc && root.svc.online ? root.svc.state : null
    if (!st || !root.cfg) return ""
    if (!root.heard(g)) return "not yet"
    if (!root.cfg[g].enabled) return "off"
    if (st.training && st.training.job === g) return "training"
    if (root.reshaped(g)) return "unsaved"     // a new command alone keeps the model
    return st.model.ok ? "trained" : "not trained"
  }

  readonly property string stateWord: root.gestureState(root.gesture)

  // The one accuracy the header shows: measured when the saved layout is
  // trained, the rough estimate otherwise.
  readonly property var headline: {
    root.rev
    var st = root.svc && root.svc.online ? root.svc.state : null
    if (st && root.stateWord === "trained") return { value: st.model.accuracy, measured: true }
    if (root.estimate && !isNaN(root.estimate.overall)) return { value: root.estimate.overall, measured: false }
    return null
  }

  // Only what needs doing or is happening; nothing when all is well (the
  // header already says how well it works).
  readonly property string stateLine: {
    root.rev; root.now
    var st = root.svc && root.svc.online ? root.svc.state : null
    if (!st || !root.cfg) return ""
    var w = root.stateWord
    if (w === "not yet") return "Can't be heard yet: saved for later."
    if (w === "training") return ""             // the bar below says it
    if (w === "unsaved" || root.dirty) return "Not saved yet."
    if (w === "not trained") return "Not trained yet: save to train."
    return ""
  }

  // Behind "Details": where the numbers come from, once.
  readonly property string detailsLine: {
    root.rev
    var st = root.svc && root.svc.online ? root.svc.state : null
    if (!st) return ""
    var lines = []
    var m = st.model
    if (m.ok) {
      lines.push("If it's unsure, it asks you to tap again (about 1 in " + Math.max(2, Math.round(1 / Math.max(1e-6, 1 - m.answered))) + " taps).")
      lines.push("Right " + Layout.pct(m.accuracy) + " of the time, based on " + m.taps + " taps from other days.")
    }
    lines.push("Numbers marked “estimate” are quick guesses and usually run a bit low.")
    return lines.join("\n")
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
          else if (k === Qt.Key_Return || k === Qt.Key_Enter) root.useBrowsed()
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
                text: !zone.est ? "" : isNaN(zone.est.acc) ? "only " + zone.est.n + " recorded taps here" : Layout.pct(zone.est.acc) + " right"
                color: !zone.est || isNaN(zone.est.acc) ? root.bad : root.grade(zone.est.acc)
                font.family: Style.font.menuFamily
                font.pixelSize: !zone.est || isNaN(zone.est.acc) ? Style.font.bodySmall : Style.font.display
              }
              Text {
                width: parent.width
                horizontalAlignment: Text.AlignHCenter
                visible: zone.est !== null && !isNaN(zone.est.acc)
                text: zone.est ? "estimate · " + zone.est.n + " taps" : ""
                color: Util.alpha(Commons.Color.foreground, 0.85)
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
        anchors.topMargin: Style.space(12)
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
          spacing: Style.space(10)

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
              anchors.right: estimateColumn.left
              anchors.rightMargin: Style.spacing.xl
              anchors.verticalCenter: parent.verticalCenter
              // The explanation lives behind the "?", not in the way.
              Row {
                spacing: Style.spacing.md
                Text {
                  anchors.verticalCenter: parent.verticalCenter
                  text: "Tap layout"
                  color: header.fg
                  font.family: Style.font.family
                  font.pixelSize: Style.font.heading
                  font.bold: true
                }
                Text {
                  anchors.verticalCenter: parent.verticalCenter
                  text: String.fromCodePoint(0xF0625)     // md-help_circle_outline
                  color: helpMouse.containsMouse ? Commons.Color.accent : Util.alpha(header.fg, 0.9)
                  font.family: Style.font.family
                  font.pixelSize: Style.font.title
                  MouseArea { id: helpMouse; anchors.fill: parent; hoverEnabled: true }
                  PanelToolTip {
                    visible: helpMouse.containsMouse
                    text: "Your screen is split into sections. Tapping one runs its command.\n"
                          + "Pick a layout here, then click a section on the screen to set what it runs."
                    fontFamily: Style.font.family
                  }
                }
              }
            }

            // One number: how often its best guess is the right section.
            Column {
              id: estimateColumn
              anchors.right: parent.right
              anchors.verticalCenter: parent.verticalCenter
              Row {
                anchors.right: parent.right
                spacing: Style.spacing.md
                Text {
                  anchors.baseline: bigNumber.baseline
                  visible: root.headline !== null && !root.headline.measured
                  text: "estimate"
                  color: Util.alpha(header.fg, 0.9)
                  font.family: Style.font.family
                  font.pixelSize: Style.font.bodySmall
                }
                Text {
                  id: bigNumber
                  text: root.headline ? Layout.pct(root.headline.value) : "—"
                  color: root.headline ? root.grade(root.headline.value) : Util.alpha(header.fg, 0.6)
                  font.family: Style.font.family
                  font.pixelSize: Style.font.display
                  font.bold: true
                }
                Text {
                  anchors.baseline: bigNumber.baseline
                  text: "right"
                  color: header.fg
                  font.family: Style.font.family
                  font.pixelSize: Style.font.body
                }
              }
              Text {
                anchors.right: parent.right
                text: "If unsure, it asks you to tap again."
                color: Util.alpha(header.fg, 0.9)
                font.family: Style.font.family
                font.pixelSize: Style.font.caption
              }
            }
          }

          PanelSeparator { visible: root.cfg !== null; width: parent.width }

          // The gesture's on/off switch shares the line, and says which in words.
          Item {
            visible: root.cfg !== null
            width: parent.width
            height: enabledRow.implicitHeight

            PanelSectionHeader { anchors.verticalCenter: parent.verticalCenter; text: "GESTURE"; foreground: header.fg }

            SolidToggle {
              id: enabledRow
              anchors.right: parent.right
              checked: (root.rev, root.cfg !== null && root.cfg[root.gesture].enabled)
              foreground: header.fg
              onToggled: root.toggleEnabled()
            }
          }

          // Selected is filled; trained carries a dot (a shape, not just a
          // colour); untrained is dimmer. The tooltip says which in words.
          Row {
            id: tabs
            visible: root.cfg !== null
            spacing: Style.spacing.controlGap
            Repeater {
              model: Layout.GESTURES
              delegate: SolidButton {
                required property string modelData
                required property int index
                readonly property string state: root.gestureState(modelData)
                readonly property bool trained: state === "trained"
                // The dot sits in the label at text size: a separate icon
                // would make the trained tab taller than the rest.
                text: (trained ? String.fromCodePoint(0xF09DE) + " " : "") + Layout.TITLES[modelData]   // md-circle_medium
                bordered: true
                selected: root.gesture === modelData
                foreground: trained || selected ? header.fg : Qt.tint(Commons.Color.popups.background, Util.alpha(header.fg, 0.55))
                fontSize: Style.font.bodySmall
                tooltipText: (trained ? "Trained"
                              : state === "not yet" ? "Not trained: the mic can't hear this gesture yet"
                              : state === "off" ? "Off: runs nothing"
                              : state === "training" ? "Training…"
                              : state === "unsaved" ? "Not saved yet" : "Not trained") + " (key " + (index + 1) + ")"
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
              PanelSectionHeader { anchors.verticalCenter: parent.verticalCenter; text: "1. HOW MANY SECTIONS"; foreground: header.fg }
              SolidChoices {
                // A count over the recommended limit carries a warning icon,
                // and says why on hover.
                readonly property int max: (root.rev, root.cfg ? root.cfg.max_sections : 5)
                function option(n) {
                  return n > max ? { value: String(n), label: n + " " + String.fromCodePoint(0xF0026),   // md-alert
                                     tooltip: "More than " + max + " sections is less accurate." }
                                 : { value: String(n), label: String(n) }
                }
                options: Layout.COUNTS.map(option)
                value: String(root.browseCount)
                foreground: header.fg
                fontSize: Style.font.bodySmall
                onChanged: v => root.setCount(parseInt(v))
              }
            }

            Text {
              anchors.right: parent.right
              anchors.verticalCenter: parent.verticalCenter
              text: (root.rev, root.cfg) ? root.choices.length + (root.choices.length === 1 ? " layout" : " layouts")
                                           + " with " + root.browseCount + (root.browseCount === 1 ? " section" : " sections") : ""
              color: Util.alpha(header.fg, 0.9)
              font.family: Style.font.family
              font.pixelSize: Style.font.bodySmall
            }
          }

          PanelSectionHeader { visible: root.cfg !== null; text: "2. WHICH LAYOUT"; foreground: header.fg }

          // The layouts with that many sections: the chosen one big in the
          // middle, its neighbours peeking in at the sides.
          // The layouts with that many sections: the one being looked at big
          // in the middle, its neighbours peeking in at the sides. Looking
          // changes nothing; "Use this layout" does.
          Item {
            id: carousel
            visible: root.cfg !== null
            width: parent.width
            height: slides.height
            readonly property real aspect: root.laptopScreen ? root.laptopScreen.height / root.laptopScreen.width : 0.625
            readonly property real slideWidth: Math.round(slides.width * 0.42)  // the neighbours show in part
            readonly property int current: root.browse
            function sync() { if (current >= 0 && current < root.choices.length && slides.currentIndex !== current) slides.currentIndex = current }
            onCurrentChanged: Qt.callLater(sync)

            ListView {
              id: slides
              anchors.left: parent.left
              anchors.right: parent.right
              // the preview, then its name (with "in use" or the button) and its numbers
              height: carousel.slideWidth * carousel.aspect + Style.spacing.md + Style.space(56)
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
              // Dragged or flicked to another layout: look at it.
              onMovementEnded: if (currentIndex >= 0 && currentIndex < root.choices.length) root.browse = currentIndex


              delegate: Item {
                id: slide
                required property var modelData
                required property int index
                readonly property bool on: index === carousel.current
                readonly property bool inUse: index === root.inUse
                // What each section would run, kept by position (Layout.applyPreset).
                readonly property var preview: (root.rev, Layout.applyPreset(root.sections, modelData.name))
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
                      id: tile
                      required property var modelData
                      required property int index
                      readonly property real inset: Math.max(2, Style.space(4))
                      readonly property var s: slide.preview[index]
                      x: inset + modelData[0] * (frame.width - 2 * inset) + 1
                      y: inset + modelData[1] * (frame.height - 2 * inset) + 1
                      width: modelData[2] * (frame.width - 2 * inset) - 2
                      height: modelData[3] * (frame.height - 2 * inset) - 2
                      readonly property bool over: index < Layout.PRESET_OVER[slide.modelData.name]
                      readonly property color tint: Util.alpha(root.zoneColors[index % root.zoneColors.length], 0.6)
                      z: over ? 1 : 0
                      radius: Style.cornerRadius
                      color: over ? Qt.tint(Commons.Color.popups.background, tint) : tint
                      // The section being edited is outlined in the accent; others
                      // that lie over a neighbour keep a gap round them, as drawn.
                      readonly property bool editing: slide.inUse && root.selected === index
                      border.width: editing || over ? Math.max(2, Style.space(3)) : 0
                      border.color: editing ? Commons.Color.accent : Commons.Color.popups.background

                      // On the layout in use, a tile is another way to pick its
                      // section: the screen one may be hidden under this panel.
                      MouseArea {
                        id: tileMouse
                        enabled: slide.on && slide.inUse
                        anchors.fill: parent
                        hoverEnabled: true
                        cursorShape: Qt.PointingHandCursor
                        onClicked: root.select(tile.index)
                      }
                      PanelToolTip {
                        visible: tileMouse.enabled && tileMouse.containsMouse
                        text: "Set what this section runs"
                        fontFamily: Style.font.family
                      }

                      // What tapping here runs, or a "+" where nothing is set yet.
                      Text {
                        readonly property bool empty: !(tile.s && tile.s.command)
                        visible: slide.on && tile.height > Style.space(16)
                        anchors.centerIn: parent
                        width: parent.width - Style.space(8)
                        horizontalAlignment: Text.AlignHCenter
                        elide: Text.ElideMiddle
                        text: empty ? String.fromCodePoint(0xF0415) : tile.s.command     // md-plus
                        color: empty ? Util.alpha(header.fg, 0.9) : header.fg
                        font.family: Style.font.family
                        font.pixelSize: empty ? Style.font.body : Style.font.caption
                      }
                    }
                  }
                }

                Column {
                  visible: slide.on                     // a neighbour's would be cut off
                  anchors.top: frame.bottom
                  anchors.topMargin: Style.spacing.md
                  anchors.horizontalCenter: parent.horizontalCenter
                  spacing: Style.spacing.xs
                  // The name, then "in use" or the button that makes it so.
                  Row {
                    anchors.horizontalCenter: parent.horizontalCenter
                    spacing: Style.spacing.lg
                    Text {
                      anchors.verticalCenter: parent.verticalCenter
                      text: slide.modelData.name
                      color: header.fg
                      font.family: Style.font.family
                      font.pixelSize: Style.font.heading
                      font.bold: true
                    }
                    Text {
                      visible: slide.inUse
                      anchors.verticalCenter: parent.verticalCenter
                      text: String.fromCodePoint(0xF012C) + " in use"     // md-check
                      color: root.good
                      font.family: Style.font.family
                      font.pixelSize: Style.font.bodySmall
                    }
                    SolidButton {
                      visible: !slide.inUse
                      anchors.verticalCenter: parent.verticalCenter
                      text: "Use this layout"
                      bordered: true
                      selected: true
                      foreground: header.fg
                      fontSize: Style.font.bodySmall
                      tooltipText: "Replaces the layout in use (Enter). Commands stay with their sections by position."
                      onClicked: root.useBrowsed()
                    }
                  }
                  // The layout in use has its real number in the header; only
                  // a different one gets an estimate.
                  Text {
                    anchors.horizontalCenter: parent.horizontalCenter
                    text: (!slide.inUse && slide.modelData.acc >= 0 ? Layout.pct(slide.modelData.acc) + " (estimate)  ·  " : "")
                          + (slide.index + 1) + " of " + root.choices.length
                    color: Util.alpha(header.fg, 0.9)
                    font.family: Style.font.family
                    font.pixelSize: Style.font.bodySmall
                  }
                }

                HoverHandler { cursorShape: slide.on ? Qt.ArrowCursor : Qt.PointingHandCursor }
                TapHandler { onTapped: root.browse = slide.index }
              }
            }

            // Arrows right beside the layout being looked at.
            SolidButton {
              id: prevButton
              z: 2
              x: (carousel.width - carousel.slideWidth) / 2 - width - Style.spacing.md
              y: (carousel.slideWidth * carousel.aspect - height) / 2
              iconText: String.fromCodePoint(0xF0141)     // md-chevron_left
              bordered: true
              enabled: root.choices.length > 1
              tooltipText: "Previous layout (Shift+P)"
              onClicked: root.stepPreset(-1)
            }
            SolidButton {
              z: 2
              x: (carousel.width + carousel.slideWidth) / 2 + Style.spacing.md
              y: prevButton.y
              iconText: String.fromCodePoint(0xF0142)     // md-chevron_right
              bordered: true
              enabled: root.choices.length > 1
              tooltipText: "Next layout (P)"
              onClicked: root.stepPreset(1)
            }
          }


          PanelSeparator { visible: root.cfg !== null; width: parent.width }

          // the chosen gesture's settings
          // How a recognised tap shows itself (every gesture).
          Row {
            id: controls
            visible: root.cfg !== null
            spacing: Style.spacing.xl

            PanelSectionHeader { anchors.verticalCenter: parent.verticalCenter; text: "WHEN A TAP IS RECOGNISED"; foreground: header.fg }
            SolidChoices {
              anchors.verticalCenter: parent.verticalCenter
              readonly property var tips: ({ flash: "The section lights up",
                                             notify: "A desktop notification with the section's name and command",
                                             none: "No feedback" })
              options: Layout.FEEDBACK.map(f => ({ value: f, label: f, tooltip: tips[f] }))
              value: (root.rev, root.cfg) ? root.cfg.feedback : ""
              foreground: header.fg
              fontSize: Style.font.bodySmall
              onChanged: v => root.setFeedback(v)
            }
            // Sounds for a tap, "tap again", pause… (every gesture).
            PanelSectionHeader { anchors.verticalCenter: parent.verticalCenter; text: "SOUND"; foreground: header.fg }
            SolidToggle {
              anchors.verticalCenter: parent.verticalCenter
              checked: (root.rev, root.cfg !== null && root.cfg.sound !== false)
              foreground: header.fg
              onToggled: root.toggleSound()
            }
          }

          PanelSeparator { visible: root.cfg !== null; width: parent.width }

          // Details on the left, the actions pinned top right: opening the
          // details grows the footer downwards and moves neither.
          Item {
            visible: root.cfg !== null
            width: parent.width
            height: Math.max(statusColumn.implicitHeight, actions.implicitHeight)

            Column {
              id: statusColumn
              anchors.left: parent.left
              anchors.top: parent.top
              width: parent.width - actions.width - Style.spacing.xl
              spacing: Style.spacing.xs
              TrainingBar {
                visible: root.stateWord === "training"
                width: Math.min(parent.width, Style.space(420))
                training: (root.rev, root.svc && root.svc.online ? root.svc.state.training : null)
                now: root.now
                foreground: header.fg
              }
              Text {
                width: parent.width
                wrapMode: Text.Wrap
                text: root.stateLine
                visible: text !== ""
                color: header.fg
                font.family: Style.font.family
                font.pixelSize: Style.font.body
              }
              Text {
                width: parent.width
                wrapMode: Text.Wrap
                text: root.warn
                visible: text !== ""
                color: /not |fail|more than|overlap|smaller/.test(root.warn) ? root.bad : Util.alpha(header.fg, 0.9)
                font.family: Style.font.family
                font.pixelSize: Style.font.bodySmall
              }
              Text {
                visible: root.detailsLine !== ""
                text: (root.showDetails ? "Hide details " : "Details ")
                      + String.fromCodePoint(root.showDetails ? 0xF0143 : 0xF0140)   // md-chevron_up / down
                font.underline: true
                color: detailsMouse.containsMouse ? Commons.Color.accent : Util.alpha(header.fg, 0.9)
                font.family: Style.font.family
                font.pixelSize: Style.font.bodySmall
                MouseArea {
                  id: detailsMouse
                  anchors.fill: parent
                  hoverEnabled: true
                  cursorShape: Qt.PointingHandCursor
                  onClicked: root.showDetails = !root.showDetails
                }
              }
              Text {
                visible: root.showDetails
                width: parent.width
                wrapMode: Text.Wrap
                text: root.detailsLine
                color: Util.alpha(header.fg, 0.9)
                font.family: Style.font.family
                font.pixelSize: Style.font.bodySmall
              }
            }

            Row {
              id: actions
              anchors.right: parent.right
              anchors.top: parent.top
              spacing: Style.spacing.controlGap
              SolidButton {
                text: "Undo last change"
                enabled: root.history.length > 0
                foreground: header.fg
                tooltipText: "Undoes your last edit in this editor (U)"
                onClicked: root.undo()
              }
              // Saved is a state, not a button: only unsaved changes get one.
              SolidButton {
                visible: root.dirty
                text: root.saveTrains ? "Save and train" : "Save"
                bordered: true
                selected: true
                foreground: header.fg
                tooltipText: root.saveTrains ? "Saves, then trains on your recorded taps in the background (S)" : "Saves your changes (S)"
                onClicked: root.save()
              }
              Row {
                visible: !root.dirty
                anchors.verticalCenter: parent.verticalCenter
                spacing: Style.space(6)
                Text {
                  anchors.verticalCenter: parent.verticalCenter
                  text: String.fromCodePoint(0xF012C)     // md-check
                  color: root.good
                  font.family: Style.font.family
                  font.pixelSize: Style.font.body
                }
                Text {
                  anchors.verticalCenter: parent.verticalCenter
                  text: "All changes saved"
                  color: header.fg
                  font.family: Style.font.family
                  font.pixelSize: Style.font.body
                }
              }
            }
          }
        }
      }

      // --------------------------------------------- name, command, picks

      BorderSurface {
        id: details
        readonly property real pad: Style.space(12)

        visible: root.selected >= 0
        // Compact and low, just above the hints, so it fits under the main
        // panel even on a short screen.
        anchors.horizontalCenter: parent.horizontalCenter
        y: Math.max(header.y + header.height + Style.space(8), hints.y - Style.space(8) - height)
        width: Math.min(parent.width - Style.space(32), Style.space(1100))
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
          spacing: Style.space(8)

          // The label says which section (it's named by where it is).
          Column {
            width: parent.width
            spacing: Style.spacing.xs
            PanelSectionHeader {
              text: "COMMAND" + ((root.rev, root.selected >= 0 && root.sections[root.selected])
                                 ? " · " + root.sections[root.selected].name.toUpperCase() : "")
              foreground: header.fg
            }
            TextField {
              id: commandField
              width: parent.width
              placeholderText: "shell command, e.g. omarchy launch browser"
              foreground: header.fg
              onTextEdited: root.edit("command", text)
              Keys.onEscapePressed: keyCatcher.forceActiveFocus()
              onAccepted: keyCatcher.forceActiveFocus()
            }
          }

          Flow {
            width: parent.width
            spacing: Style.spacing.controlGap
            Repeater {
              model: Layout.OMARCHY_ACTIONS
              delegate: SolidButton {
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
        text: "click a section: set its command · drag: move / resize · right-click: delete   |   keys: 1–5 gesture · N sections · P look at next layout · Enter use it · E on/off · U undo · S save · Esc close"
        color: Util.alpha(Commons.Color.foreground, 0.7)
        font.family: Style.font.menuFamily
        font.pixelSize: Style.font.caption
      }
    }
  }
}
