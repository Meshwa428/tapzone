.pragma library

// Layouts for the editor, the same rules as gestures.py (keep the two in
// step): a layout is a list of [x, y, width, height] rects as fractions of
// the screen from the top-left, one per section.

var GESTURES = ["single", "double", "triple", "two_finger", "three_finger"]
var TITLES = { single: "Single tap", double: "Double tap", triple: "Triple tap",
               two_finger: "2-finger tap", three_finger: "3-finger tap" }
var FEEDBACK = ["flash", "notify", "none"]
var MIN_SIDE = 0.05     // a section smaller than this is below the mic's typical error
var SNAP = 0.012        // edges snap to the screen and to other sections within this

function rows(k) {
  var out = []
  for (var i = 0; i < k; i++) out.push([0, i / k, 1, 1 / k])
  return out
}

function columns(k) {
  var out = []
  for (var i = 0; i < k; i++) out.push([i / k, 0, 1 / k, 1])
  return out
}

function grid(cols, rws) {
  var out = []
  for (var r = 0; r < rws; r++)
    for (var c = 0; c < cols; c++) out.push([c / cols, r / rws, 1 / cols, 1 / rws])
  return out
}

var PRESET_ORDER = ["whole screen", "top / bottom", "left / right", "3 rows", "quadrants", "5 rows", "3 columns"]
var PRESETS = {
  "whole screen": rows(1), "top / bottom": rows(2), "left / right": columns(2), "3 rows": rows(3),
  "quadrants": grid(2, 2), "5 rows": rows(5), "3 columns": columns(3)
}
var PRESET_NAMES = {
  "whole screen": ["Anywhere"], "top / bottom": ["Top", "Bottom"], "left / right": ["Left", "Right"],
  "3 rows": ["Top", "Middle", "Bottom"], "quadrants": ["Top left", "Top right", "Bottom left", "Bottom right"],
  "5 rows": ["Row 1", "Row 2", "Row 3", "Row 4", "Row 5"], "3 columns": ["Left", "Middle", "Right"]
}

// Commands offered next to the free-text field: common Omarchy actions.
// Any shell command works; these are only shortcuts.
var OMARCHY_ACTIONS = [
  { label: "Browser", command: "omarchy launch browser" },
  { label: "Terminal", command: "omarchy launch terminal" },
  { label: "Omarchy menu", command: "omarchy menu toggle" },
  { label: "Screenshot", command: "omarchy capture screenshot" },
  { label: "Lock screen", command: "omarchy system lock" },
  { label: "Night light", command: "omarchy toggle nightlight" },
  { label: "Clipboard", command: "omarchy menu clipboard" },
  { label: "Emoji", command: "omarchy menu emoji" },
  { label: "Play / pause", command: "playerctl play-pause" }
]

function clone(v) { return JSON.parse(JSON.stringify(v)) }

function layoutOf(sections) {
  return sections.map(function(s) { return s.rect })
}

function sameRect(a, b) {
  for (var i = 0; i < 4; i++) if (Math.abs(a[i] - b[i]) > 1e-4) return false
  return true
}

// The preset a layout is, or "custom".
function presetOf(sections) {
  for (var k = 0; k < PRESET_ORDER.length; k++) {
    var p = PRESETS[PRESET_ORDER[k]]
    if (p.length !== sections.length) continue
    var same = true
    for (var i = 0; i < p.length && same; i++) same = sameRect(p[i], sections[i].rect)
    if (same) return PRESET_ORDER[k]
  }
  return "custom"
}

// Sections of the next (step 1) or previous (-1) preset; commands are kept
// by position, so trying presets doesn't lose them.
function stepPreset(sections, step) {
  var cur = PRESET_ORDER.indexOf(presetOf(sections))
  var k = cur < 0 ? 0 : (cur + step + PRESET_ORDER.length) % PRESET_ORDER.length
  var name = PRESET_ORDER[k]
  return PRESETS[name].map(function(r, j) {
    return { name: PRESET_NAMES[name][j], rect: r.slice(), command: j < sections.length ? sections[j].command : "" }
  })
}

// The section a point (screen fractions) falls in, or -1 for a dead zone.
function sectionOf(x, y, layout) {
  x = Math.min(Math.max(x, 0), 1 - 1e-9)
  y = Math.min(Math.max(y, 0), 1 - 1e-9)
  for (var i = 0; i < layout.length; i++) {
    var r = layout[i]
    if (x >= r[0] && x < r[0] + r[2] && y >= r[1] && y < r[1] + r[3]) return i
  }
  return -1
}

// What's wrong with a gesture's layout, as messages (empty: fine).
function problems(cfg, gesture) {
  var g = cfg[gesture], out = []
  var rects = layoutOf(g.sections)
  if (g.enabled && rects.length === 0) out.push("add a section, or turn the gesture off")
  if (rects.length > cfg.max_sections)
    out.push(rects.length + " sections: more than max_sections (" + cfg.max_sections + ")")
  for (var i = 0; i < rects.length; i++) {
    var r = rects[i], name = g.sections[i].name
    if (r[0] < -1e-6 || r[1] < -1e-6 || r[0] + r[2] > 1 + 1e-6 || r[1] + r[3] > 1 + 1e-6)
      out.push(name + " goes off the screen")
    if (r[2] < MIN_SIDE || r[3] < MIN_SIDE)
      out.push(name + " is smaller than " + Math.round(MIN_SIDE * 100) + "% of the screen")
    for (var j = 0; j < i; j++) {
      var a = rects[j]
      if (a[0] < r[0] + r[2] - 1e-6 && r[0] < a[0] + a[2] - 1e-6 && a[1] < r[1] + r[3] - 1e-6 && r[1] < a[1] + a[3] - 1e-6)
        out.push(g.sections[j].name + " and " + name + " overlap")
    }
  }
  return out
}

// v rounded to half a percent, pulled onto the screen edge or any other
// section's edge within SNAP. axis 0 is x, 1 is y.
function snap(v, axis, sections, skip) {
  var others = [0, 1]
  for (var j = 0; j < sections.length; j++) {
    if (j === skip) continue
    var p = sections[j].rect[axis], n = sections[j].rect[axis + 2]
    others.push(p, p + n)
  }
  var near = others[0]
  for (var k = 1; k < others.length; k++) if (Math.abs(others[k] - v) < Math.abs(near - v)) near = others[k]
  return Math.abs(near - v) < SNAP ? near : Math.min(1, Math.max(0, Math.round(v / 0.005) * 0.005))
}

// The live estimate (gestures.quick_estimate): est is the service's
// "estimate" event, every recorded tap's probability of each cell of a
// cells × cells grid from a model that never heard that tap's session. A
// section's probability is the sum of the cells whose centre it covers.
// Returns { overall, per: [{ acc, n }] }; acc is NaN under min_taps taps.
function quickEstimate(est, layout) {
  var n = layout.length, per = []
  if (!est || n === 0) return null
  var y = est.where.map(function(w) { return sectionOf(w[0], w[1], layout) })
  var counts = [], right = []
  for (var i = 0; i < n; i++) { counts.push(0); right.push(0) }
  for (var t = 0; t < y.length; t++) if (y[t] >= 0) counts[y[t]]++
  if (n < 2) return { overall: 1, per: [{ acc: 1, n: counts[0] }] }
  var c = est.cells, owner = []
  for (var r = 0; r < c; r++)
    for (var q = 0; q < c; q++) owner.push(sectionOf((q + 0.5) / c, (r + 0.5) / c, layout))
  var kept = 0, ok = 0
  for (t = 0; t < y.length; t++) {
    if (y[t] < 0) continue
    var score = [], P = est.P[t]
    for (i = 0; i < n; i++) score.push(0)
    for (var cell = 0; cell < owner.length; cell++) if (owner[cell] >= 0) score[owner[cell]] += P[cell]
    var best = 0
    for (i = 1; i < n; i++) if (score[i] > score[best]) best = i
    kept++
    if (best === y[t]) { ok++; right[y[t]]++ }
  }
  for (i = 0; i < n; i++)
    per.push({ acc: counts[i] >= est.min_taps ? right[i] / counts[i] : NaN, n: counts[i] })
  return { overall: kept ? ok / kept : NaN, per: per }
}

function pct(a) { return Math.round(a * 100) + "%" }
