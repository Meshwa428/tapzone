.pragma library

// Layouts for the editor, the same rules as tapzones/layout.py (kept in
// step by tests/test_layout.py): a layout is a list of [x, y, width, height] rects as fractions of
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

// ------------------------------------------------------- the presets
//
// Layouts come only from presets (layout.py has the same generator): every
// layout straight cuts can make for 1–5 sections, "1 / 2" for rows (top to
// bottom) and "1 | 2" for columns (left to right), plus corners + centre
// (a section over the middle of the quadrants) and a few of 6.

var COUNTS = [1, 2, 3, 4, 5, 6]

function compositions(n) {
  var out = []
  for (var first = 1; first < n; first++) {
    out.push([first, n - first])
    var tails = compositions(n - first)
    for (var t = 0; t < tails.length; t++) out.push([first].concat(tails[t]))
  }
  return out
}

function cuts(n, box, across) {
  var x = box[0], y = box[1], w = box[2], h = box[3], out = []
  var all = compositions(n)
  for (var p = 0; p < all.length; p++) {
    var parts = all[p], k = parts.length, options = []
    for (var i = 0; i < k; i++) {
      var band = across ? [x, y + i * h / k, w, h / k] : [x + i * w / k, y, w / k, h]
      options.push(fill(parts[i], band, !across))
    }
    // Every combination of the bands' options, the first band varying slowest.
    var combos = [[]]
    for (i = 0; i < k; i++) {
      var next = []
      for (var c = 0; c < combos.length; c++)
        for (var o = 0; o < options[i].length; o++) next.push(combos[c].concat([options[i][o]]))
      combos = next
    }
    for (c = 0; c < combos.length; c++) {
      var rects = [], labels = []
      for (i = 0; i < k; i++) { rects = rects.concat(combos[c][i][0]); labels.push(combos[c][i][1]) }
      out.push([rects, labels.join(across ? " / " : " | ")])
    }
  }
  return out
}

function fill(n, box, across) {
  if (n === 1) return [[[box], "1"]]
  var plain = []
  for (var i = 0; i < n; i++) plain.push("1")
  plain = plain.join(across ? " / " : " | ")
  return cuts(n, box, across).map(function(r) { return [r[0], r[1] === plain ? String(n) : "(" + r[1] + ")"] })
}

function round6(v) { return Math.round(v * 1e6) / 1e6 }

function readingOrder(rects) {
  return rects.slice().sort(function(a, b) { return round6(a[1]) - round6(b[1]) || round6(a[0]) - round6(b[0]) })
}

function sectionNames(rects) {
  var n = rects.length, i
  var numbered = function(word) { return rects.map(function(r, j) { return word + " " + (j + 1) }) }
  if (n > 3 && rects.every(function(r) { return r[2] > 1 - 1e-9 })) return numbered("Row")
  if (n > 3 && rects.every(function(r) { return r[3] > 1 - 1e-9 })) return numbered("Column")
  var word = function(c, low, mid, high) { return c < 1 / 3 - 1e-9 ? low : c > 2 / 3 + 1e-9 ? high : mid }
  var names = rects.map(function(r) {
    var v = r[3] > 1 - 1e-9 ? "" : word(r[1] + r[3] / 2, "Top", "Middle", "Bottom")
    var s = r[2] > 1 - 1e-9 ? "" : word(r[0] + r[2] / 2, "Left", "Centre", "Right")
    return v && s ? v + " " + s.toLowerCase() : v || s || "Anywhere"
  })
  // The same place twice: number them.
  var distinct = names.filter(function(m, j) { return names.indexOf(m) === j }).sort()
  for (var d = 0; d < distinct.length; d++) {
    var same = []
    for (i = 0; i < n; i++) if (names[i] === distinct[d]) same.push(i)
    if (same.length > 1) for (var j = 0; j < same.length; j++) names[same[j]] = distinct[d] + " " + (j + 1)
  }
  return names
}

function friendly(rects) {
  var n = rects.length
  if (n === 1) return "whole screen"
  if (rects.every(function(r) { return r[2] > 1 - 1e-9 })) return n === 2 ? "top / bottom" : n + " rows"
  if (rects.every(function(r) { return r[3] > 1 - 1e-9 })) return n === 2 ? "left / right" : n + " columns"
  if (n === 4 && rects.every(function(r) { return Math.abs(r[2] - 0.5) < 1e-9 && Math.abs(r[3] - 0.5) < 1e-9 }))
    return "quadrants"
  return null
}

// What most people pick for each count: listed first, and where the editor
// lands when the count changes.
var DEFAULTS = { 1: "whole screen", 2: "top / bottom", 3: "3 rows", 4: "quadrants",
                 5: "corners + centre", 6: "3 × 2 grid" }

// [how deeply the cuts nest, how many cuts] from a layout's label.
function complexity(label) {
  var depth = 0, deepest = 0, numbers = label.match(/\d+/g) || [], n = 1
  for (var i = 0; i < label.length; i++) {
    depth += (label[i] === "(") - (label[i] === ")")
    deepest = Math.max(deepest, depth)
    if (label[i] === "(") n++
  }
  for (i = 0; i < numbers.length; i++) if (Number(numbers[i]) > 1) n++
  return [deepest, n]
}

// The quadrants with a section over the middle (sections listed first sit
// over the others): four L-shaped corners round a centre.
function cornersAndCentre() {
  return { rects: [[0.3, 0.3, 0.4, 0.4]].concat(grid(2, 2)),
           names: ["Centre", "Top left", "Top right", "Bottom left", "Bottom right"] }
}

// [{ name, rects, names, over }] of every preset with n sections, the usual
// one first, then the simplest; `over` is how many of the first sections
// sit over the others.
function catalogue(n) {
  if (n === 6) {
    var six = [[grid(3, 2), "3 × 2 grid"], [grid(2, 3), "2 × 3 grid"], [rows(6), null], [columns(6), null]]
    return six.map(function(s) {
      return { name: friendly(s[0]) || s[1], rects: s[0], names: sectionNames(s[0]), over: 0 }
    })
  }
  var out = [], seen = {}
  for (var pass = 0; pass < 2; pass++) {
    var found = n > 1 ? cuts(n, [0, 0, 1, 1], pass === 0) : [[[[0, 0, 1, 1]], "1"]]
    for (var f = 0; f < found.length; f++) {
      var rects = readingOrder(found[f][0])
      var key = JSON.stringify(rects.map(function(r) { return r.map(round6) }))
      if (seen[key]) continue
      seen[key] = true
      out.push({ name: friendly(rects) || found[f][1], rects: rects, names: sectionNames(rects), over: 0,
                 rank: complexity(found[f][1]) })
    }
  }
  if (n === 5) {
    var cc = cornersAndCentre()
    out.push({ name: "corners + centre", rects: cc.rects, names: cc.names, over: 1, rank: [0, 0] })
  }
  out.forEach(function(p, i) { p.i = i })
  out.sort(function(a, b) {
    return (a.name !== DEFAULTS[n]) - (b.name !== DEFAULTS[n]) || a.rank[0] - b.rank[0] || a.rank[1] - b.rank[1] || a.i - b.i
  })
  return out.map(function(p) { return { name: p.name, rects: p.rects, names: p.names, over: p.over } })
}

// name → rects, in order of section count, the usual first; BY_COUNT[n]
// lists the names with n sections.
var PRESETS = {}, PRESET_NAMES = {}, PRESET_OVER = {}, BY_COUNT = {}
for (var _c = 0; _c < COUNTS.length; _c++) {
  var _all = catalogue(COUNTS[_c])
  BY_COUNT[COUNTS[_c]] = _all.map(function(p) { return p.name })
  for (var _k = 0; _k < _all.length; _k++) {
    PRESETS[_all[_k].name] = _all[_k].rects
    PRESET_NAMES[_all[_k].name] = _all[_k].names
    PRESET_OVER[_all[_k].name] = _all[_k].over
  }
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

// The preset a layout is, or "custom" (moved or resized).
function presetOf(sections) {
  var names = BY_COUNT[sections.length] || []
  for (var k = 0; k < names.length; k++) {
    var p = PRESETS[names[k]], same = true
    for (var i = 0; i < p.length && same; i++) same = sameRect(p[i], sections[i].rect)
    if (same) return names[k]
  }
  return "custom"
}

// The sections of preset `name`; commands are kept by position, so trying
// presets doesn't lose them.
function applyPreset(sections, name) {
  return PRESETS[name].map(function(r, j) {
    var s = { name: PRESET_NAMES[name][j], rect: r.slice(), command: j < sections.length ? sections[j].command : "" }
    if (j < PRESET_OVER[name]) s.over = true
    return s
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
  // More than max_sections isn't a problem: it's the recommended limit, and the editor only warns.
  for (var i = 0; i < rects.length; i++) {
    var r = rects[i], name = g.sections[i].name
    if (r[0] < -1e-6 || r[1] < -1e-6 || r[0] + r[2] > 1 + 1e-6 || r[1] + r[3] > 1 + 1e-6)
      out.push(name + " goes off the screen")
    if (r[2] < MIN_SIDE || r[3] < MIN_SIDE)
      out.push(name + " is smaller than " + Math.round(MIN_SIDE * 100) + "% of the screen")
    var over = function(k) { return !!g.sections[k].over }
    if (over(i)) {
      for (var k = 0; k < i; k++) if (!over(k)) { out.push(name + " sits over the others, so it must come first"); break }
    }
    for (var j = 0; j < i; j++) {
      if (over(i) || over(j)) continue    // one sits over the other and takes that area
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

// The presets with n sections as [{ name, acc }], the usual one first, then
// the simplest; acc is the estimate, -1 without one.
function choicesFor(est, n) {
  return (BY_COUNT[n] || []).map(function(name) {
    var e = est ? quickEstimate(est, PRESETS[name]) : null
    return { name: name, acc: e && !isNaN(e.overall) ? e.overall : -1 }
  })
}

function pct(a) { return Math.round(a * 100) + "%" }
