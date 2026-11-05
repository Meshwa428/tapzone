.pragma library
// The UI's own sounds: a tap for every button, toggle on/off for every
// switch. The same files the service plays (tapzones/sound.py), and the
// same setting: the editor and the panel keep `on` in step with the
// config's `sound`. One copy for every importer (.pragma library).

var on = true

// The command that plays `moment` ("tap", "toggle_on", "toggle_off"), or
// null when sound is off. `dir` is this folder's URL (Qt.resolvedUrl(".")
// from a QML file next to this one).
function command(dir, moment) {
  if (!on) return null
  var name = moment === "tap" ? "tap_0" + (1 + Math.floor(Math.random() * 5)) : moment
  var path = decodeURIComponent(String(dir).replace(/^file:\/\//, "")) + "../tapzones/sounds/" + name + ".wav"
  var q = "'" + path.replace(/'/g, "'\\''") + "'"
  return ["sh", "-c", "pw-play " + q + " 2>/dev/null || paplay " + q + " 2>/dev/null || aplay -q " + q]
}
