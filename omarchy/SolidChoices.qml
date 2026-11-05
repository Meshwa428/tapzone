import QtQuick
import qs.Commons
import qs.Commons as Commons
import qs.Ui

// A row of SolidButtons, one per option, the chosen one selected: the
// shell's ButtonGroup with opaque buttons. `options` is a string[] or
// { value, label, tooltip? } objects, as in ButtonGroup.
Row {
  id: root
  property var options: []
  property string value: ""
  property color foreground: Commons.Color.foreground
  property real fontSize: Style.font.body
  signal changed(string value)

  spacing: Style.spacing.controlGap

  function optionValue(o) { return o && typeof o === "object" ? String(o.value) : String(o) }

  Repeater {
    model: root.options
    delegate: SolidButton {
      required property var modelData
      text: modelData && typeof modelData === "object" && modelData.label !== undefined ? String(modelData.label) : String(modelData)
      tooltipText: modelData && typeof modelData === "object" && modelData.tooltip ? String(modelData.tooltip) : ""
      selected: root.optionValue(modelData) === root.value
      bordered: true
      foreground: root.foreground
      fontSize: root.fontSize
      onClicked: root.changed(root.optionValue(modelData))
    }
  }
}
