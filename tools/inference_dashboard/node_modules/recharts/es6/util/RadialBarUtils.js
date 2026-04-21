function _extends() { return _extends = Object.assign ? Object.assign.bind() : function (n) { for (var e = 1; e < arguments.length; e++) { var t = arguments[e]; for (var r in t) ({}).hasOwnProperty.call(t, r) && (n[r] = t[r]); } return n; }, _extends.apply(null, arguments); }
import * as React from 'react';
import { Shape } from './ActiveShapeUtils';
export function parseCornerRadius(cornerRadius) {
  if (typeof cornerRadius === 'string') {
    return parseInt(cornerRadius, 10);
  }
  return cornerRadius;
}
export function RadialBarSector(props) {
  return /*#__PURE__*/React.createElement(Shape, _extends({
    shapeType: "sector"
  }, props));
}