function _extends() { return _extends = Object.assign ? Object.assign.bind() : function (n) { for (var e = 1; e < arguments.length; e++) { var t = arguments[e]; for (var r in t) ({}).hasOwnProperty.call(t, r) && (n[r] = t[r]); } return n; }, _extends.apply(null, arguments); }
import * as React from 'react';
import invariant from 'tiny-invariant';
import { Shape } from './ActiveShapeUtils';
import { isNullish, isNumber } from './DataUtils';
export function BarRectangle(props) {
  return /*#__PURE__*/React.createElement(Shape, _extends({
    shapeType: "rectangle",
    activeClassName: "recharts-active-bar",
    inActiveClassName: "recharts-inactive-bar"
  }, props));
}
/**
 * Safely gets minPointSize from the minPointSize prop if it is a function
 * @param minPointSize minPointSize as passed to the Bar component
 * @param defaultValue default minPointSize
 * @returns minPointSize
 */
export var minPointSizeCallback = function minPointSizeCallback(minPointSize) {
  var defaultValue = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : 0;
  return (value, index) => {
    if (isNumber(minPointSize)) return minPointSize;
    var isValueNumberOrNil = isNumber(value) || isNullish(value);
    if (isValueNumberOrNil) {
      return minPointSize(value, index);
    }
    !isValueNumberOrNil ? true ? invariant(false, "minPointSize callback function received a value with type of ".concat(typeof value, ". Currently only numbers or null/undefined are supported.")) : invariant(false) : void 0;
    return defaultValue;
  };
};