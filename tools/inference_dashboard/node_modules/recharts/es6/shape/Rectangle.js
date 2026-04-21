var _excluded = ["radius"],
  _excluded2 = ["radius"];
var _templateObject, _templateObject2, _templateObject3, _templateObject4, _templateObject5, _templateObject6, _templateObject7, _templateObject8, _templateObject9, _templateObject0;
function ownKeys(e, r) { var t = Object.keys(e); if (Object.getOwnPropertySymbols) { var o = Object.getOwnPropertySymbols(e); r && (o = o.filter(function (r) { return Object.getOwnPropertyDescriptor(e, r).enumerable; })), t.push.apply(t, o); } return t; }
function _objectSpread(e) { for (var r = 1; r < arguments.length; r++) { var t = null != arguments[r] ? arguments[r] : {}; r % 2 ? ownKeys(Object(t), !0).forEach(function (r) { _defineProperty(e, r, t[r]); }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(t)) : ownKeys(Object(t)).forEach(function (r) { Object.defineProperty(e, r, Object.getOwnPropertyDescriptor(t, r)); }); } return e; }
function _defineProperty(e, r, t) { return (r = _toPropertyKey(r)) in e ? Object.defineProperty(e, r, { value: t, enumerable: !0, configurable: !0, writable: !0 }) : e[r] = t, e; }
function _toPropertyKey(t) { var i = _toPrimitive(t, "string"); return "symbol" == typeof i ? i : i + ""; }
function _toPrimitive(t, r) { if ("object" != typeof t || !t) return t; var e = t[Symbol.toPrimitive]; if (void 0 !== e) { var i = e.call(t, r || "default"); if ("object" != typeof i) return i; throw new TypeError("@@toPrimitive must return a primitive value."); } return ("string" === r ? String : Number)(t); }
function _extends() { return _extends = Object.assign ? Object.assign.bind() : function (n) { for (var e = 1; e < arguments.length; e++) { var t = arguments[e]; for (var r in t) ({}).hasOwnProperty.call(t, r) && (n[r] = t[r]); } return n; }, _extends.apply(null, arguments); }
function _objectWithoutProperties(e, t) { if (null == e) return {}; var o, r, i = _objectWithoutPropertiesLoose(e, t); if (Object.getOwnPropertySymbols) { var n = Object.getOwnPropertySymbols(e); for (r = 0; r < n.length; r++) o = n[r], -1 === t.indexOf(o) && {}.propertyIsEnumerable.call(e, o) && (i[o] = e[o]); } return i; }
function _objectWithoutPropertiesLoose(r, e) { if (null == r) return {}; var t = {}; for (var n in r) if ({}.hasOwnProperty.call(r, n)) { if (-1 !== e.indexOf(n)) continue; t[n] = r[n]; } return t; }
function _taggedTemplateLiteral(e, t) { return t || (t = e.slice(0)), Object.freeze(Object.defineProperties(e, { raw: { value: Object.freeze(t) } })); }
/**
 * @fileOverview Rectangle
 */
import * as React from 'react';
import { useEffect, useMemo, useRef, useState } from 'react';
import { clsx } from 'clsx';
import { resolveDefaultProps } from '../util/resolveDefaultProps';
import { JavascriptAnimate } from '../animation/JavascriptAnimate';
import { interpolate } from '../util/DataUtils';
import { useAnimationId } from '../util/useAnimationId';
import { getTransitionVal } from '../animation/util';
import { svgPropertiesAndEvents } from '../util/svgPropertiesAndEvents';
import { round, roundTemplateLiteral } from '../util/round';

/**
 * @inline
 */

var getRectanglePath = (x, y, width, height, radius) => {
  var roundedWidth = round(width);
  var roundedHeight = round(height);
  var maxRadius = Math.min(Math.abs(roundedWidth) / 2, Math.abs(roundedHeight) / 2);
  var ySign = roundedHeight >= 0 ? 1 : -1;
  var xSign = roundedWidth >= 0 ? 1 : -1;
  var clockWise = roundedHeight >= 0 && roundedWidth >= 0 || roundedHeight < 0 && roundedWidth < 0 ? 1 : 0;
  var path;
  if (maxRadius > 0 && Array.isArray(radius)) {
    var newRadius = [0, 0, 0, 0];
    for (var i = 0, len = 4; i < len; i++) {
      var _radius$i;
      var r = (_radius$i = radius[i]) !== null && _radius$i !== void 0 ? _radius$i : 0;
      newRadius[i] = r > maxRadius ? maxRadius : r;
    }
    path = roundTemplateLiteral(_templateObject || (_templateObject = _taggedTemplateLiteral(["M", ",", ""])), x, y + ySign * newRadius[0]);
    if (newRadius[0] > 0) {
      path += roundTemplateLiteral(_templateObject2 || (_templateObject2 = _taggedTemplateLiteral(["A ", ",", ",0,0,", ",", ",", ""])), newRadius[0], newRadius[0], clockWise, x + xSign * newRadius[0], y);
    }
    path += roundTemplateLiteral(_templateObject3 || (_templateObject3 = _taggedTemplateLiteral(["L ", ",", ""])), x + width - xSign * newRadius[1], y);
    if (newRadius[1] > 0) {
      path += roundTemplateLiteral(_templateObject4 || (_templateObject4 = _taggedTemplateLiteral(["A ", ",", ",0,0,", ",\n        ", ",", ""])), newRadius[1], newRadius[1], clockWise, x + width, y + ySign * newRadius[1]);
    }
    path += roundTemplateLiteral(_templateObject5 || (_templateObject5 = _taggedTemplateLiteral(["L ", ",", ""])), x + width, y + height - ySign * newRadius[2]);
    if (newRadius[2] > 0) {
      path += roundTemplateLiteral(_templateObject6 || (_templateObject6 = _taggedTemplateLiteral(["A ", ",", ",0,0,", ",\n        ", ",", ""])), newRadius[2], newRadius[2], clockWise, x + width - xSign * newRadius[2], y + height);
    }
    path += roundTemplateLiteral(_templateObject7 || (_templateObject7 = _taggedTemplateLiteral(["L ", ",", ""])), x + xSign * newRadius[3], y + height);
    if (newRadius[3] > 0) {
      path += roundTemplateLiteral(_templateObject8 || (_templateObject8 = _taggedTemplateLiteral(["A ", ",", ",0,0,", ",\n        ", ",", ""])), newRadius[3], newRadius[3], clockWise, x, y + height - ySign * newRadius[3]);
    }
    path += 'Z';
  } else if (maxRadius > 0 && radius === +radius && radius > 0) {
    var _newRadius = Math.min(maxRadius, radius);
    path = roundTemplateLiteral(_templateObject9 || (_templateObject9 = _taggedTemplateLiteral(["M ", ",", "\n            A ", ",", ",0,0,", ",", ",", "\n            L ", ",", "\n            A ", ",", ",0,0,", ",", ",", "\n            L ", ",", "\n            A ", ",", ",0,0,", ",", ",", "\n            L ", ",", "\n            A ", ",", ",0,0,", ",", ",", " Z"])), x, y + ySign * _newRadius, _newRadius, _newRadius, clockWise, x + xSign * _newRadius, y, x + width - xSign * _newRadius, y, _newRadius, _newRadius, clockWise, x + width, y + ySign * _newRadius, x + width, y + height - ySign * _newRadius, _newRadius, _newRadius, clockWise, x + width - xSign * _newRadius, y + height, x + xSign * _newRadius, y + height, _newRadius, _newRadius, clockWise, x, y + height - ySign * _newRadius);
  } else {
    path = roundTemplateLiteral(_templateObject0 || (_templateObject0 = _taggedTemplateLiteral(["M ", ",", " h ", " v ", " h ", " Z"])), x, y, width, height, -width);
  }
  return path;
};
export var defaultRectangleProps = {
  x: 0,
  y: 0,
  width: 0,
  height: 0,
  radius: 0,
  isAnimationActive: false,
  isUpdateAnimationActive: false,
  animationBegin: 0,
  animationDuration: 1500,
  animationEasing: 'ease'
};

/**
 * Renders a rectangle element. Unlike the {@link https://developer.mozilla.org/en-US/docs/Web/SVG/Reference/Element/rect rect SVG element}, this component supports rounded corners
 * and animation.
 *
 * This component accepts X and Y coordinates in pixels.
 * If you need to position the rectangle based on your chart's data,
 * consider using the {@link ReferenceArea} component instead.
 *
 * @param rectangleProps
 * @constructor
 */
export var Rectangle = rectangleProps => {
  var props = resolveDefaultProps(rectangleProps, defaultRectangleProps);
  var pathRef = useRef(null);
  var [totalLength, setTotalLength] = useState(-1);
  useEffect(() => {
    if (pathRef.current && pathRef.current.getTotalLength) {
      try {
        var pathTotalLength = pathRef.current.getTotalLength();
        if (pathTotalLength) {
          setTotalLength(pathTotalLength);
        }
      } catch (_unused) {
        // calculate total length error
      }
    }
  }, []);
  var {
    x,
    y,
    width,
    height,
    radius,
    className
  } = props;
  var {
    animationEasing,
    animationDuration,
    animationBegin,
    isAnimationActive,
    isUpdateAnimationActive
  } = props;
  var prevWidthRef = useRef(width);
  var prevHeightRef = useRef(height);
  var prevXRef = useRef(x);
  var prevYRef = useRef(y);
  var animationIdInput = useMemo(() => ({
    x,
    y,
    width,
    height,
    radius
  }), [x, y, width, height, radius]);
  var animationId = useAnimationId(animationIdInput, 'rectangle-');
  if (x !== +x || y !== +y || width !== +width || height !== +height || width === 0 || height === 0) {
    return null;
  }
  var layerClass = clsx('recharts-rectangle', className);
  if (!isUpdateAnimationActive) {
    var _svgPropertiesAndEven = svgPropertiesAndEvents(props),
      {
        radius: _
      } = _svgPropertiesAndEven,
      otherPathProps = _objectWithoutProperties(_svgPropertiesAndEven, _excluded);
    return /*#__PURE__*/React.createElement("path", _extends({}, otherPathProps, {
      x: round(x),
      y: round(y),
      width: round(width),
      height: round(height),
      radius: typeof radius === 'number' ? radius : undefined,
      className: layerClass,
      d: getRectanglePath(x, y, width, height, radius)
    }));
  }
  var prevWidth = prevWidthRef.current;
  var prevHeight = prevHeightRef.current;
  var prevX = prevXRef.current;
  var prevY = prevYRef.current;
  var from = "0px ".concat(totalLength === -1 ? 1 : totalLength, "px");
  var to = "".concat(totalLength, "px ").concat(totalLength, "px");
  var transition = getTransitionVal(['strokeDasharray'], animationDuration, typeof animationEasing === 'string' ? animationEasing : defaultRectangleProps.animationEasing);
  return /*#__PURE__*/React.createElement(JavascriptAnimate, {
    animationId: animationId,
    key: animationId,
    canBegin: totalLength > 0,
    duration: animationDuration,
    easing: animationEasing,
    isActive: isUpdateAnimationActive,
    begin: animationBegin
  }, t => {
    var currWidth = interpolate(prevWidth, width, t);
    var currHeight = interpolate(prevHeight, height, t);
    var currX = interpolate(prevX, x, t);
    var currY = interpolate(prevY, y, t);
    if (pathRef.current) {
      prevWidthRef.current = currWidth;
      prevHeightRef.current = currHeight;
      prevXRef.current = currX;
      prevYRef.current = currY;
    }
    var animationStyle;
    if (!isAnimationActive) {
      animationStyle = {
        strokeDasharray: to
      };
    } else if (t > 0) {
      animationStyle = {
        transition,
        strokeDasharray: to
      };
    } else {
      animationStyle = {
        strokeDasharray: from
      };
    }
    var _svgPropertiesAndEven2 = svgPropertiesAndEvents(props),
      {
        radius: _
      } = _svgPropertiesAndEven2,
      otherPathProps = _objectWithoutProperties(_svgPropertiesAndEven2, _excluded2);
    return /*#__PURE__*/React.createElement("path", _extends({}, otherPathProps, {
      radius: typeof radius === 'number' ? radius : undefined,
      className: layerClass,
      d: getRectanglePath(currX, currY, currWidth, currHeight, radius),
      ref: pathRef,
      style: _objectSpread(_objectSpread({}, animationStyle), props.style)
    }));
  });
};