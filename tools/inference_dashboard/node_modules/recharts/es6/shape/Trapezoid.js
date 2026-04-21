var _templateObject, _templateObject2, _templateObject3, _templateObject4, _templateObject5;
function ownKeys(e, r) { var t = Object.keys(e); if (Object.getOwnPropertySymbols) { var o = Object.getOwnPropertySymbols(e); r && (o = o.filter(function (r) { return Object.getOwnPropertyDescriptor(e, r).enumerable; })), t.push.apply(t, o); } return t; }
function _objectSpread(e) { for (var r = 1; r < arguments.length; r++) { var t = null != arguments[r] ? arguments[r] : {}; r % 2 ? ownKeys(Object(t), !0).forEach(function (r) { _defineProperty(e, r, t[r]); }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(t)) : ownKeys(Object(t)).forEach(function (r) { Object.defineProperty(e, r, Object.getOwnPropertyDescriptor(t, r)); }); } return e; }
function _defineProperty(e, r, t) { return (r = _toPropertyKey(r)) in e ? Object.defineProperty(e, r, { value: t, enumerable: !0, configurable: !0, writable: !0 }) : e[r] = t, e; }
function _toPropertyKey(t) { var i = _toPrimitive(t, "string"); return "symbol" == typeof i ? i : i + ""; }
function _toPrimitive(t, r) { if ("object" != typeof t || !t) return t; var e = t[Symbol.toPrimitive]; if (void 0 !== e) { var i = e.call(t, r || "default"); if ("object" != typeof i) return i; throw new TypeError("@@toPrimitive must return a primitive value."); } return ("string" === r ? String : Number)(t); }
function _extends() { return _extends = Object.assign ? Object.assign.bind() : function (n) { for (var e = 1; e < arguments.length; e++) { var t = arguments[e]; for (var r in t) ({}).hasOwnProperty.call(t, r) && (n[r] = t[r]); } return n; }, _extends.apply(null, arguments); }
function _taggedTemplateLiteral(e, t) { return t || (t = e.slice(0)), Object.freeze(Object.defineProperties(e, { raw: { value: Object.freeze(t) } })); }
/**
 * @fileOverview Rectangle
 */
import * as React from 'react';
import { useEffect, useRef, useState } from 'react';
import { clsx } from 'clsx';
import { resolveDefaultProps } from '../util/resolveDefaultProps';
import { JavascriptAnimate } from '../animation/JavascriptAnimate';
import { useAnimationId } from '../util/useAnimationId';
import { interpolate } from '../util/DataUtils';
import { getTransitionVal } from '../animation/util';
import { svgPropertiesAndEvents } from '../util/svgPropertiesAndEvents';
import { roundTemplateLiteral } from '../util/round';
var getTrapezoidPath = (x, y, upperWidth, lowerWidth, height) => {
  var widthGap = upperWidth - lowerWidth;
  var path;
  path = roundTemplateLiteral(_templateObject || (_templateObject = _taggedTemplateLiteral(["M ", ",", ""])), x, y);
  path += roundTemplateLiteral(_templateObject2 || (_templateObject2 = _taggedTemplateLiteral(["L ", ",", ""])), x + upperWidth, y);
  path += roundTemplateLiteral(_templateObject3 || (_templateObject3 = _taggedTemplateLiteral(["L ", ",", ""])), x + upperWidth - widthGap / 2, y + height);
  path += roundTemplateLiteral(_templateObject4 || (_templateObject4 = _taggedTemplateLiteral(["L ", ",", ""])), x + upperWidth - widthGap / 2 - lowerWidth, y + height);
  path += roundTemplateLiteral(_templateObject5 || (_templateObject5 = _taggedTemplateLiteral(["L ", ",", " Z"])), x, y);
  return path;
};
export var defaultTrapezoidProps = {
  x: 0,
  y: 0,
  upperWidth: 0,
  lowerWidth: 0,
  height: 0,
  isUpdateAnimationActive: false,
  animationBegin: 0,
  animationDuration: 1500,
  animationEasing: 'ease'
};
export var Trapezoid = outsideProps => {
  var trapezoidProps = resolveDefaultProps(outsideProps, defaultTrapezoidProps);
  var {
    x,
    y,
    upperWidth,
    lowerWidth,
    height,
    className
  } = trapezoidProps;
  var {
    animationEasing,
    animationDuration,
    animationBegin,
    isUpdateAnimationActive
  } = trapezoidProps;
  var pathRef = useRef(null);
  var [totalLength, setTotalLength] = useState(-1);
  var prevUpperWidthRef = useRef(upperWidth);
  var prevLowerWidthRef = useRef(lowerWidth);
  var prevHeightRef = useRef(height);
  var prevXRef = useRef(x);
  var prevYRef = useRef(y);
  var animationId = useAnimationId(outsideProps, 'trapezoid-');
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
  if (x !== +x || y !== +y || upperWidth !== +upperWidth || lowerWidth !== +lowerWidth || height !== +height || upperWidth === 0 && lowerWidth === 0 || height === 0) {
    return null;
  }
  var layerClass = clsx('recharts-trapezoid', className);
  if (!isUpdateAnimationActive) {
    return /*#__PURE__*/React.createElement("g", null, /*#__PURE__*/React.createElement("path", _extends({}, svgPropertiesAndEvents(trapezoidProps), {
      className: layerClass,
      d: getTrapezoidPath(x, y, upperWidth, lowerWidth, height)
    })));
  }
  var prevUpperWidth = prevUpperWidthRef.current;
  var prevLowerWidth = prevLowerWidthRef.current;
  var prevHeight = prevHeightRef.current;
  var prevX = prevXRef.current;
  var prevY = prevYRef.current;
  var from = "0px ".concat(totalLength === -1 ? 1 : totalLength, "px");
  var to = "".concat(totalLength, "px ").concat(totalLength, "px");
  var transition = getTransitionVal(['strokeDasharray'], animationDuration, animationEasing);
  return /*#__PURE__*/React.createElement(JavascriptAnimate, {
    animationId: animationId,
    key: animationId,
    canBegin: totalLength > 0,
    duration: animationDuration,
    easing: animationEasing,
    isActive: isUpdateAnimationActive,
    begin: animationBegin
  }, t => {
    var currUpperWidth = interpolate(prevUpperWidth, upperWidth, t);
    var currLowerWidth = interpolate(prevLowerWidth, lowerWidth, t);
    var currHeight = interpolate(prevHeight, height, t);
    var currX = interpolate(prevX, x, t);
    var currY = interpolate(prevY, y, t);
    if (pathRef.current) {
      prevUpperWidthRef.current = currUpperWidth;
      prevLowerWidthRef.current = currLowerWidth;
      prevHeightRef.current = currHeight;
      prevXRef.current = currX;
      prevYRef.current = currY;
    }
    var animationStyle = t > 0 ? {
      transition,
      strokeDasharray: to
    } : {
      strokeDasharray: from
    };
    return /*#__PURE__*/React.createElement("path", _extends({}, svgPropertiesAndEvents(trapezoidProps), {
      className: layerClass,
      d: getTrapezoidPath(currX, currY, currUpperWidth, currLowerWidth, currHeight),
      ref: pathRef,
      style: _objectSpread(_objectSpread({}, animationStyle), trapezoidProps.style)
    }));
  });
};