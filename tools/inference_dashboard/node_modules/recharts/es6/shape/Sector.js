var _templateObject, _templateObject2, _templateObject3, _templateObject4, _templateObject5, _templateObject6, _templateObject7;
function _extends() { return _extends = Object.assign ? Object.assign.bind() : function (n) { for (var e = 1; e < arguments.length; e++) { var t = arguments[e]; for (var r in t) ({}).hasOwnProperty.call(t, r) && (n[r] = t[r]); } return n; }, _extends.apply(null, arguments); }
function _taggedTemplateLiteral(e, t) { return t || (t = e.slice(0)), Object.freeze(Object.defineProperties(e, { raw: { value: Object.freeze(t) } })); }
import * as React from 'react';
import { clsx } from 'clsx';
import { polarToCartesian, RADIAN } from '../util/PolarUtils';
import { getPercentValue, mathSign } from '../util/DataUtils';
import { resolveDefaultProps } from '../util/resolveDefaultProps';
import { svgPropertiesAndEvents } from '../util/svgPropertiesAndEvents';
import { roundTemplateLiteral } from '../util/round';
var getDeltaAngle = (startAngle, endAngle) => {
  var sign = mathSign(endAngle - startAngle);
  var deltaAngle = Math.min(Math.abs(endAngle - startAngle), 359.999);
  return sign * deltaAngle;
};
var getTangentCircle = _ref => {
  var {
    cx,
    cy,
    radius,
    angle,
    sign,
    isExternal,
    cornerRadius,
    cornerIsExternal
  } = _ref;
  var centerRadius = cornerRadius * (isExternal ? 1 : -1) + radius;
  var theta = Math.asin(cornerRadius / centerRadius) / RADIAN;
  var centerAngle = cornerIsExternal ? angle : angle + sign * theta;
  var center = polarToCartesian(cx, cy, centerRadius, centerAngle);
  // The coordinate of point which is tangent to the circle
  var circleTangency = polarToCartesian(cx, cy, radius, centerAngle);
  // The coordinate of point which is tangent to the radius line
  var lineTangencyAngle = cornerIsExternal ? angle - sign * theta : angle;
  var lineTangency = polarToCartesian(cx, cy, centerRadius * Math.cos(theta * RADIAN), lineTangencyAngle);
  return {
    center,
    circleTangency,
    lineTangency,
    theta
  };
};
var getSectorPath = _ref2 => {
  var {
    cx,
    cy,
    innerRadius,
    outerRadius,
    startAngle,
    endAngle
  } = _ref2;
  var angle = getDeltaAngle(startAngle, endAngle);

  // When the angle of sector equals to 360, star point and end point coincide
  var tempEndAngle = startAngle + angle;
  var outerStartPoint = polarToCartesian(cx, cy, outerRadius, startAngle);
  var outerEndPoint = polarToCartesian(cx, cy, outerRadius, tempEndAngle);
  var path = roundTemplateLiteral(_templateObject || (_templateObject = _taggedTemplateLiteral(["M ", ",", "\n    A ", ",", ",0,\n    ", ",", ",\n    ", ",", "\n  "])), outerStartPoint.x, outerStartPoint.y, outerRadius, outerRadius, +(Math.abs(angle) > 180), +(startAngle > tempEndAngle), outerEndPoint.x, outerEndPoint.y);
  if (innerRadius > 0) {
    var innerStartPoint = polarToCartesian(cx, cy, innerRadius, startAngle);
    var innerEndPoint = polarToCartesian(cx, cy, innerRadius, tempEndAngle);
    path += roundTemplateLiteral(_templateObject2 || (_templateObject2 = _taggedTemplateLiteral(["L ", ",", "\n            A ", ",", ",0,\n            ", ",", ",\n            ", ",", " Z"])), innerEndPoint.x, innerEndPoint.y, innerRadius, innerRadius, +(Math.abs(angle) > 180), +(startAngle <= tempEndAngle), innerStartPoint.x, innerStartPoint.y);
  } else {
    path += roundTemplateLiteral(_templateObject3 || (_templateObject3 = _taggedTemplateLiteral(["L ", ",", " Z"])), cx, cy);
  }
  return path;
};
var getSectorWithCorner = _ref3 => {
  var {
    cx,
    cy,
    innerRadius,
    outerRadius,
    cornerRadius,
    forceCornerRadius,
    cornerIsExternal,
    startAngle,
    endAngle
  } = _ref3;
  var sign = mathSign(endAngle - startAngle);
  var {
    circleTangency: soct,
    lineTangency: solt,
    theta: sot
  } = getTangentCircle({
    cx,
    cy,
    radius: outerRadius,
    angle: startAngle,
    sign,
    cornerRadius,
    cornerIsExternal
  });
  var {
    circleTangency: eoct,
    lineTangency: eolt,
    theta: eot
  } = getTangentCircle({
    cx,
    cy,
    radius: outerRadius,
    angle: endAngle,
    sign: -sign,
    cornerRadius,
    cornerIsExternal
  });
  var outerArcAngle = cornerIsExternal ? Math.abs(startAngle - endAngle) : Math.abs(startAngle - endAngle) - sot - eot;
  if (outerArcAngle < 0) {
    if (forceCornerRadius) {
      return roundTemplateLiteral(_templateObject4 || (_templateObject4 = _taggedTemplateLiteral(["M ", ",", "\n        a", ",", ",0,0,1,", ",0\n        a", ",", ",0,0,1,", ",0\n      "])), solt.x, solt.y, cornerRadius, cornerRadius, cornerRadius * 2, cornerRadius, cornerRadius, -cornerRadius * 2);
    }
    return getSectorPath({
      cx,
      cy,
      innerRadius,
      outerRadius,
      startAngle,
      endAngle
    });
  }
  var path = roundTemplateLiteral(_templateObject5 || (_templateObject5 = _taggedTemplateLiteral(["M ", ",", "\n    A", ",", ",0,0,", ",", ",", "\n    A", ",", ",0,", ",", ",", ",", "\n    A", ",", ",0,0,", ",", ",", "\n  "])), solt.x, solt.y, cornerRadius, cornerRadius, +(sign < 0), soct.x, soct.y, outerRadius, outerRadius, +(outerArcAngle > 180), +(sign < 0), eoct.x, eoct.y, cornerRadius, cornerRadius, +(sign < 0), eolt.x, eolt.y);
  if (innerRadius > 0) {
    var {
      circleTangency: sict,
      lineTangency: silt,
      theta: sit
    } = getTangentCircle({
      cx,
      cy,
      radius: innerRadius,
      angle: startAngle,
      sign,
      isExternal: true,
      cornerRadius,
      cornerIsExternal
    });
    var {
      circleTangency: eict,
      lineTangency: eilt,
      theta: eit
    } = getTangentCircle({
      cx,
      cy,
      radius: innerRadius,
      angle: endAngle,
      sign: -sign,
      isExternal: true,
      cornerRadius,
      cornerIsExternal
    });
    var innerArcAngle = cornerIsExternal ? Math.abs(startAngle - endAngle) : Math.abs(startAngle - endAngle) - sit - eit;
    if (innerArcAngle < 0 && cornerRadius === 0) {
      return "".concat(path, "L").concat(cx, ",").concat(cy, "Z");
    }
    path += roundTemplateLiteral(_templateObject6 || (_templateObject6 = _taggedTemplateLiteral(["L", ",", "\n      A", ",", ",0,0,", ",", ",", "\n      A", ",", ",0,", ",", ",", ",", "\n      A", ",", ",0,0,", ",", ",", "Z"])), eilt.x, eilt.y, cornerRadius, cornerRadius, +(sign < 0), eict.x, eict.y, innerRadius, innerRadius, +(innerArcAngle > 180), +(sign > 0), sict.x, sict.y, cornerRadius, cornerRadius, +(sign < 0), silt.x, silt.y);
  } else {
    path += roundTemplateLiteral(_templateObject7 || (_templateObject7 = _taggedTemplateLiteral(["L", ",", "Z"])), cx, cy);
  }
  return path;
};

/**
 * SVG cx, cy are `string | number | undefined`, but internally we use `number` so let's
 * override the types here.
 */

export var defaultSectorProps = {
  cx: 0,
  cy: 0,
  innerRadius: 0,
  outerRadius: 0,
  startAngle: 0,
  endAngle: 0,
  cornerRadius: 0,
  forceCornerRadius: false,
  cornerIsExternal: false
};
export var Sector = sectorProps => {
  var props = resolveDefaultProps(sectorProps, defaultSectorProps);
  var {
    cx,
    cy,
    innerRadius,
    outerRadius,
    cornerRadius,
    forceCornerRadius,
    cornerIsExternal,
    startAngle,
    endAngle,
    className
  } = props;
  if (outerRadius < innerRadius || startAngle === endAngle) {
    return null;
  }
  var layerClass = clsx('recharts-sector', className);
  var deltaRadius = outerRadius - innerRadius;
  var cr = getPercentValue(cornerRadius, deltaRadius, 0, true);
  var path;
  if (cr > 0 && Math.abs(startAngle - endAngle) < 360) {
    path = getSectorWithCorner({
      cx,
      cy,
      innerRadius,
      outerRadius,
      cornerRadius: Math.min(cr, deltaRadius / 2),
      forceCornerRadius,
      cornerIsExternal,
      startAngle,
      endAngle
    });
  } else {
    path = getSectorPath({
      cx,
      cy,
      innerRadius,
      outerRadius,
      startAngle,
      endAngle
    });
  }
  return /*#__PURE__*/React.createElement("path", _extends({}, svgPropertiesAndEvents(props), {
    className: layerClass,
    d: path
  }));
};