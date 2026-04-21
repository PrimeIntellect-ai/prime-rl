import { clsx } from 'clsx';
import { isNumber } from '../DataUtils';
var CSS_CLASS_PREFIX = 'recharts-tooltip-wrapper';
var TOOLTIP_HIDDEN = {
  visibility: 'hidden'
};
export function getTooltipCSSClassName(_ref) {
  var {
    coordinate,
    translateX,
    translateY
  } = _ref;
  return clsx(CSS_CLASS_PREFIX, {
    ["".concat(CSS_CLASS_PREFIX, "-right")]: isNumber(translateX) && coordinate && isNumber(coordinate.x) && translateX >= coordinate.x,
    ["".concat(CSS_CLASS_PREFIX, "-left")]: isNumber(translateX) && coordinate && isNumber(coordinate.x) && translateX < coordinate.x,
    ["".concat(CSS_CLASS_PREFIX, "-bottom")]: isNumber(translateY) && coordinate && isNumber(coordinate.y) && translateY >= coordinate.y,
    ["".concat(CSS_CLASS_PREFIX, "-top")]: isNumber(translateY) && coordinate && isNumber(coordinate.y) && translateY < coordinate.y
  });
}
export function getTooltipTranslateXY(_ref2) {
  var {
    allowEscapeViewBox,
    coordinate,
    key,
    offset,
    position,
    reverseDirection,
    tooltipDimension,
    viewBox,
    viewBoxDimension
  } = _ref2;
  if (position && isNumber(position[key])) {
    return position[key];
  }
  var negative = coordinate[key] - tooltipDimension - (offset > 0 ? offset : 0);
  var positive = coordinate[key] + offset;
  if (allowEscapeViewBox[key]) {
    return reverseDirection[key] ? negative : positive;
  }
  var viewBoxKey = viewBox[key];
  if (viewBoxKey == null) {
    return 0;
  }
  if (reverseDirection[key]) {
    var _tooltipBoundary = negative;
    var _viewBoxBoundary = viewBoxKey;
    if (_tooltipBoundary < _viewBoxBoundary) {
      return Math.max(positive, viewBoxKey);
    }
    return Math.max(negative, viewBoxKey);
  }
  if (viewBoxDimension == null) {
    return 0;
  }
  var tooltipBoundary = positive + tooltipDimension;
  var viewBoxBoundary = viewBoxKey + viewBoxDimension;
  if (tooltipBoundary > viewBoxBoundary) {
    return Math.max(negative, viewBoxKey);
  }
  return Math.max(positive, viewBoxKey);
}
export function getTransformStyle(_ref3) {
  var {
    translateX,
    translateY,
    useTranslate3d
  } = _ref3;
  return {
    transform: useTranslate3d ? "translate3d(".concat(translateX, "px, ").concat(translateY, "px, 0)") : "translate(".concat(translateX, "px, ").concat(translateY, "px)")
  };
}
export function getTooltipTranslate(_ref4) {
  var {
    allowEscapeViewBox,
    coordinate,
    offsetTop,
    offsetLeft,
    position,
    reverseDirection,
    tooltipBox,
    useTranslate3d,
    viewBox
  } = _ref4;
  var cssProperties, translateX, translateY;
  if (tooltipBox.height > 0 && tooltipBox.width > 0 && coordinate) {
    translateX = getTooltipTranslateXY({
      allowEscapeViewBox,
      coordinate,
      key: 'x',
      offset: offsetLeft,
      position,
      reverseDirection,
      tooltipDimension: tooltipBox.width,
      viewBox,
      viewBoxDimension: viewBox.width
    });
    translateY = getTooltipTranslateXY({
      allowEscapeViewBox,
      coordinate,
      key: 'y',
      offset: offsetTop,
      position,
      reverseDirection,
      tooltipDimension: tooltipBox.height,
      viewBox,
      viewBoxDimension: viewBox.height
    });
    cssProperties = getTransformStyle({
      translateX,
      translateY,
      useTranslate3d
    });
  } else {
    cssProperties = TOOLTIP_HIDDEN;
  }
  return {
    cssProperties,
    cssClasses: getTooltipCSSClassName({
      translateX,
      translateY,
      coordinate
    })
  };
}