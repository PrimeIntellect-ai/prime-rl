var _excluded = ["id"],
  _excluded2 = ["type", "layout", "connectNulls", "needClip", "shape"],
  _excluded3 = ["activeDot", "animateNewValues", "animationBegin", "animationDuration", "animationEasing", "connectNulls", "dot", "hide", "isAnimationActive", "label", "legendType", "xAxisId", "yAxisId", "id"];
function _extends() { return _extends = Object.assign ? Object.assign.bind() : function (n) { for (var e = 1; e < arguments.length; e++) { var t = arguments[e]; for (var r in t) ({}).hasOwnProperty.call(t, r) && (n[r] = t[r]); } return n; }, _extends.apply(null, arguments); }
function ownKeys(e, r) { var t = Object.keys(e); if (Object.getOwnPropertySymbols) { var o = Object.getOwnPropertySymbols(e); r && (o = o.filter(function (r) { return Object.getOwnPropertyDescriptor(e, r).enumerable; })), t.push.apply(t, o); } return t; }
function _objectSpread(e) { for (var r = 1; r < arguments.length; r++) { var t = null != arguments[r] ? arguments[r] : {}; r % 2 ? ownKeys(Object(t), !0).forEach(function (r) { _defineProperty(e, r, t[r]); }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(t)) : ownKeys(Object(t)).forEach(function (r) { Object.defineProperty(e, r, Object.getOwnPropertyDescriptor(t, r)); }); } return e; }
function _defineProperty(e, r, t) { return (r = _toPropertyKey(r)) in e ? Object.defineProperty(e, r, { value: t, enumerable: !0, configurable: !0, writable: !0 }) : e[r] = t, e; }
function _toPropertyKey(t) { var i = _toPrimitive(t, "string"); return "symbol" == typeof i ? i : i + ""; }
function _toPrimitive(t, r) { if ("object" != typeof t || !t) return t; var e = t[Symbol.toPrimitive]; if (void 0 !== e) { var i = e.call(t, r || "default"); if ("object" != typeof i) return i; throw new TypeError("@@toPrimitive must return a primitive value."); } return ("string" === r ? String : Number)(t); }
function _objectWithoutProperties(e, t) { if (null == e) return {}; var o, r, i = _objectWithoutPropertiesLoose(e, t); if (Object.getOwnPropertySymbols) { var n = Object.getOwnPropertySymbols(e); for (r = 0; r < n.length; r++) o = n[r], -1 === t.indexOf(o) && {}.propertyIsEnumerable.call(e, o) && (i[o] = e[o]); } return i; }
function _objectWithoutPropertiesLoose(r, e) { if (null == r) return {}; var t = {}; for (var n in r) if ({}.hasOwnProperty.call(r, n)) { if (-1 !== e.indexOf(n)) continue; t[n] = r[n]; } return t; }
import * as React from 'react';
import { Component, useCallback, useMemo, useRef, useState } from 'react';
import { clsx } from 'clsx';
import { Layer } from '../container/Layer';
import { CartesianLabelListContextProvider, LabelListFromLabelProp } from '../component/LabelList';
import { Dots } from '../component/Dots';
import { interpolate, isNullish, noop } from '../util/DataUtils';
import { isClipDot } from '../util/ReactUtils';
import { getCateCoordinateOfLine, getTooltipNameProp, getValueByDataKey } from '../util/ChartUtils';
import { ActivePoints } from '../component/ActivePoints';
import { SetTooltipEntrySettings } from '../state/SetTooltipEntrySettings';
import { SetErrorBarContext } from '../context/ErrorBarContext';
import { GraphicalItemClipPath, useNeedsClip } from './GraphicalItemClipPath';
import { useChartLayout } from '../context/chartLayoutContext';
import { useIsPanorama } from '../context/PanoramaContext';
import { selectLinePoints } from '../state/selectors/lineSelectors';
import { useAppSelector } from '../state/hooks';
import { SetLegendPayload } from '../state/SetLegendPayload';
import { useAnimationId } from '../util/useAnimationId';
import { resolveDefaultProps } from '../util/resolveDefaultProps';
import { usePlotArea } from '../hooks';
import { RegisterGraphicalItemId } from '../context/RegisterGraphicalItemId';
import { SetCartesianGraphicalItem } from '../state/SetGraphicalItem';
import { svgPropertiesNoEvents } from '../util/svgPropertiesNoEvents';
import { JavascriptAnimate } from '../animation/JavascriptAnimate';
import { svgPropertiesAndEvents } from '../util/svgPropertiesAndEvents';
import { getRadiusAndStrokeWidthFromDot } from '../util/getRadiusAndStrokeWidthFromDot';
import { Shape } from '../util/ActiveShapeUtils';
import { ZIndexLayer } from '../zIndex/ZIndexLayer';
import { DefaultZIndexes } from '../zIndex/DefaultZIndexes';
import { propsAreEqual } from '../util/propsAreEqual';

/**
 * Internal props, combination of external props + defaultProps + private Recharts state
 */

/**
 * External props, intended for end users to fill in
 */

/**
 * Because of naming conflict, we are forced to ignore certain (valid) SVG attributes.
 */

var computeLegendPayloadFromAreaData = props => {
  var {
    dataKey,
    name,
    stroke,
    legendType,
    hide
  } = props;
  return [{
    inactive: hide,
    dataKey,
    type: legendType,
    color: stroke,
    value: getTooltipNameProp(name, dataKey),
    payload: props
  }];
};
var SetLineTooltipEntrySettings = /*#__PURE__*/React.memo(_ref => {
  var {
    dataKey,
    data,
    stroke,
    strokeWidth,
    fill,
    name,
    hide,
    unit,
    tooltipType,
    id
  } = _ref;
  var tooltipEntrySettings = {
    dataDefinedOnItem: data,
    getPosition: noop,
    settings: {
      stroke,
      strokeWidth,
      fill,
      dataKey,
      nameKey: undefined,
      name: getTooltipNameProp(name, dataKey),
      hide,
      type: tooltipType,
      color: stroke,
      unit,
      graphicalItemId: id
    }
  };
  return /*#__PURE__*/React.createElement(SetTooltipEntrySettings, {
    tooltipEntrySettings: tooltipEntrySettings
  });
});

/**
 * Generates a simple stroke-dasharray string for animating a line draw effect.
 *
 * Uses `totalLength` as the gap (instead of `totalLength - length`) to prevent a floating-point
 * precision artifact: when fractional dash and gap values are serialized to a string attribute
 * and re-parsed by the SVG renderer, their sum can differ from the actual path length by a ULP,
 * causing the dasharray pattern to repeat and render a phantom dot at the path endpoint
 * with round or square strokeLinecap.
 *
 * @param totalLength The total length of the SVG path
 * @param length The currently visible portion of the path
 * @returns A stroke-dasharray string like "50px 200px"
 */
var generateSimpleStrokeDasharray = (totalLength, length) => {
  return "".concat(length, "px ").concat(totalLength, "px");
};

/**
 * Repeats a dash pattern array a given number of times.
 *
 * If the input array has an odd length, a trailing `0` is appended to make it even
 * before repeating, because SVG stroke-dasharray patterns must have an even number
 * of values to cycle correctly between dash and gap segments.
 *
 * @param lines Array of dash/gap lengths to repeat
 * @param count Number of times to repeat the pattern
 * @returns A new array with the pattern repeated `count` times
 */
function repeat(lines, count) {
  var linesUnit = lines.length % 2 !== 0 ? [...lines, 0] : lines;
  var result = [];
  for (var i = 0; i < count; ++i) {
    result.push(...linesUnit);
  }
  return result;
}

/**
 * Computes a stroke-dasharray string for animating a custom-dashed line draw effect.
 *
 * Given a user-specified dash pattern (e.g. `"7,3"`), this function builds a dasharray
 * that reveals exactly `length` pixels of that pattern, followed by a gap of `totalLength`
 * to hide the remainder of the path.
 *
 * Like {@link generateSimpleStrokeDasharray}, the trailing gap uses `totalLength` rather than
 * `totalLength - length` to avoid floating-point precision artifacts with round/square strokeLinecap.
 *
 * @param length The currently visible portion of the path
 * @param totalLength The total length of the SVG path
 * @param lines The user-specified dash pattern as an array of numbers (e.g. [7, 3])
 * @returns A stroke-dasharray string incorporating the custom dash pattern
 */
var getStrokeDasharray = (length, totalLength, lines) => {
  var lineLength = lines.reduce((pre, next) => pre + next, 0);

  // if lineLength is 0 return the default when no strokeDasharray is provided
  if (!lineLength) {
    return generateSimpleStrokeDasharray(totalLength, length);
  }
  var count = Math.floor(length / lineLength);
  var remainLength = length % lineLength;
  var remainLines = [];
  for (var i = 0, sum = 0; i < lines.length; sum += (_lines$i = lines[i]) !== null && _lines$i !== void 0 ? _lines$i : 0, ++i) {
    var _lines$i;
    var lineValue = lines[i];
    if (lineValue != null && sum + lineValue > remainLength) {
      remainLines = [...lines.slice(0, i), remainLength - sum];
      break;
    }
  }
  var emptyLines = remainLines.length % 2 === 0 ? [0, totalLength] : [totalLength];
  return [...repeat(lines, count), ...remainLines, ...emptyLines].map(line => "".concat(line, "px")).join(', ');
};
function LineDotsWrapper(_ref2) {
  var {
    clipPathId,
    points,
    props
  } = _ref2;
  var {
    dot,
    dataKey,
    needClip
  } = props;

  /*
   * Exclude ID from the props passed to the Dots component
   * because then the ID would be applied to multiple dots, and it would no longer be unique.
   */
  var {
      id
    } = props,
    propsWithoutId = _objectWithoutProperties(props, _excluded);
  var lineProps = svgPropertiesNoEvents(propsWithoutId);
  return /*#__PURE__*/React.createElement(Dots, {
    points: points,
    dot: dot,
    className: "recharts-line-dots",
    dotClassName: "recharts-line-dot",
    dataKey: dataKey,
    baseProps: lineProps,
    needClip: needClip,
    clipPathId: clipPathId
  });
}
function LineLabelListProvider(_ref3) {
  var {
    showLabels,
    children,
    points
  } = _ref3;
  var labelListEntries = useMemo(() => {
    return points === null || points === void 0 ? void 0 : points.map(point => {
      var _point$x, _point$y;
      var viewBox = {
        x: (_point$x = point.x) !== null && _point$x !== void 0 ? _point$x : 0,
        y: (_point$y = point.y) !== null && _point$y !== void 0 ? _point$y : 0,
        width: 0,
        lowerWidth: 0,
        upperWidth: 0,
        height: 0
      };
      return _objectSpread(_objectSpread({}, viewBox), {}, {
        value: point.value,
        payload: point.payload,
        viewBox,
        /*
         * Line is not passing parentViewBox to the LabelList so the labels can escape - looks like a bug, should we pass parentViewBox?
         * Or should this just be the root chart viewBox?
         */
        parentViewBox: undefined,
        fill: undefined
      });
    });
  }, [points]);
  return /*#__PURE__*/React.createElement(CartesianLabelListContextProvider, {
    value: showLabels ? labelListEntries : undefined
  }, children);
}
function StaticCurve(_ref4) {
  var {
    clipPathId,
    pathRef,
    points,
    strokeDasharray,
    props
  } = _ref4;
  var {
      type,
      layout,
      connectNulls,
      needClip,
      shape
    } = props,
    others = _objectWithoutProperties(props, _excluded2);
  var curveProps = _objectSpread(_objectSpread({}, svgPropertiesAndEvents(others)), {}, {
    fill: 'none',
    className: 'recharts-line-curve',
    clipPath: needClip ? "url(#clipPath-".concat(clipPathId, ")") : undefined,
    points,
    type,
    layout,
    connectNulls,
    strokeDasharray: strokeDasharray !== null && strokeDasharray !== void 0 ? strokeDasharray : props.strokeDasharray
  });
  return /*#__PURE__*/React.createElement(React.Fragment, null, (points === null || points === void 0 ? void 0 : points.length) > 1 && /*#__PURE__*/React.createElement(Shape, _extends({
    shapeType: "curve",
    option: shape
  }, curveProps, {
    pathRef: pathRef
  })), /*#__PURE__*/React.createElement(LineDotsWrapper, {
    points: points,
    clipPathId: clipPathId,
    props: props
  }));
}
function getTotalLength(mainCurve) {
  try {
    return mainCurve && mainCurve.getTotalLength && mainCurve.getTotalLength() || 0;
  } catch (_unused) {
    return 0;
  }
}
function CurveWithAnimation(_ref5) {
  var {
    clipPathId,
    props,
    pathRef,
    previousPointsRef,
    longestAnimatedLengthRef
  } = _ref5;
  var {
    points,
    strokeDasharray,
    isAnimationActive,
    animationBegin,
    animationDuration,
    animationEasing,
    animateNewValues,
    width,
    height,
    onAnimationEnd,
    onAnimationStart
  } = props;
  var prevPoints = previousPointsRef.current;
  var animationId = useAnimationId(points, 'recharts-line-');
  var animationIdRef = useRef(animationId);
  var [isAnimating, setIsAnimating] = useState(false);
  var showLabels = !isAnimating;
  var handleAnimationEnd = useCallback(() => {
    if (typeof onAnimationEnd === 'function') {
      onAnimationEnd();
    }
    setIsAnimating(false);
  }, [onAnimationEnd]);
  var handleAnimationStart = useCallback(() => {
    if (typeof onAnimationStart === 'function') {
      onAnimationStart();
    }
    setIsAnimating(true);
  }, [onAnimationStart]);
  var totalLength = getTotalLength(pathRef.current);
  /*
   * Here we want to detect if the length animation has been interrupted.
   * For that we keep a reference to the furthest length that has been animated.
   *
   * And then, to keep things smooth, we add to it the current length that is being animated right now.
   *
   * If we did Math.max then it makes the length animation "pause" but we want to keep it smooth
   * so in case we have some "leftover" length from the previous animation we add it to the current length.
   *
   * This is not perfect because the animation changes speed due to easing. The default easing is 'ease' which is not linear
   * and makes it stand out. But it's good enough I suppose.
   * If we want to fix it then we need to keep track of multiple animations and their easing and timings.
   *
   * If you want to see this in action, try to change the dataKey of the line chart while the initial animation is running.
   * The Line begins with zero length and slowly grows to the full length. While this growth is in progress,
   * change the dataKey and the Line will continue growing from where it has grown so far.
   *
   * This is for the case when new animation triggers. When that happens we get new points, everything re-renders,
   * and we get fresh new state in this component and use the ref stored above.
   *
   * In case when we get render without new animation - for example when opacity changes, or color changes,
   * then the animationId remains the same, and we do not update the starting point.
   * See https://github.com/recharts/recharts/issues/6044
   */
  var startingPointRef = useRef(0);
  if (animationIdRef.current !== animationId) {
    startingPointRef.current = longestAnimatedLengthRef.current;
    animationIdRef.current = animationId;
  }
  var startingPoint = startingPointRef.current;
  return /*#__PURE__*/React.createElement(LineLabelListProvider, {
    points: points,
    showLabels: showLabels
  }, props.children, /*#__PURE__*/React.createElement(JavascriptAnimate, {
    animationId: animationId,
    begin: animationBegin,
    duration: animationDuration,
    isActive: isAnimationActive,
    easing: animationEasing,
    onAnimationEnd: handleAnimationEnd,
    onAnimationStart: handleAnimationStart,
    key: animationId
  }, t => {
    var lengthInterpolated = interpolate(startingPoint, totalLength + startingPoint, t);
    var curLength = Math.min(lengthInterpolated, totalLength);
    var currentStrokeDasharray;
    if (isAnimationActive) {
      if (strokeDasharray) {
        var lines = "".concat(strokeDasharray).split(/[,\s]+/gim).map(num => parseFloat(num));
        currentStrokeDasharray = getStrokeDasharray(curLength, totalLength, lines);
      } else {
        currentStrokeDasharray = generateSimpleStrokeDasharray(totalLength, curLength);
      }
    } else {
      currentStrokeDasharray = strokeDasharray == null ? undefined : String(strokeDasharray);
    }

    /*
     * Here it is important to wait a little bit with updating the previousPointsRef
     * before the animation has a time to initialize.
     * If we set the previous pointsRef immediately, we set it before the Legend height it calculated
     * and before pathRef is set.
     * If that happens, the Line will re-render again after Legend had reported its height
     * which will start a new animation with the previous points as the starting point
     * which gives the effect of the Line animating slightly upwards (where the animation distance equals the Legend height).
     * Waiting for t > 0 is indirect but good enough to ensure that the Legend height is calculated and animation works properly.
     *
     * Total length similarly is calculated from the pathRef. We should not update the previousPointsRef
     * before the pathRef is set, otherwise we will have a wrong total length.
     */
    if (t > 0 && totalLength > 0) {
      // eslint-disable-next-line no-param-reassign
      previousPointsRef.current = points;
      /*
       * totalLength is set from a ref and is not updated in the first tick of the animation.
       * It defaults to zero which is exactly what we want here because we want to grow from zero,
       * however the same happens when the data change.
       *
       * In that case we want to remember the previous length and continue from there, and only animate the shape.
       *
       * Therefore the totalLength > 0 check.
       *
       * The Animate is about to fire handleAnimationStart which will update the state
       * and cause a re-render and read a new proper totalLength which will be used in the next tick
       * and update the longestAnimatedLengthRef.
       *
       * Why Math.max? Sometimes the curve goes through a smaller length than previously recorded.
       * If we just set it to curLength, then the next animation would start from a smaller length
       * which looks weird. So we keep the longest length ever reached and then animate from there.
       */
      // eslint-disable-next-line no-param-reassign
      longestAnimatedLengthRef.current = Math.max(longestAnimatedLengthRef.current, curLength);
    }
    if (prevPoints) {
      var prevPointsDiffFactor = prevPoints.length / points.length;
      var stepData = t === 1 ? points : points.map((entry, index) => {
        var prevPointIndex = Math.floor(index * prevPointsDiffFactor);
        if (prevPoints[prevPointIndex]) {
          var prev = prevPoints[prevPointIndex];
          return _objectSpread(_objectSpread({}, entry), {}, {
            x: interpolate(prev.x, entry.x, t),
            y: interpolate(prev.y, entry.y, t)
          });
        }

        // magic number of faking previous x and y location
        if (animateNewValues) {
          return _objectSpread(_objectSpread({}, entry), {}, {
            x: interpolate(width * 2, entry.x, t),
            y: interpolate(height / 2, entry.y, t)
          });
        }
        return _objectSpread(_objectSpread({}, entry), {}, {
          x: entry.x,
          y: entry.y
        });
      });
      // eslint-disable-next-line no-param-reassign
      previousPointsRef.current = stepData;
      return /*#__PURE__*/React.createElement(StaticCurve, {
        props: props,
        points: stepData,
        clipPathId: clipPathId,
        pathRef: pathRef,
        strokeDasharray: currentStrokeDasharray
      });
    }
    return /*#__PURE__*/React.createElement(StaticCurve, {
      props: props,
      points: points,
      clipPathId: clipPathId,
      pathRef: pathRef,
      strokeDasharray: currentStrokeDasharray
    });
  }), /*#__PURE__*/React.createElement(LabelListFromLabelProp, {
    label: props.label
  }));
}
function RenderCurve(_ref6) {
  var {
    clipPathId,
    props
  } = _ref6;
  var previousPointsRef = useRef(null);
  var longestAnimatedLengthRef = useRef(0);
  var pathRef = useRef(null);
  return /*#__PURE__*/React.createElement(CurveWithAnimation, {
    props: props,
    clipPathId: clipPathId,
    previousPointsRef: previousPointsRef,
    longestAnimatedLengthRef: longestAnimatedLengthRef,
    pathRef: pathRef
  });
}
var errorBarDataPointFormatter = (dataPoint, dataKey) => {
  var _dataPoint$x, _dataPoint$y;
  return {
    x: (_dataPoint$x = dataPoint.x) !== null && _dataPoint$x !== void 0 ? _dataPoint$x : undefined,
    y: (_dataPoint$y = dataPoint.y) !== null && _dataPoint$y !== void 0 ? _dataPoint$y : undefined,
    value: dataPoint.value,
    // getValueByDataKey does not validate the output type
    errorVal: getValueByDataKey(dataPoint.payload, dataKey)
  };
};

// eslint-disable-next-line react/prefer-stateless-function
class LineWithState extends Component {
  render() {
    var {
      hide,
      dot,
      points,
      className,
      xAxisId,
      yAxisId,
      top,
      left,
      width,
      height,
      id,
      needClip,
      zIndex
    } = this.props;
    if (hide) {
      return null;
    }
    var layerClass = clsx('recharts-line', className);
    var clipPathId = id;
    var {
      r,
      strokeWidth
    } = getRadiusAndStrokeWidthFromDot(dot);
    var clipDot = isClipDot(dot);
    var dotSize = r * 2 + strokeWidth;
    var activePointsClipPath = needClip ? "url(#clipPath-".concat(clipDot ? '' : 'dots-').concat(clipPathId, ")") : undefined;
    return /*#__PURE__*/React.createElement(ZIndexLayer, {
      zIndex: zIndex
    }, /*#__PURE__*/React.createElement(Layer, {
      className: layerClass
    }, needClip && /*#__PURE__*/React.createElement("defs", null, /*#__PURE__*/React.createElement(GraphicalItemClipPath, {
      clipPathId: clipPathId,
      xAxisId: xAxisId,
      yAxisId: yAxisId
    }), !clipDot && /*#__PURE__*/React.createElement("clipPath", {
      id: "clipPath-dots-".concat(clipPathId)
    }, /*#__PURE__*/React.createElement("rect", {
      x: left - dotSize / 2,
      y: top - dotSize / 2,
      width: width + dotSize,
      height: height + dotSize
    }))), /*#__PURE__*/React.createElement(SetErrorBarContext, {
      xAxisId: xAxisId,
      yAxisId: yAxisId,
      data: points,
      dataPointFormatter: errorBarDataPointFormatter,
      errorBarOffset: 0
    }, /*#__PURE__*/React.createElement(RenderCurve, {
      props: this.props,
      clipPathId: clipPathId
    }))), /*#__PURE__*/React.createElement(ActivePoints, {
      activeDot: this.props.activeDot,
      points: points,
      mainColor: this.props.stroke,
      itemDataKey: this.props.dataKey,
      clipPath: activePointsClipPath
    }));
  }
}
export var defaultLineProps = {
  activeDot: true,
  animateNewValues: true,
  animationBegin: 0,
  animationDuration: 1500,
  animationEasing: 'ease',
  connectNulls: false,
  dot: true,
  fill: '#fff',
  hide: false,
  isAnimationActive: 'auto',
  label: false,
  legendType: 'line',
  stroke: '#3182bd',
  strokeWidth: 1,
  xAxisId: 0,
  yAxisId: 0,
  zIndex: DefaultZIndexes.line,
  type: 'linear'
};
function LineImpl(props) {
  var _resolveDefaultProps = resolveDefaultProps(props, defaultLineProps),
    {
      activeDot,
      animateNewValues,
      animationBegin,
      animationDuration,
      animationEasing,
      connectNulls,
      dot,
      hide,
      isAnimationActive,
      label,
      legendType,
      xAxisId,
      yAxisId,
      id
    } = _resolveDefaultProps,
    everythingElse = _objectWithoutProperties(_resolveDefaultProps, _excluded3);
  var {
    needClip
  } = useNeedsClip(xAxisId, yAxisId);
  var plotArea = usePlotArea();
  var layout = useChartLayout();
  var isPanorama = useIsPanorama();
  var points = useAppSelector(state => selectLinePoints(state, xAxisId, yAxisId, isPanorama, id));
  if (layout !== 'horizontal' && layout !== 'vertical' || points == null || plotArea == null) {
    // Cannot render Line in an unsupported layout
    return null;
  }
  var {
    height,
    width,
    x: left,
    y: top
  } = plotArea;
  return /*#__PURE__*/React.createElement(LineWithState, _extends({}, everythingElse, {
    id: id,
    connectNulls: connectNulls,
    dot: dot,
    activeDot: activeDot,
    animateNewValues: animateNewValues,
    animationBegin: animationBegin,
    animationDuration: animationDuration,
    animationEasing: animationEasing,
    isAnimationActive: isAnimationActive,
    hide: hide,
    label: label,
    legendType: legendType,
    xAxisId: xAxisId,
    yAxisId: yAxisId,
    points: points,
    layout: layout,
    height: height,
    width: width,
    left: left,
    top: top,
    needClip: needClip
  }));
}
export function computeLinePoints(_ref7) {
  var {
    layout,
    xAxis,
    yAxis,
    xAxisTicks,
    yAxisTicks,
    dataKey,
    bandSize,
    displayedData
  } = _ref7;
  return displayedData.map((entry, index) => {
    // getValueByDataKey does not validate the output type
    var value = getValueByDataKey(entry, dataKey);
    if (layout === 'horizontal') {
      var _x = getCateCoordinateOfLine({
        axis: xAxis,
        ticks: xAxisTicks,
        bandSize,
        entry,
        index
      });
      var _y = isNullish(value) ? null : yAxis.scale.map(value);
      return {
        x: _x,
        y: _y !== null && _y !== void 0 ? _y : null,
        value,
        payload: entry
      };
    }
    var x = isNullish(value) ? null : xAxis.scale.map(value);
    var y = getCateCoordinateOfLine({
      axis: yAxis,
      ticks: yAxisTicks,
      bandSize,
      entry,
      index
    });
    if (x == null || y == null) {
      return null;
    }
    return {
      x,
      y,
      value,
      payload: entry
    };
  }).filter(Boolean);
}
function LineFn(outsideProps) {
  var props = resolveDefaultProps(outsideProps, defaultLineProps);
  var isPanorama = useIsPanorama();
  return /*#__PURE__*/React.createElement(RegisterGraphicalItemId, {
    id: props.id,
    type: "line"
  }, id => /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement(SetLegendPayload, {
    legendPayload: computeLegendPayloadFromAreaData(props)
  }), /*#__PURE__*/React.createElement(SetLineTooltipEntrySettings, {
    dataKey: props.dataKey,
    data: props.data,
    stroke: props.stroke,
    strokeWidth: props.strokeWidth,
    fill: props.fill,
    name: props.name,
    hide: props.hide,
    unit: props.unit,
    tooltipType: props.tooltipType,
    id: id
  }), /*#__PURE__*/React.createElement(SetCartesianGraphicalItem, {
    type: "line",
    id: id,
    data: props.data,
    xAxisId: props.xAxisId,
    yAxisId: props.yAxisId,
    zAxisId: 0,
    dataKey: props.dataKey,
    hide: props.hide,
    isPanorama: isPanorama
  }), /*#__PURE__*/React.createElement(LineImpl, _extends({}, props, {
    id: id
  }))));
}

/**
 * @provides LabelListContext
 * @provides ErrorBarContext
 * @consumes CartesianChartContext
 */
export var Line = /*#__PURE__*/React.memo(LineFn, propsAreEqual);
// @ts-expect-error we need to set the displayName for debugging purposes
Line.displayName = 'Line';