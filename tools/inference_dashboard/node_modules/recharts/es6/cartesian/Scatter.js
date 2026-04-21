var _excluded = ["id"],
  _excluded2 = ["onMouseEnter", "onClick", "onMouseLeave"],
  _excluded3 = ["animationBegin", "animationDuration", "animationEasing", "hide", "isAnimationActive", "legendType", "lineJointType", "lineType", "shape", "xAxisId", "yAxisId", "zAxisId"];
function _objectWithoutProperties(e, t) { if (null == e) return {}; var o, r, i = _objectWithoutPropertiesLoose(e, t); if (Object.getOwnPropertySymbols) { var n = Object.getOwnPropertySymbols(e); for (r = 0; r < n.length; r++) o = n[r], -1 === t.indexOf(o) && {}.propertyIsEnumerable.call(e, o) && (i[o] = e[o]); } return i; }
function _objectWithoutPropertiesLoose(r, e) { if (null == r) return {}; var t = {}; for (var n in r) if ({}.hasOwnProperty.call(r, n)) { if (-1 !== e.indexOf(n)) continue; t[n] = r[n]; } return t; }
function _extends() { return _extends = Object.assign ? Object.assign.bind() : function (n) { for (var e = 1; e < arguments.length; e++) { var t = arguments[e]; for (var r in t) ({}).hasOwnProperty.call(t, r) && (n[r] = t[r]); } return n; }, _extends.apply(null, arguments); }
function ownKeys(e, r) { var t = Object.keys(e); if (Object.getOwnPropertySymbols) { var o = Object.getOwnPropertySymbols(e); r && (o = o.filter(function (r) { return Object.getOwnPropertyDescriptor(e, r).enumerable; })), t.push.apply(t, o); } return t; }
function _objectSpread(e) { for (var r = 1; r < arguments.length; r++) { var t = null != arguments[r] ? arguments[r] : {}; r % 2 ? ownKeys(Object(t), !0).forEach(function (r) { _defineProperty(e, r, t[r]); }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(t)) : ownKeys(Object(t)).forEach(function (r) { Object.defineProperty(e, r, Object.getOwnPropertyDescriptor(t, r)); }); } return e; }
function _defineProperty(e, r, t) { return (r = _toPropertyKey(r)) in e ? Object.defineProperty(e, r, { value: t, enumerable: !0, configurable: !0, writable: !0 }) : e[r] = t, e; }
function _toPropertyKey(t) { var i = _toPrimitive(t, "string"); return "symbol" == typeof i ? i : i + ""; }
function _toPrimitive(t, r) { if ("object" != typeof t || !t) return t; var e = t[Symbol.toPrimitive]; if (void 0 !== e) { var i = e.call(t, r || "default"); if ("object" != typeof i) return i; throw new TypeError("@@toPrimitive must return a primitive value."); } return ("string" === r ? String : Number)(t); }
import * as React from 'react';
import { useCallback, useMemo, useRef, useState } from 'react';
import { clsx } from 'clsx';
import { Layer } from '../container/Layer';
import { CartesianLabelListContextProvider, LabelListFromLabelProp } from '../component/LabelList';
import { findAllByType } from '../util/ReactUtils';
import { Curve } from '../shape/Curve';
import { Cell } from '../component/Cell';
import { getLinearRegression, interpolate, isNullish } from '../util/DataUtils';
import { getCateCoordinateOfLine, getTooltipNameProp, getValueByDataKey } from '../util/ChartUtils';
import { adaptEventsOfChild, isNonEmptyArray } from '../util/types';
import { ScatterSymbol } from '../util/ScatterUtils';
import { useMouseClickItemDispatch, useMouseEnterItemDispatch, useMouseLeaveItemDispatch } from '../context/tooltipContext';
import { SetTooltipEntrySettings } from '../state/SetTooltipEntrySettings';
import { SetErrorBarContext } from '../context/ErrorBarContext';
import { GraphicalItemClipPath, useNeedsClip } from './GraphicalItemClipPath';
import { selectScatterPoints } from '../state/selectors/scatterSelectors';
import { useAppSelector } from '../state/hooks';
import { implicitZAxis } from '../state/selectors/axisSelectors';
import { useIsPanorama } from '../context/PanoramaContext';
import { selectActiveTooltipIndex } from '../state/selectors/tooltipSelectors';
import { SetLegendPayload } from '../state/SetLegendPayload';
import { DATA_ITEM_GRAPHICAL_ITEM_ID_ATTRIBUTE_NAME } from '../util/Constants';
import { useAnimationId } from '../util/useAnimationId';
import { resolveDefaultProps } from '../util/resolveDefaultProps';
import { RegisterGraphicalItemId } from '../context/RegisterGraphicalItemId';
import { SetCartesianGraphicalItem } from '../state/SetGraphicalItem';
import { svgPropertiesNoEvents, svgPropertiesNoEventsFromUnknown } from '../util/svgPropertiesNoEvents';
import { JavascriptAnimate } from '../animation/JavascriptAnimate';
import { useViewBox } from '../context/chartLayoutContext';
import { ZIndexLayer } from '../zIndex/ZIndexLayer';
import { DefaultZIndexes } from '../zIndex/DefaultZIndexes';
import { propsAreEqual } from '../util/propsAreEqual';

/**
 * Scatter coordinates are nullable because sometimes the point value is out of the domain,
 * and we can't compute a valid coordinate for it.
 *
 * Scatter -> Symbol ignores points with null cx or cy so those won't render if using the default shapes.
 * However: the points are exposed via various props and can be used in custom shapes so we keep them around.
 */

/**
 * Internal props, combination of external props + defaultProps + private Recharts state
 */

/**
 * External props, intended for end users to fill in
 */

/**
 * Because of naming conflict, we are forced to ignore certain (valid) SVG attributes.
 */

var computeLegendPayloadFromScatterProps = props => {
  var {
    dataKey,
    name,
    fill,
    legendType,
    hide
  } = props;
  return [{
    inactive: hide,
    dataKey,
    type: legendType,
    color: fill,
    value: getTooltipNameProp(name, dataKey),
    payload: props
  }];
};
var SetScatterTooltipEntrySettings = /*#__PURE__*/React.memo(_ref => {
  var {
    dataKey,
    points,
    stroke,
    strokeWidth,
    fill,
    name,
    hide,
    tooltipType,
    id
  } = _ref;
  var tooltipEntrySettings = {
    dataDefinedOnItem: points === null || points === void 0 ? void 0 : points.map(p => p.tooltipPayload),
    getPosition: index => {
      var _points$Number;
      return points === null || points === void 0 || (_points$Number = points[Number(index)]) === null || _points$Number === void 0 ? void 0 : _points$Number.tooltipPosition;
    },
    settings: {
      stroke,
      strokeWidth,
      fill,
      nameKey: undefined,
      dataKey,
      name: getTooltipNameProp(name, dataKey),
      hide,
      type: tooltipType,
      color: fill,
      unit: '',
      // why doesn't Scatter support unit?
      graphicalItemId: id
    }
  };
  return /*#__PURE__*/React.createElement(SetTooltipEntrySettings, {
    tooltipEntrySettings: tooltipEntrySettings
  });
});
function ScatterLine(_ref2) {
  var {
    points,
    props
  } = _ref2;
  var {
    line,
    lineType,
    lineJointType
  } = props;
  if (!line) {
    return null;
  }
  var scatterProps = svgPropertiesNoEvents(props);
  var customLineProps = svgPropertiesNoEventsFromUnknown(line);
  var linePoints, lineItem;
  if (lineType === 'joint') {
    linePoints = points.map(entry => {
      var _entry$cx, _entry$cy;
      return {
        x: (_entry$cx = entry.cx) !== null && _entry$cx !== void 0 ? _entry$cx : null,
        y: (_entry$cy = entry.cy) !== null && _entry$cy !== void 0 ? _entry$cy : null
      };
    });
  } else if (lineType === 'fitting') {
    var {
      xmin,
      xmax,
      a,
      b
    } = getLinearRegression(points);
    var linearExp = x => a * x + b;
    linePoints = [{
      x: xmin,
      y: linearExp(xmin)
    }, {
      x: xmax,
      y: linearExp(xmax)
    }];
  }
  var lineProps = _objectSpread(_objectSpread(_objectSpread({}, scatterProps), {}, {
    // @ts-expect-error customLineProps is contributing unknown props
    fill: 'none',
    // @ts-expect-error customLineProps is contributing unknown props
    stroke: scatterProps && scatterProps.fill
  }, customLineProps), {}, {
    // @ts-expect-error linePoints is used before it is assigned (???)
    points: linePoints
  });
  if (/*#__PURE__*/React.isValidElement(line)) {
    lineItem = /*#__PURE__*/React.cloneElement(line, lineProps);
  } else if (typeof line === 'function') {
    lineItem = line(lineProps);
  } else {
    lineItem = /*#__PURE__*/React.createElement(Curve, _extends({}, lineProps, {
      type: lineJointType
    }));
  }
  return /*#__PURE__*/React.createElement(Layer, {
    className: "recharts-scatter-line",
    key: "recharts-scatter-line"
  }, lineItem);
}
function ScatterLabelListProvider(_ref3) {
  var {
    showLabels,
    points,
    children
  } = _ref3;
  var chartViewBox = useViewBox();
  var labelListEntries = useMemo(() => {
    return points === null || points === void 0 ? void 0 : points.map(point => {
      var _point$x, _point$y;
      var viewBox = {
        /*
         * Scatter label uses x and y as the reference point for the label,
         * not cx and cy.
         */
        x: (_point$x = point.x) !== null && _point$x !== void 0 ? _point$x : 0,
        /*
         * Scatter label uses x and y as the reference point for the label,
         * not cx and cy.
         */
        y: (_point$y = point.y) !== null && _point$y !== void 0 ? _point$y : 0,
        width: point.width,
        height: point.height,
        lowerWidth: point.width,
        upperWidth: point.width
      };
      return _objectSpread(_objectSpread({}, viewBox), {}, {
        /*
         * Here we put undefined because Scatter shows two values usually, one for X and one for Y.
         * LabelList will see this undefined and will use its own `dataKey` prop to determine which value to show,
         * using the payload below.
         */
        value: undefined,
        payload: point.payload,
        viewBox,
        parentViewBox: chartViewBox,
        fill: undefined
      });
    });
  }, [chartViewBox, points]);
  return /*#__PURE__*/React.createElement(CartesianLabelListContextProvider, {
    value: showLabels ? labelListEntries : undefined
  }, children);
}
function ScatterSymbols(props) {
  var {
    points,
    allOtherScatterProps
  } = props;
  var {
    shape,
    activeShape,
    dataKey
  } = allOtherScatterProps;
  var {
      id
    } = allOtherScatterProps,
    allOtherPropsWithoutId = _objectWithoutProperties(allOtherScatterProps, _excluded);
  var activeIndex = useAppSelector(selectActiveTooltipIndex);
  var {
      onMouseEnter: onMouseEnterFromProps,
      onClick: onItemClickFromProps,
      onMouseLeave: onMouseLeaveFromProps
    } = allOtherScatterProps,
    restOfAllOtherProps = _objectWithoutProperties(allOtherScatterProps, _excluded2);
  var onMouseEnterFromContext = useMouseEnterItemDispatch(onMouseEnterFromProps, dataKey, id);
  var onMouseLeaveFromContext = useMouseLeaveItemDispatch(onMouseLeaveFromProps);
  var onClickFromContext = useMouseClickItemDispatch(onItemClickFromProps, dataKey, id);
  if (!isNonEmptyArray(points)) {
    return null;
  }
  var baseProps = svgPropertiesNoEvents(allOtherPropsWithoutId);
  return /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement(ScatterLine, {
    points: points,
    props: allOtherPropsWithoutId
  }), points.map((entry, i) => {
    var hasActiveShape = activeShape != null && activeShape !== false;
    var isActive = hasActiveShape && activeIndex === String(i);
    var option = hasActiveShape && isActive ? activeShape : shape;
    var symbolProps = _objectSpread(_objectSpread(_objectSpread({}, baseProps), entry), {}, {
      index: i,
      [DATA_ITEM_GRAPHICAL_ITEM_ID_ATTRIBUTE_NAME]: String(id)
    });
    return /*#__PURE__*/React.createElement(ZIndexLayer, {
      key: "symbol-".concat(entry === null || entry === void 0 ? void 0 : entry.cx, "-").concat(entry === null || entry === void 0 ? void 0 : entry.cy, "-").concat(entry === null || entry === void 0 ? void 0 : entry.size, "-").concat(i)
      /*
       * inactive Scatters use the parent zIndex, which is represented by undefined here.
       * ZIndexLayer will render undefined zIndex as-is, as regular children, without portals.
       * Active Scatters use the activeDot zIndex so they render above other elements.
       */,
      zIndex: isActive ? DefaultZIndexes.activeDot : undefined
    }, /*#__PURE__*/React.createElement(Layer, _extends({
      className: "recharts-scatter-symbol"
    }, adaptEventsOfChild(restOfAllOtherProps, entry, i), {
      onMouseEnter: onMouseEnterFromContext(entry, i),
      onMouseLeave: onMouseLeaveFromContext(entry, i),
      onClick: onClickFromContext(entry, i)
    }), /*#__PURE__*/React.createElement(ScatterSymbol, _extends({
      option: option,
      isActive: isActive
    }, symbolProps))));
  }));
}
function SymbolsWithAnimation(_ref4) {
  var {
    previousPointsRef,
    props
  } = _ref4;
  var {
    points,
    isAnimationActive,
    animationBegin,
    animationDuration,
    animationEasing
  } = props;
  var prevPoints = previousPointsRef.current;
  var animationId = useAnimationId(props, 'recharts-scatter-');
  var [isAnimating, setIsAnimating] = useState(false);
  var handleAnimationEnd = useCallback(() => {
    // Scatter doesn't have onAnimationEnd prop, and if we want to add it we do it here
    // if (typeof onAnimationEnd === 'function') {
    //   onAnimationEnd();
    // }
    setIsAnimating(false);
  }, []);
  var handleAnimationStart = useCallback(() => {
    // Scatter doesn't have onAnimationStart prop, and if we want to add it we do it here
    // if (typeof onAnimationStart === 'function') {
    //   onAnimationStart();
    // }
    setIsAnimating(true);
  }, []);
  var showLabels = !isAnimating;
  return /*#__PURE__*/React.createElement(ScatterLabelListProvider, {
    showLabels: showLabels,
    points: points
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
    var stepData = t === 1 ? points : points === null || points === void 0 ? void 0 : points.map((entry, index) => {
      var prev = prevPoints && prevPoints[index];
      if (prev) {
        return _objectSpread(_objectSpread({}, entry), {}, {
          cx: entry.cx == null ? undefined : interpolate(prev.cx, entry.cx, t),
          cy: entry.cy == null ? undefined : interpolate(prev.cy, entry.cy, t),
          size: interpolate(prev.size, entry.size, t)
        });
      }
      return _objectSpread(_objectSpread({}, entry), {}, {
        size: interpolate(0, entry.size, t)
      });
    });
    if (t > 0) {
      // eslint-disable-next-line no-param-reassign
      previousPointsRef.current = stepData;
    }
    return /*#__PURE__*/React.createElement(Layer, null, /*#__PURE__*/React.createElement(ScatterSymbols, {
      points: stepData,
      allOtherScatterProps: props,
      showLabels: showLabels
    }));
  }), /*#__PURE__*/React.createElement(LabelListFromLabelProp, {
    label: props.label
  }));
}
export function computeScatterPoints(_ref5) {
  var {
    displayedData,
    xAxis,
    yAxis,
    zAxis,
    scatterSettings,
    xAxisTicks,
    yAxisTicks,
    cells
  } = _ref5;
  var xAxisDataKey = isNullish(xAxis.dataKey) ? scatterSettings.dataKey : xAxis.dataKey;
  var yAxisDataKey = isNullish(yAxis.dataKey) ? scatterSettings.dataKey : yAxis.dataKey;
  var zAxisDataKey = zAxis && zAxis.dataKey;
  var defaultRangeZ = zAxis ? zAxis.range : implicitZAxis.range;
  var defaultZ = defaultRangeZ && defaultRangeZ[0];
  var xBandSize = xAxis.scale.bandwidth ? xAxis.scale.bandwidth() : 0;
  var yBandSize = yAxis.scale.bandwidth ? yAxis.scale.bandwidth() : 0;
  return displayedData.map((entry, index) => {
    var x = getValueByDataKey(entry, xAxisDataKey);
    var y = getValueByDataKey(entry, yAxisDataKey);
    var z = !isNullish(zAxisDataKey) && getValueByDataKey(entry, zAxisDataKey) || '-';
    var tooltipPayload = [{
      name: isNullish(xAxis.dataKey) ? scatterSettings.name : xAxis.name || String(xAxis.dataKey),
      unit: xAxis.unit || '',
      // @ts-expect-error getValueByDataKey does not validate the output type
      value: x,
      payload: entry,
      dataKey: xAxisDataKey,
      type: scatterSettings.tooltipType,
      graphicalItemId: scatterSettings.id
    }, {
      name: isNullish(yAxis.dataKey) ? scatterSettings.name : yAxis.name || String(yAxis.dataKey),
      unit: yAxis.unit || '',
      // @ts-expect-error getValueByDataKey does not validate the output type
      value: y,
      payload: entry,
      dataKey: yAxisDataKey,
      type: scatterSettings.tooltipType,
      graphicalItemId: scatterSettings.id
    }];
    if (z !== '-' && zAxis != null) {
      tooltipPayload.push({
        // @ts-expect-error name prop should not have dataKey in it
        name: zAxis.name || zAxis.dataKey,
        unit: zAxis.unit || '',
        // @ts-expect-error getValueByDataKey does not validate the output type
        value: z,
        payload: entry,
        dataKey: zAxisDataKey,
        type: scatterSettings.tooltipType,
        graphicalItemId: scatterSettings.id
      });
    }
    var cx = getCateCoordinateOfLine({
      axis: xAxis,
      ticks: xAxisTicks,
      bandSize: xBandSize,
      entry,
      index,
      dataKey: xAxisDataKey
    });
    var cy = getCateCoordinateOfLine({
      axis: yAxis,
      ticks: yAxisTicks,
      bandSize: yBandSize,
      entry,
      index,
      dataKey: yAxisDataKey
    });
    var size = z !== '-' && zAxis != null ? zAxis.scale.map(z) : defaultZ;
    var radius = size == null ? 0 : Math.sqrt(Math.max(size, 0) / Math.PI);
    return _objectSpread(_objectSpread({}, entry), {}, {
      cx,
      cy,
      x: cx == null ? undefined : cx - radius,
      y: cy == null ? undefined : cy - radius,
      width: 2 * radius,
      height: 2 * radius,
      size,
      node: {
        x,
        y,
        z
      },
      tooltipPayload,
      tooltipPosition: {
        x: cx,
        y: cy
      },
      payload: entry
    }, cells && cells[index] && cells[index].props);
  });
}
var errorBarDataPointFormatter = (dataPoint, dataKey, direction) => {
  return {
    x: dataPoint.cx,
    y: dataPoint.cy,
    value: direction === 'x' ? Number(dataPoint.node.x) : Number(dataPoint.node.y),
    // @ts-expect-error getValueByDataKey does not validate the output type
    errorVal: getValueByDataKey(dataPoint, dataKey)
  };
};
function ScatterWithId(props) {
  var {
    hide,
    points,
    className,
    needClip,
    xAxisId,
    yAxisId,
    id
  } = props;
  var previousPointsRef = useRef(null);
  if (hide) {
    return null;
  }
  var layerClass = clsx('recharts-scatter', className);
  var clipPathId = id;
  return /*#__PURE__*/React.createElement(ZIndexLayer, {
    zIndex: props.zIndex
  }, /*#__PURE__*/React.createElement(Layer, {
    className: layerClass,
    clipPath: needClip ? "url(#clipPath-".concat(clipPathId, ")") : undefined,
    id: id
  }, needClip && /*#__PURE__*/React.createElement("defs", null, /*#__PURE__*/React.createElement(GraphicalItemClipPath, {
    clipPathId: clipPathId,
    xAxisId: xAxisId,
    yAxisId: yAxisId
  })), /*#__PURE__*/React.createElement(SetErrorBarContext, {
    xAxisId: xAxisId,
    yAxisId: yAxisId,
    data: points,
    dataPointFormatter: errorBarDataPointFormatter,
    errorBarOffset: 0
  }, /*#__PURE__*/React.createElement(Layer, {
    key: "recharts-scatter-symbols"
  }, /*#__PURE__*/React.createElement(SymbolsWithAnimation, {
    props: props,
    previousPointsRef: previousPointsRef
  })))));
}
export var defaultScatterProps = {
  xAxisId: 0,
  yAxisId: 0,
  zAxisId: 0,
  label: false,
  line: false,
  legendType: 'circle',
  lineType: 'joint',
  lineJointType: 'linear',
  shape: 'circle',
  hide: false,
  isAnimationActive: 'auto',
  animationBegin: 0,
  animationDuration: 400,
  animationEasing: 'linear',
  zIndex: DefaultZIndexes.scatter
};
function ScatterImpl(props) {
  var _resolveDefaultProps = resolveDefaultProps(props, defaultScatterProps),
    {
      animationBegin,
      animationDuration,
      animationEasing,
      hide,
      isAnimationActive,
      legendType,
      lineJointType,
      lineType,
      shape,
      xAxisId,
      yAxisId,
      zAxisId
    } = _resolveDefaultProps,
    everythingElse = _objectWithoutProperties(_resolveDefaultProps, _excluded3);
  var {
    needClip
  } = useNeedsClip(xAxisId, yAxisId);
  var cells = useMemo(() => findAllByType(props.children, Cell), [props.children]);
  var isPanorama = useIsPanorama();
  var points = useAppSelector(state => {
    return selectScatterPoints(state, xAxisId, yAxisId, zAxisId, props.id, cells, isPanorama);
  });
  if (needClip == null) {
    return null;
  }
  if (points == null) {
    return null;
  }
  return /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement(SetScatterTooltipEntrySettings, {
    dataKey: props.dataKey,
    points: points,
    stroke: props.stroke,
    strokeWidth: props.strokeWidth,
    fill: props.fill,
    name: props.name,
    hide: props.hide,
    tooltipType: props.tooltipType,
    id: props.id
  }), /*#__PURE__*/React.createElement(ScatterWithId, _extends({}, everythingElse, {
    xAxisId: xAxisId,
    yAxisId: yAxisId,
    zAxisId: zAxisId,
    lineType: lineType,
    lineJointType: lineJointType,
    legendType: legendType,
    shape: shape,
    hide: hide,
    isAnimationActive: isAnimationActive,
    animationBegin: animationBegin,
    animationDuration: animationDuration,
    animationEasing: animationEasing,
    points: points,
    needClip: needClip
  })));
}
function ScatterFn(outsideProps) {
  var props = resolveDefaultProps(outsideProps, defaultScatterProps);
  var isPanorama = useIsPanorama();
  return /*#__PURE__*/React.createElement(RegisterGraphicalItemId, {
    id: props.id,
    type: "scatter"
  }, id => /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement(SetLegendPayload, {
    legendPayload: computeLegendPayloadFromScatterProps(props)
  }), /*#__PURE__*/React.createElement(SetCartesianGraphicalItem, {
    type: "scatter",
    id: id,
    data: props.data,
    xAxisId: props.xAxisId,
    yAxisId: props.yAxisId,
    zAxisId: props.zAxisId,
    dataKey: props.dataKey,
    hide: props.hide,
    name: props.name,
    tooltipType: props.tooltipType,
    isPanorama: isPanorama
  }), /*#__PURE__*/React.createElement(ScatterImpl, _extends({}, props, {
    id: id
  }))));
}

/**
 * @provides LabelListContext
 * @provides ErrorBarContext
 * @provides CellReader
 * @consumes CartesianChartContext
 */
export var Scatter = /*#__PURE__*/React.memo(ScatterFn, propsAreEqual);
// @ts-expect-error we need to set the displayName for debugging purposes

Scatter.displayName = 'Scatter';