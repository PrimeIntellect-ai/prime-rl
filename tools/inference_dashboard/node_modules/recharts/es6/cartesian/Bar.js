var _excluded = ["onMouseEnter", "onMouseLeave", "onClick"],
  _excluded2 = ["value", "background", "tooltipPosition"],
  _excluded3 = ["id"],
  _excluded4 = ["onMouseEnter", "onClick", "onMouseLeave"];
function _extends() { return _extends = Object.assign ? Object.assign.bind() : function (n) { for (var e = 1; e < arguments.length; e++) { var t = arguments[e]; for (var r in t) ({}).hasOwnProperty.call(t, r) && (n[r] = t[r]); } return n; }, _extends.apply(null, arguments); }
function ownKeys(e, r) { var t = Object.keys(e); if (Object.getOwnPropertySymbols) { var o = Object.getOwnPropertySymbols(e); r && (o = o.filter(function (r) { return Object.getOwnPropertyDescriptor(e, r).enumerable; })), t.push.apply(t, o); } return t; }
function _objectSpread(e) { for (var r = 1; r < arguments.length; r++) { var t = null != arguments[r] ? arguments[r] : {}; r % 2 ? ownKeys(Object(t), !0).forEach(function (r) { _defineProperty(e, r, t[r]); }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(t)) : ownKeys(Object(t)).forEach(function (r) { Object.defineProperty(e, r, Object.getOwnPropertyDescriptor(t, r)); }); } return e; }
function _defineProperty(e, r, t) { return (r = _toPropertyKey(r)) in e ? Object.defineProperty(e, r, { value: t, enumerable: !0, configurable: !0, writable: !0 }) : e[r] = t, e; }
function _toPropertyKey(t) { var i = _toPrimitive(t, "string"); return "symbol" == typeof i ? i : i + ""; }
function _toPrimitive(t, r) { if ("object" != typeof t || !t) return t; var e = t[Symbol.toPrimitive]; if (void 0 !== e) { var i = e.call(t, r || "default"); if ("object" != typeof i) return i; throw new TypeError("@@toPrimitive must return a primitive value."); } return ("string" === r ? String : Number)(t); }
function _objectWithoutProperties(e, t) { if (null == e) return {}; var o, r, i = _objectWithoutPropertiesLoose(e, t); if (Object.getOwnPropertySymbols) { var n = Object.getOwnPropertySymbols(e); for (r = 0; r < n.length; r++) o = n[r], -1 === t.indexOf(o) && {}.propertyIsEnumerable.call(e, o) && (i[o] = e[o]); } return i; }
function _objectWithoutPropertiesLoose(r, e) { if (null == r) return {}; var t = {}; for (var n in r) if ({}.hasOwnProperty.call(r, n)) { if (-1 !== e.indexOf(n)) continue; t[n] = r[n]; } return t; }
import * as React from 'react';
import { PureComponent, useCallback, useEffect, useRef, useState } from 'react';
import { clsx } from 'clsx';
import { Layer } from '../container/Layer';
import { Cell } from '../component/Cell';
import { CartesianLabelListContextProvider, LabelListFromLabelProp } from '../component/LabelList';
import { interpolate, isNan, mathSign, noop } from '../util/DataUtils';
import { findAllByType } from '../util/ReactUtils';
import { getBaseValueOfBar, getCateCoordinateOfBar, getTooltipNameProp, getValueByDataKey, truncateByDomain } from '../util/ChartUtils';
import { adaptEventsOfChild } from '../util/types';
import { BarRectangle, minPointSizeCallback } from '../util/BarUtils';
import { useMouseClickItemDispatch, useMouseEnterItemDispatch, useMouseLeaveItemDispatch } from '../context/tooltipContext';
import { SetTooltipEntrySettings } from '../state/SetTooltipEntrySettings';
import { SetErrorBarContext } from '../context/ErrorBarContext';
import { GraphicalItemClipPath, useNeedsClip } from './GraphicalItemClipPath';
import { useChartLayout } from '../context/chartLayoutContext';
import { selectBarRectangles } from '../state/selectors/barSelectors';
import { useAppSelector } from '../state/hooks';
import { useIsPanorama } from '../context/PanoramaContext';
import { selectActiveTooltipDataKey, selectActiveTooltipIndex } from '../state/selectors/tooltipSelectors';
import { SetLegendPayload } from '../state/SetLegendPayload';
import { useAnimationId } from '../util/useAnimationId';
import { resolveDefaultProps } from '../util/resolveDefaultProps';
import { RegisterGraphicalItemId } from '../context/RegisterGraphicalItemId';
import { SetCartesianGraphicalItem } from '../state/SetGraphicalItem';
import { svgPropertiesNoEvents, svgPropertiesNoEventsFromUnknown } from '../util/svgPropertiesNoEvents';
import { JavascriptAnimate } from '../animation/JavascriptAnimate';
import { ZIndexLayer } from '../zIndex/ZIndexLayer';
import { DefaultZIndexes } from '../zIndex/DefaultZIndexes';
import { getZIndexFromUnknown } from '../zIndex/getZIndexFromUnknown';
import { propsAreEqual } from '../util/propsAreEqual';
import { BarStackClipLayer, useStackId } from './BarStack';
var computeLegendPayloadFromBarData = props => {
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
var SetBarTooltipEntrySettings = /*#__PURE__*/React.memo(_ref => {
  var {
    dataKey,
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
    dataDefinedOnItem: undefined,
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
      color: fill,
      unit,
      graphicalItemId: id
    }
  };
  return /*#__PURE__*/React.createElement(SetTooltipEntrySettings, {
    tooltipEntrySettings: tooltipEntrySettings
  });
});
function BarBackground(props) {
  var activeIndex = useAppSelector(selectActiveTooltipIndex);
  var {
    data,
    dataKey,
    background: backgroundFromProps,
    allOtherBarProps
  } = props;
  var {
      onMouseEnter: onMouseEnterFromProps,
      onMouseLeave: onMouseLeaveFromProps,
      onClick: onItemClickFromProps
    } = allOtherBarProps,
    restOfAllOtherProps = _objectWithoutProperties(allOtherBarProps, _excluded);
  var onMouseEnterFromContext = useMouseEnterItemDispatch(onMouseEnterFromProps, dataKey, allOtherBarProps.id);
  var onMouseLeaveFromContext = useMouseLeaveItemDispatch(onMouseLeaveFromProps);
  var onClickFromContext = useMouseClickItemDispatch(onItemClickFromProps, dataKey, allOtherBarProps.id);
  if (!backgroundFromProps || data == null) {
    return null;
  }
  var backgroundProps = svgPropertiesNoEventsFromUnknown(backgroundFromProps);
  return /*#__PURE__*/React.createElement(ZIndexLayer, {
    zIndex: getZIndexFromUnknown(backgroundFromProps, DefaultZIndexes.barBackground)
  }, data.map((entry, i) => {
    var {
        value,
        background: backgroundFromDataEntry,
        tooltipPosition
      } = entry,
      rest = _objectWithoutProperties(entry, _excluded2);
    if (!backgroundFromDataEntry) {
      return null;
    }
    var onMouseEnter = onMouseEnterFromContext(entry, i);
    var onMouseLeave = onMouseLeaveFromContext(entry, i);
    var onClick = onClickFromContext(entry, i);
    var barRectangleProps = _objectSpread(_objectSpread(_objectSpread(_objectSpread(_objectSpread({
      option: backgroundFromProps,
      isActive: String(i) === activeIndex
    }, rest), {}, {
      // @ts-expect-error backgroundProps is contributing unknown props
      fill: '#eee'
    }, backgroundFromDataEntry), backgroundProps), adaptEventsOfChild(restOfAllOtherProps, entry, i)), {}, {
      onMouseEnter,
      onMouseLeave,
      onClick,
      dataKey,
      index: i,
      className: 'recharts-bar-background-rectangle'
    });
    return /*#__PURE__*/React.createElement(BarRectangle, _extends({
      key: "background-bar-".concat(i)
    }, barRectangleProps));
  }));
}
function BarLabelListProvider(_ref2) {
  var {
    showLabels,
    children,
    rects
  } = _ref2;
  var labelListEntries = rects === null || rects === void 0 ? void 0 : rects.map(entry => {
    var viewBox = {
      x: entry.x,
      y: entry.y,
      width: entry.width,
      lowerWidth: entry.width,
      upperWidth: entry.width,
      height: entry.height
    };
    return _objectSpread(_objectSpread({}, viewBox), {}, {
      value: entry.value,
      payload: entry.payload,
      parentViewBox: entry.parentViewBox,
      viewBox,
      fill: entry.fill
    });
  });
  return /*#__PURE__*/React.createElement(CartesianLabelListContextProvider, {
    value: showLabels ? labelListEntries : undefined
  }, children);
}
function BarRectangleWithActiveState(props) {
  var {
    shape,
    activeBar,
    baseProps,
    entry,
    index,
    dataKey
  } = props;
  var activeIndex = useAppSelector(selectActiveTooltipIndex);
  var activeDataKey = useAppSelector(selectActiveTooltipDataKey);
  /*
   * Bars support stacking, meaning that there can be multiple bars at the same x value.
   * With Tooltip shared=false we only want to highlight the currently active Bar, not all.
   *
   * Also, if the tooltip is shared, we want to highlight all bars at the same x value
   * regardless of the dataKey.
   *
   * With shared Tooltip, the activeDataKey is undefined.
   *
   * We use entry.originalDataIndex to match against activeIndex because the render index parameter
   * is based on the filtered array, while activeIndex is based on the pre-filter displayed data slice.
   * When entries are filtered out (for example null/zero-dimension bars), these indices can differ.
   */
  var isActive = activeBar && String(entry.originalDataIndex) === activeIndex && (activeDataKey == null || dataKey === activeDataKey);
  var [stayInLayer, setStayInLayer] = useState(false);
  var [hasMountedActive, setHasMountedActive] = useState(false);
  useEffect(() => {
    var rafId;
    if (isActive) {
      // 1. Enter the layer immediately
      setStayInLayer(true);

      // 2. Wait for the browser to paint the "inactive" state in the new layer,
      // then switch to active to trigger the CSS transition (width grow).
      rafId = requestAnimationFrame(() => {
        setHasMountedActive(true);
      });
    } else {
      setHasMountedActive(false);
    }
    return () => {
      cancelAnimationFrame(rafId);
    };
  }, [isActive]);
  var handleTransitionEnd = useCallback(() => {
    // 4. Leave the layer only when the exit transition finishes
    if (!isActive) {
      setStayInLayer(false);
    }
  }, [isActive]);

  // Determine props:
  // - If entering (isActive=true) but not mounted yet (hasMountedActive=false), pass isActive=false (inactive size).
  // - If exiting (isActive=false), pass isActive=false (inactive size).
  var isVisuallyActive = isActive && hasMountedActive;

  // Render in ZIndexLayer if active OR if we are waiting for exit transition
  var shouldRenderInLayer = isActive || stayInLayer;
  var option;
  if (isActive) {
    if (activeBar === true) {
      option = shape;
    } else {
      option = activeBar;
    }
  } else {
    option = shape;
  }
  var content = /*#__PURE__*/React.createElement(BarRectangle, _extends({}, baseProps, {
    name: String(baseProps.name)
  }, entry, {
    isActive: isVisuallyActive,
    option: option,
    index: index,
    dataKey: dataKey,
    onTransitionEnd: handleTransitionEnd
  }));
  if (shouldRenderInLayer) {
    return /*#__PURE__*/React.createElement(ZIndexLayer, {
      zIndex: DefaultZIndexes.activeBar
    }, /*#__PURE__*/React.createElement(BarStackClipLayer, {
      index: entry.originalDataIndex
    }, content));
  }
  return content;
}
function BarRectangleNeverActive(props) {
  var {
    shape,
    baseProps,
    entry,
    index,
    dataKey
  } = props;
  return /*#__PURE__*/React.createElement(BarRectangle, _extends({}, baseProps, {
    name: String(baseProps.name)
  }, entry, {
    isActive: false,
    option: shape,
    index: index,
    dataKey: dataKey
  }));
}
function BarRectangles(_ref3) {
  var _svgPropertiesNoEvent;
  var {
    data,
    props
  } = _ref3;
  var _ref4 = (_svgPropertiesNoEvent = svgPropertiesNoEvents(props)) !== null && _svgPropertiesNoEvent !== void 0 ? _svgPropertiesNoEvent : {},
    {
      id
    } = _ref4,
    baseProps = _objectWithoutProperties(_ref4, _excluded3);
  var {
    shape,
    dataKey,
    activeBar
  } = props;
  var {
      onMouseEnter: onMouseEnterFromProps,
      onClick: onItemClickFromProps,
      onMouseLeave: onMouseLeaveFromProps
    } = props,
    restOfAllOtherProps = _objectWithoutProperties(props, _excluded4);
  var onMouseEnterFromContext = useMouseEnterItemDispatch(onMouseEnterFromProps, dataKey, id);
  var onMouseLeaveFromContext = useMouseLeaveItemDispatch(onMouseLeaveFromProps);
  var onClickFromContext = useMouseClickItemDispatch(onItemClickFromProps, dataKey, id);
  if (!data) {
    return null;
  }
  return /*#__PURE__*/React.createElement(React.Fragment, null, data.map((entry, i) => {
    return /*#__PURE__*/React.createElement(BarStackClipLayer, _extends({
      index: entry.originalDataIndex
      // https://github.com/recharts/recharts/issues/5415
      ,
      key: "rectangle-".concat(entry === null || entry === void 0 ? void 0 : entry.x, "-").concat(entry === null || entry === void 0 ? void 0 : entry.y, "-").concat(entry === null || entry === void 0 ? void 0 : entry.value, "-").concat(i),
      className: "recharts-bar-rectangle"
    }, adaptEventsOfChild(restOfAllOtherProps, entry, i), {
      onMouseEnter: onMouseEnterFromContext(entry, i),
      onMouseLeave: onMouseLeaveFromContext(entry, i),
      onClick: onClickFromContext(entry, i)
    }), activeBar ? /*#__PURE__*/React.createElement(BarRectangleWithActiveState, {
      shape: shape,
      activeBar: activeBar,
      baseProps: baseProps,
      entry: entry,
      index: i,
      dataKey: dataKey
    }) :
    /*#__PURE__*/
    /*
     * If the `activeBar` prop is falsy, then let's call the variant without hooks.
     * Using the `selectActiveTooltipIndex` selector is usually fast
     * but in charts with large-ish amount of data even the few nanoseconds add up to a noticeable jank.
     * If the activeBar is false then we don't need to know which index is active - because we won't use it anyway.
     * So let's just skip the hooks altogether. That way, React can skip rendering the component,
     * and can skip the tree reconciliation for its children too.
     * Because we can't call hooks conditionally, we need to have a separate component for that.
     */
    React.createElement(BarRectangleNeverActive, {
      shape: shape,
      baseProps: baseProps,
      entry: entry,
      index: i,
      dataKey: dataKey
    }));
  }));
}
function RectanglesWithAnimation(_ref5) {
  var {
    props,
    previousRectanglesRef
  } = _ref5;
  var {
    data,
    layout,
    isAnimationActive,
    animationBegin,
    animationDuration,
    animationEasing,
    onAnimationEnd,
    onAnimationStart
  } = props;
  var prevData = previousRectanglesRef.current;
  var animationId = useAnimationId(props, 'recharts-bar-');
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
  return /*#__PURE__*/React.createElement(BarLabelListProvider, {
    showLabels: showLabels,
    rects: data
  }, /*#__PURE__*/React.createElement(JavascriptAnimate, {
    animationId: animationId,
    begin: animationBegin,
    duration: animationDuration,
    isActive: isAnimationActive,
    easing: animationEasing,
    onAnimationEnd: handleAnimationEnd,
    onAnimationStart: handleAnimationStart,
    key: animationId
  }, t => {
    var stepData = t === 1 ? data : data === null || data === void 0 ? void 0 : data.map((entry, index) => {
      var prev = prevData && prevData[index];
      if (prev) {
        return _objectSpread(_objectSpread({}, entry), {}, {
          x: interpolate(prev.x, entry.x, t),
          y: interpolate(prev.y, entry.y, t),
          width: interpolate(prev.width, entry.width, t),
          height: interpolate(prev.height, entry.height, t)
        });
      }
      if (layout === 'horizontal') {
        var height = interpolate(0, entry.height, t);
        var y = interpolate(entry.stackedBarStart, entry.y, t);
        return _objectSpread(_objectSpread({}, entry), {}, {
          y,
          height
        });
      }
      var w = interpolate(0, entry.width, t);
      var x = interpolate(entry.stackedBarStart, entry.x, t);
      return _objectSpread(_objectSpread({}, entry), {}, {
        width: w,
        x
      });
    });
    if (t > 0) {
      // eslint-disable-next-line no-param-reassign
      previousRectanglesRef.current = stepData !== null && stepData !== void 0 ? stepData : null;
    }
    if (stepData == null) {
      return null;
    }
    return /*#__PURE__*/React.createElement(Layer, null, /*#__PURE__*/React.createElement(BarRectangles, {
      props: props,
      data: stepData
    }));
  }), /*#__PURE__*/React.createElement(LabelListFromLabelProp, {
    label: props.label
  }), props.children);
}
function RenderRectangles(props) {
  var previousRectanglesRef = useRef(null);
  return /*#__PURE__*/React.createElement(RectanglesWithAnimation, {
    previousRectanglesRef: previousRectanglesRef,
    props: props
  });
}
var defaultMinPointSize = 0;
var errorBarDataPointFormatter = (dataPoint, dataKey) => {
  /**
   * if the value coming from `selectBarRectangles` is an array then this is a stacked bar chart.
   * arr[1] represents end value of the bar since the data is in the form of [startValue, endValue].
   * */
  var value = Array.isArray(dataPoint.value) ? dataPoint.value[1] : dataPoint.value;
  return {
    x: dataPoint.x,
    y: dataPoint.y,
    value,
    // getValueByDataKey does not validate the output type
    errorVal: getValueByDataKey(dataPoint, dataKey)
  };
};
class BarWithState extends PureComponent {
  render() {
    var {
      hide,
      data,
      dataKey,
      className,
      xAxisId,
      yAxisId,
      needClip,
      background,
      id
    } = this.props;
    if (hide || data == null) {
      return null;
    }
    var layerClass = clsx('recharts-bar', className);
    var clipPathId = id;
    return /*#__PURE__*/React.createElement(Layer, {
      className: layerClass,
      id: id
    }, needClip && /*#__PURE__*/React.createElement("defs", null, /*#__PURE__*/React.createElement(GraphicalItemClipPath, {
      clipPathId: clipPathId,
      xAxisId: xAxisId,
      yAxisId: yAxisId
    })), /*#__PURE__*/React.createElement(Layer, {
      className: "recharts-bar-rectangles",
      clipPath: needClip ? "url(#clipPath-".concat(clipPathId, ")") : undefined
    }, /*#__PURE__*/React.createElement(BarBackground, {
      data: data,
      dataKey: dataKey,
      background: background,
      allOtherBarProps: this.props
    }), /*#__PURE__*/React.createElement(RenderRectangles, this.props)));
  }
}
export var defaultBarProps = {
  activeBar: false,
  animationBegin: 0,
  animationDuration: 400,
  animationEasing: 'ease',
  background: false,
  hide: false,
  isAnimationActive: 'auto',
  label: false,
  legendType: 'rect',
  minPointSize: defaultMinPointSize,
  xAxisId: 0,
  yAxisId: 0,
  zIndex: DefaultZIndexes.bar
};
function BarImpl(props) {
  var {
    xAxisId,
    yAxisId,
    hide,
    legendType,
    minPointSize,
    activeBar,
    animationBegin,
    animationDuration,
    animationEasing,
    isAnimationActive
  } = props;
  var {
    needClip
  } = useNeedsClip(xAxisId, yAxisId);
  var layout = useChartLayout();
  var isPanorama = useIsPanorama();
  var cells = findAllByType(props.children, Cell);
  var rects = useAppSelector(state => selectBarRectangles(state, props.id, isPanorama, cells));
  if (layout !== 'vertical' && layout !== 'horizontal') {
    return null;
  }
  var errorBarOffset;
  var firstDataPoint = rects === null || rects === void 0 ? void 0 : rects[0];
  if (firstDataPoint == null || firstDataPoint.height == null || firstDataPoint.width == null) {
    errorBarOffset = 0;
  } else {
    errorBarOffset = layout === 'vertical' ? firstDataPoint.height / 2 : firstDataPoint.width / 2;
  }
  return /*#__PURE__*/React.createElement(SetErrorBarContext, {
    xAxisId: xAxisId,
    yAxisId: yAxisId,
    data: rects,
    dataPointFormatter: errorBarDataPointFormatter,
    errorBarOffset: errorBarOffset
  }, /*#__PURE__*/React.createElement(BarWithState, _extends({}, props, {
    layout: layout,
    needClip: needClip,
    data: rects,
    xAxisId: xAxisId,
    yAxisId: yAxisId,
    hide: hide,
    legendType: legendType,
    minPointSize: minPointSize,
    activeBar: activeBar,
    animationBegin: animationBegin,
    animationDuration: animationDuration,
    animationEasing: animationEasing,
    isAnimationActive: isAnimationActive
  })));
}
export function computeBarRectangles(_ref6) {
  var {
    layout,
    barSettings: {
      dataKey,
      minPointSize: minPointSizeProp,
      hasCustomShape
    },
    pos,
    bandSize,
    xAxis,
    yAxis,
    xAxisTicks,
    yAxisTicks,
    stackedData,
    displayedData,
    offset,
    cells,
    parentViewBox,
    dataStartIndex
  } = _ref6;
  var numericAxis = layout === 'horizontal' ? yAxis : xAxis;
  // @ts-expect-error this assumes that the domain is always numeric, but doesn't check for it
  var stackedDomain = stackedData ? numericAxis.scale.domain() : null;
  var baseValue = getBaseValueOfBar({
    numericAxis
  });
  var stackedBarStart = numericAxis.scale.map(baseValue);
  return displayedData.map((entry, index) => {
    var value, x, y, width, height, background;
    if (stackedData) {
      // Use dataStartIndex to access the correct element in the full stackedData array
      var untruncatedValue = stackedData[index + dataStartIndex];
      if (untruncatedValue == null) {
        return null;
      }
      value = truncateByDomain(untruncatedValue, stackedDomain);
    } else {
      value = getValueByDataKey(entry, dataKey);
      if (!Array.isArray(value)) {
        value = [baseValue, value];
      }
    }
    var minPointSize = minPointSizeCallback(minPointSizeProp, defaultMinPointSize)(value[1], index);
    if (layout === 'horizontal') {
      var _ref7;
      var baseValueScale = yAxis.scale.map(value[0]);
      var currentValueScale = yAxis.scale.map(value[1]);
      if (baseValueScale == null || currentValueScale == null) {
        return null;
      }
      x = getCateCoordinateOfBar({
        axis: xAxis,
        ticks: xAxisTicks,
        bandSize,
        offset: pos.offset,
        entry,
        index
      });
      y = (_ref7 = currentValueScale !== null && currentValueScale !== void 0 ? currentValueScale : baseValueScale) !== null && _ref7 !== void 0 ? _ref7 : undefined;
      width = pos.size;
      var computedHeight = baseValueScale - currentValueScale;
      height = isNan(computedHeight) ? 0 : computedHeight;
      background = {
        x,
        y: offset.top,
        width,
        height: offset.height
      };
      if (Math.abs(minPointSize) > 0 && Math.abs(height) < Math.abs(minPointSize)) {
        var delta = mathSign(height || minPointSize) * (Math.abs(minPointSize) - Math.abs(height));
        y -= delta;
        height += delta;
      }
    } else {
      var _baseValueScale = xAxis.scale.map(value[0]);
      var _currentValueScale = xAxis.scale.map(value[1]);
      if (_baseValueScale == null || _currentValueScale == null) {
        return null;
      }
      x = _baseValueScale;
      y = getCateCoordinateOfBar({
        axis: yAxis,
        ticks: yAxisTicks,
        bandSize,
        offset: pos.offset,
        entry,
        index
      });
      width = _currentValueScale - _baseValueScale;
      height = pos.size;
      background = {
        x: offset.left,
        y,
        width: offset.width,
        height
      };
      if (Math.abs(minPointSize) > 0 && Math.abs(width) < Math.abs(minPointSize)) {
        var _delta = mathSign(width || minPointSize) * (Math.abs(minPointSize) - Math.abs(width));
        width += _delta;
      }
    }

    /*
     * Filter out 0-dimension rectangles early to avoid creating unnecessary component trees.
     * BarStack clip-paths use originalDataIndex, so sparse filtered arrays remain index-stable.
     * Bars with a custom shape are not filtered out: the custom renderer may still draw something
     * visible at zero-dimension positions (e.g. horizontal lines in a BoxPlot).
     */
    if (x == null || y == null || width == null || height == null || !hasCustomShape && (width === 0 || height === 0)) {
      return null;
    }
    var barRectangleItem = _objectSpread(_objectSpread({}, entry), {}, {
      stackedBarStart,
      x,
      y,
      width,
      height,
      value: stackedData ? value : value[1],
      payload: entry,
      background,
      tooltipPosition: {
        x: x + width / 2,
        y: y + height / 2
      },
      parentViewBox,
      originalDataIndex: index
    }, cells && cells[index] && cells[index].props);
    return barRectangleItem;
  }).filter(Boolean);
}
function BarFn(outsideProps) {
  var props = resolveDefaultProps(outsideProps, defaultBarProps);
  // stackId may arrive from props or from BarStack context
  var stackId = useStackId(props.stackId);
  var isPanorama = useIsPanorama();
  // Report all props to Redux store first, before calling any hooks, to avoid circular dependencies.
  return /*#__PURE__*/React.createElement(RegisterGraphicalItemId, {
    id: props.id,
    type: "bar"
  }, id => /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement(SetLegendPayload, {
    legendPayload: computeLegendPayloadFromBarData(props)
  }), /*#__PURE__*/React.createElement(SetBarTooltipEntrySettings, {
    dataKey: props.dataKey,
    stroke: props.stroke,
    strokeWidth: props.strokeWidth,
    fill: props.fill,
    name: props.name,
    hide: props.hide,
    unit: props.unit,
    tooltipType: props.tooltipType,
    id: id
  }), /*#__PURE__*/React.createElement(SetCartesianGraphicalItem, {
    type: "bar",
    id: id
    // Bar does not allow setting data directly on the graphical item (why?)
    ,
    data: undefined,
    xAxisId: props.xAxisId,
    yAxisId: props.yAxisId,
    zAxisId: 0,
    dataKey: props.dataKey,
    stackId: stackId,
    hide: props.hide,
    barSize: props.barSize,
    minPointSize: props.minPointSize,
    maxBarSize: props.maxBarSize,
    isPanorama: isPanorama,
    hasCustomShape: props.shape != null
  }), /*#__PURE__*/React.createElement(ZIndexLayer, {
    zIndex: props.zIndex
  }, /*#__PURE__*/React.createElement(BarImpl, _extends({}, props, {
    id: id
  })))));
}

/**
 * @provides ErrorBarContext
 * @provides LabelListContext
 * @provides CellReader
 * @consumes CartesianChartContext
 * @consumes BarStackContext
 */
export var Bar = /*#__PURE__*/React.memo(BarFn, propsAreEqual);
// @ts-expect-error we need to set the displayName for debugging purposes
Bar.displayName = 'Bar';