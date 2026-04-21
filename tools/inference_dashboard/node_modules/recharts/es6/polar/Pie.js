var _excluded = ["key"],
  _excluded2 = ["onMouseEnter", "onClick", "onMouseLeave"],
  _excluded3 = ["id"],
  _excluded4 = ["id"];
function _extends() { return _extends = Object.assign ? Object.assign.bind() : function (n) { for (var e = 1; e < arguments.length; e++) { var t = arguments[e]; for (var r in t) ({}).hasOwnProperty.call(t, r) && (n[r] = t[r]); } return n; }, _extends.apply(null, arguments); }
function _objectWithoutProperties(e, t) { if (null == e) return {}; var o, r, i = _objectWithoutPropertiesLoose(e, t); if (Object.getOwnPropertySymbols) { var n = Object.getOwnPropertySymbols(e); for (r = 0; r < n.length; r++) o = n[r], -1 === t.indexOf(o) && {}.propertyIsEnumerable.call(e, o) && (i[o] = e[o]); } return i; }
function _objectWithoutPropertiesLoose(r, e) { if (null == r) return {}; var t = {}; for (var n in r) if ({}.hasOwnProperty.call(r, n)) { if (-1 !== e.indexOf(n)) continue; t[n] = r[n]; } return t; }
function ownKeys(e, r) { var t = Object.keys(e); if (Object.getOwnPropertySymbols) { var o = Object.getOwnPropertySymbols(e); r && (o = o.filter(function (r) { return Object.getOwnPropertyDescriptor(e, r).enumerable; })), t.push.apply(t, o); } return t; }
function _objectSpread(e) { for (var r = 1; r < arguments.length; r++) { var t = null != arguments[r] ? arguments[r] : {}; r % 2 ? ownKeys(Object(t), !0).forEach(function (r) { _defineProperty(e, r, t[r]); }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(t)) : ownKeys(Object(t)).forEach(function (r) { Object.defineProperty(e, r, Object.getOwnPropertyDescriptor(t, r)); }); } return e; }
function _defineProperty(e, r, t) { return (r = _toPropertyKey(r)) in e ? Object.defineProperty(e, r, { value: t, enumerable: !0, configurable: !0, writable: !0 }) : e[r] = t, e; }
function _toPropertyKey(t) { var i = _toPrimitive(t, "string"); return "symbol" == typeof i ? i : i + ""; }
function _toPrimitive(t, r) { if ("object" != typeof t || !t) return t; var e = t[Symbol.toPrimitive]; if (void 0 !== e) { var i = e.call(t, r || "default"); if ("object" != typeof i) return i; throw new TypeError("@@toPrimitive must return a primitive value."); } return ("string" === r ? String : Number)(t); }
import * as React from 'react';
import { useCallback, useMemo, useRef, useState } from 'react';
import get from 'es-toolkit/compat/get';
import { clsx } from 'clsx';
import { selectPieLegend, selectPieSectors } from '../state/selectors/pieSelectors';
import { useAppSelector } from '../state/hooks';
import { Layer } from '../container/Layer';
import { Curve } from '../shape/Curve';
import { Text } from '../component/Text';
import { Cell } from '../component/Cell';
import { findAllByType } from '../util/ReactUtils';
import { getMaxRadius, polarToCartesian } from '../util/PolarUtils';
import { getPercentValue, interpolate, isNumber, mathSign } from '../util/DataUtils';
import { getTooltipNameProp, getValueByDataKey } from '../util/ChartUtils';
import { adaptEventsOfChild } from '../util/types';
import { Shape } from '../util/ActiveShapeUtils';
import { useMouseClickItemDispatch, useMouseEnterItemDispatch, useMouseLeaveItemDispatch } from '../context/tooltipContext';
import { SetTooltipEntrySettings } from '../state/SetTooltipEntrySettings';
import { selectActiveTooltipDataKey, selectActiveTooltipGraphicalItemId, selectActiveTooltipIndex } from '../state/selectors/tooltipSelectors';
import { SetPolarLegendPayload } from '../state/SetLegendPayload';
import { DATA_ITEM_GRAPHICAL_ITEM_ID_ATTRIBUTE_NAME, DATA_ITEM_INDEX_ATTRIBUTE_NAME } from '../util/Constants';
import { useAnimationId } from '../util/useAnimationId';
import { resolveDefaultProps } from '../util/resolveDefaultProps';
import { RegisterGraphicalItemId } from '../context/RegisterGraphicalItemId';
import { SetPolarGraphicalItem } from '../state/SetGraphicalItem';
import { svgPropertiesNoEvents, svgPropertiesNoEventsFromUnknown } from '../util/svgPropertiesNoEvents';
import { JavascriptAnimate } from '../animation/JavascriptAnimate';
import { LabelListFromLabelProp, PolarLabelListContextProvider } from '../component/LabelList';
import { ZIndexLayer } from '../zIndex/ZIndexLayer';
import { DefaultZIndexes } from '../zIndex/DefaultZIndexes';
import { getClassNameFromUnknown } from '../util/getClassNameFromUnknown';

/**
 * The `label` prop in Pie accepts a variety of alternatives.
 */

/**
 * We spread the data object into the sector data item,
 * so we can't really know what is going to be inside.
 *
 * This type represents our best effort, but it all depends on the input data
 * and what is inside of it.
 *
 * https://github.com/recharts/recharts/issues/6380
 * https://github.com/recharts/recharts/discussions/6375
 */

/**
 * Internal props, combination of external props + defaultProps + private Recharts state
 */

function SetPiePayloadLegend(props) {
  var cells = useMemo(() => findAllByType(props.children, Cell), [props.children]);
  var legendPayload = useAppSelector(state => selectPieLegend(state, props.id, cells));
  if (legendPayload == null) {
    return null;
  }
  return /*#__PURE__*/React.createElement(SetPolarLegendPayload, {
    legendPayload: legendPayload
  });
}
function getActiveShapeFill(activeShape) {
  // activeShape can be boolean/function/element/object; only element/object can carry a static fill value.
  if (activeShape == null || typeof activeShape === 'boolean' || typeof activeShape === 'function') {
    return undefined;
  }
  if (/*#__PURE__*/React.isValidElement(activeShape)) {
    var _activeShape$props;
    // React element form: <Sector fill="..."/> or custom element with fill prop.
    var _fill = (_activeShape$props = activeShape.props) === null || _activeShape$props === void 0 ? void 0 : _activeShape$props.fill;
    return typeof _fill === 'string' ? _fill : undefined;
  }
  var {
    fill
  } = activeShape;
  return typeof fill === 'string' ? fill : undefined;
}
var SetPieTooltipEntrySettings = /*#__PURE__*/React.memo(_ref => {
  var {
    dataKey,
    nameKey,
    sectors,
    stroke,
    strokeWidth,
    fill,
    name,
    hide,
    tooltipType,
    id,
    activeShape
  } = _ref;
  var activeShapeFill = getActiveShapeFill(activeShape);
  var tooltipDataDefinedOnItem = sectors.map(sector => {
    var sectorTooltipPayload = sector.tooltipPayload;
    if (activeShapeFill == null || sectorTooltipPayload == null) {
      return sectorTooltipPayload;
    }
    return sectorTooltipPayload.map(item => _objectSpread(_objectSpread({}, item), {}, {
      color: activeShapeFill,
      fill: activeShapeFill
    }));
  });
  var tooltipEntrySettings = {
    dataDefinedOnItem: tooltipDataDefinedOnItem,
    getPosition: index => {
      var _sectors$Number;
      return (_sectors$Number = sectors[Number(index)]) === null || _sectors$Number === void 0 ? void 0 : _sectors$Number.tooltipPosition;
    },
    settings: {
      stroke,
      strokeWidth,
      fill,
      dataKey,
      nameKey,
      name: getTooltipNameProp(name, dataKey),
      hide,
      type: tooltipType,
      color: fill,
      unit: '',
      // why doesn't Pie support unit?
      graphicalItemId: id
    }
  };
  return /*#__PURE__*/React.createElement(SetTooltipEntrySettings, {
    tooltipEntrySettings: tooltipEntrySettings
  });
});
var getTextAnchor = (x, cx) => {
  if (x > cx) {
    return 'start';
  }
  if (x < cx) {
    return 'end';
  }
  return 'middle';
};
var getOuterRadius = (dataPoint, outerRadius, maxPieRadius) => {
  if (typeof outerRadius === 'function') {
    return getPercentValue(outerRadius(dataPoint), maxPieRadius, maxPieRadius * 0.8);
  }
  return getPercentValue(outerRadius, maxPieRadius, maxPieRadius * 0.8);
};
var parseCoordinateOfPie = (pieSettings, offset, dataPoint) => {
  var {
    top,
    left,
    width,
    height
  } = offset;
  var maxPieRadius = getMaxRadius(width, height);
  var cx = left + getPercentValue(pieSettings.cx, width, width / 2);
  var cy = top + getPercentValue(pieSettings.cy, height, height / 2);
  var innerRadius = getPercentValue(pieSettings.innerRadius, maxPieRadius, 0);
  var outerRadius = getOuterRadius(dataPoint, pieSettings.outerRadius, maxPieRadius);
  var maxRadius = pieSettings.maxRadius || Math.sqrt(width * width + height * height) / 2;
  return {
    cx,
    cy,
    innerRadius,
    outerRadius,
    maxRadius
  };
};
var parseDeltaAngle = (startAngle, endAngle) => {
  var sign = mathSign(endAngle - startAngle);
  var deltaAngle = Math.min(Math.abs(endAngle - startAngle), 360);
  return sign * deltaAngle;
};
var renderLabelLineItem = (option, props) => {
  if (/*#__PURE__*/React.isValidElement(option)) {
    // @ts-expect-error we can't know if the type of props matches the element
    return /*#__PURE__*/React.cloneElement(option, props);
  }
  if (typeof option === 'function') {
    return option(props);
  }
  var className = clsx('recharts-pie-label-line', typeof option !== 'boolean' ? option.className : '');
  // React doesn't like it when we spread a key property onto an element
  var {
      key
    } = props,
    otherProps = _objectWithoutProperties(props, _excluded);
  return /*#__PURE__*/React.createElement(Curve, _extends({}, otherProps, {
    type: "linear",
    className: className
  }));
};
var renderLabelItem = (option, props, value) => {
  if (/*#__PURE__*/React.isValidElement(option)) {
    // @ts-expect-error element cloning is not typed
    return /*#__PURE__*/React.cloneElement(option, props);
  }
  var label = value;
  if (typeof option === 'function') {
    label = option(props);
    if (/*#__PURE__*/React.isValidElement(label)) {
      return label;
    }
  }
  var className = clsx('recharts-pie-label-text', getClassNameFromUnknown(option));
  return /*#__PURE__*/React.createElement(Text, _extends({}, props, {
    alignmentBaseline: "middle",
    className: className
  }), label);
};
function PieLabels(_ref2) {
  var {
    sectors,
    props,
    showLabels
  } = _ref2;
  var {
    label,
    labelLine,
    dataKey
  } = props;
  if (!showLabels || !label || !sectors) {
    return null;
  }
  var pieProps = svgPropertiesNoEvents(props);
  var customLabelProps = svgPropertiesNoEventsFromUnknown(label);
  var customLabelLineProps = svgPropertiesNoEventsFromUnknown(labelLine);
  var offsetRadius = typeof label === 'object' && 'offsetRadius' in label && typeof label.offsetRadius === 'number' && label.offsetRadius || 20;
  var labels = sectors.map((entry, i) => {
    var midAngle = (entry.startAngle + entry.endAngle) / 2;
    var endPoint = polarToCartesian(entry.cx, entry.cy, entry.outerRadius + offsetRadius, midAngle);
    var labelProps = _objectSpread(_objectSpread(_objectSpread(_objectSpread({}, pieProps), entry), {}, {
      // @ts-expect-error customLabelProps is contributing unknown props
      stroke: 'none'
    }, customLabelProps), {}, {
      index: i,
      textAnchor: getTextAnchor(endPoint.x, entry.cx)
    }, endPoint);
    var lineProps = _objectSpread(_objectSpread(_objectSpread(_objectSpread({}, pieProps), entry), {}, {
      // @ts-expect-error customLabelLineProps is contributing unknown props
      fill: 'none',
      // @ts-expect-error customLabelLineProps is contributing unknown props
      stroke: entry.fill
    }, customLabelLineProps), {}, {
      index: i,
      points: [polarToCartesian(entry.cx, entry.cy, entry.outerRadius, midAngle), endPoint],
      key: 'line'
    });
    return /*#__PURE__*/React.createElement(ZIndexLayer, {
      zIndex: DefaultZIndexes.label,
      key: "label-".concat(entry.startAngle, "-").concat(entry.endAngle, "-").concat(entry.midAngle, "-").concat(i)
    }, /*#__PURE__*/React.createElement(Layer, null, labelLine && renderLabelLineItem(labelLine, lineProps), renderLabelItem(label, labelProps, getValueByDataKey(entry, dataKey))));
  });
  return /*#__PURE__*/React.createElement(Layer, {
    className: "recharts-pie-labels"
  }, labels);
}
function PieLabelList(_ref3) {
  var {
    sectors,
    props,
    showLabels
  } = _ref3;
  var {
    label
  } = props;
  if (typeof label === 'object' && label != null && 'position' in label) {
    return /*#__PURE__*/React.createElement(LabelListFromLabelProp, {
      label: label
    });
  }
  return /*#__PURE__*/React.createElement(PieLabels, {
    sectors: sectors,
    props: props,
    showLabels: showLabels
  });
}
function PieSectors(props) {
  var {
    sectors,
    activeShape,
    inactiveShape: inactiveShapeProp,
    allOtherPieProps,
    shape,
    id
  } = props;
  var activeIndex = useAppSelector(selectActiveTooltipIndex);
  var activeDataKey = useAppSelector(selectActiveTooltipDataKey);
  var activeGraphicalItemId = useAppSelector(selectActiveTooltipGraphicalItemId);
  var {
      onMouseEnter: onMouseEnterFromProps,
      onClick: onItemClickFromProps,
      onMouseLeave: onMouseLeaveFromProps
    } = allOtherPieProps,
    restOfAllOtherProps = _objectWithoutProperties(allOtherPieProps, _excluded2);
  var onMouseEnterFromContext = useMouseEnterItemDispatch(onMouseEnterFromProps, allOtherPieProps.dataKey, id);
  var onMouseLeaveFromContext = useMouseLeaveItemDispatch(onMouseLeaveFromProps);
  var onClickFromContext = useMouseClickItemDispatch(onItemClickFromProps, allOtherPieProps.dataKey, id);
  if (sectors == null || sectors.length === 0) {
    return null;
  }
  return /*#__PURE__*/React.createElement(React.Fragment, null, sectors.map((entry, i) => {
    if ((entry === null || entry === void 0 ? void 0 : entry.startAngle) === 0 && (entry === null || entry === void 0 ? void 0 : entry.endAngle) === 0 && sectors.length !== 1) return null;

    // For Pie charts, when multiple Pies share the same dataKey, we need to ensure only the hovered Pie's sector is active.
    // We do this by checking if the active graphical item ID matches this Pie's ID.
    var graphicalItemMatches = activeGraphicalItemId == null || activeGraphicalItemId === id;
    var isActive = String(i) === activeIndex && (activeDataKey == null || allOtherPieProps.dataKey === activeDataKey) && graphicalItemMatches;
    var inactiveShape = activeIndex ? inactiveShapeProp : null;
    var sectorOptions = activeShape && isActive ? activeShape : inactiveShape;
    var sectorProps = _objectSpread(_objectSpread({}, entry), {}, {
      stroke: entry.stroke,
      tabIndex: -1,
      [DATA_ITEM_INDEX_ATTRIBUTE_NAME]: i,
      [DATA_ITEM_GRAPHICAL_ITEM_ID_ATTRIBUTE_NAME]: id
    });
    return /*#__PURE__*/React.createElement(Layer, _extends({
      key: "sector-".concat(entry === null || entry === void 0 ? void 0 : entry.startAngle, "-").concat(entry === null || entry === void 0 ? void 0 : entry.endAngle, "-").concat(entry.midAngle, "-").concat(i),
      tabIndex: -1,
      className: "recharts-pie-sector"
    }, adaptEventsOfChild(restOfAllOtherProps, entry, i), {
      onMouseEnter: onMouseEnterFromContext(entry, i),
      onMouseLeave: onMouseLeaveFromContext(entry, i),
      onClick: onClickFromContext(entry, i)
    }), /*#__PURE__*/React.createElement(Shape, _extends({
      option: shape !== null && shape !== void 0 ? shape : sectorOptions,
      index: i,
      shapeType: "sector",
      isActive: isActive
    }, sectorProps)));
  }));
}
export function computePieSectors(_ref4) {
  var _pieSettings$paddingA;
  var {
    pieSettings,
    displayedData,
    cells,
    offset
  } = _ref4;
  var {
    cornerRadius,
    startAngle,
    endAngle,
    dataKey,
    nameKey,
    tooltipType
  } = pieSettings;
  var minAngle = Math.abs(pieSettings.minAngle);
  var deltaAngle = parseDeltaAngle(startAngle, endAngle);
  var absDeltaAngle = Math.abs(deltaAngle);
  var paddingAngle = displayedData.length <= 1 ? 0 : (_pieSettings$paddingA = pieSettings.paddingAngle) !== null && _pieSettings$paddingA !== void 0 ? _pieSettings$paddingA : 0;
  var notZeroItemCount = displayedData.filter(entry => getValueByDataKey(entry, dataKey, 0) !== 0).length;
  var totalPaddingAngle = (absDeltaAngle >= 360 ? notZeroItemCount : notZeroItemCount - 1) * paddingAngle;
  var sum = displayedData.reduce((result, entry) => {
    var val = getValueByDataKey(entry, dataKey, 0);
    return result + (isNumber(val) ? val : 0);
  }, 0);

  // Only apply minAngle redistribution when at least one non-zero segment's
  // natural angle falls below the minAngle threshold. Otherwise, minAngle
  // unnecessarily shifts all segments even when none need the boost.
  // See: https://github.com/recharts/recharts/issues/6814
  var needsMinAngleAdjustment = minAngle > 0 && sum > 0 && displayedData.some(entry => {
    var val = getValueByDataKey(entry, dataKey, 0);
    var percent = (isNumber(val) ? val : 0) / sum;
    return val !== 0 && percent * absDeltaAngle < minAngle;
  });
  var effectiveMinAngle = needsMinAngleAdjustment ? minAngle : 0;
  var realTotalAngle = absDeltaAngle - notZeroItemCount * effectiveMinAngle - totalPaddingAngle;
  var sectors;
  if (sum > 0) {
    var prev;
    sectors = displayedData.map((entry, i) => {
      var val = getValueByDataKey(entry, dataKey, 0);
      var name = getValueByDataKey(entry, nameKey, i);
      var coordinate = parseCoordinateOfPie(pieSettings, offset, entry);
      var percent = (isNumber(val) ? val : 0) / sum;
      var tempStartAngle;

      // @ts-expect-error can't spread unknown
      var entryWithCellInfo = _objectSpread(_objectSpread({}, entry), cells && cells[i] && cells[i].props);
      var sectorColor = entryWithCellInfo != null && 'fill' in entryWithCellInfo && typeof entryWithCellInfo.fill === 'string' ? entryWithCellInfo.fill : pieSettings.fill;
      if (i) {
        tempStartAngle = prev.endAngle + mathSign(deltaAngle) * paddingAngle * (val !== 0 ? 1 : 0);
      } else {
        tempStartAngle = startAngle;
      }
      var tempEndAngle = tempStartAngle + mathSign(deltaAngle) * ((val !== 0 ? effectiveMinAngle : 0) + percent * realTotalAngle);
      var midAngle = (tempStartAngle + tempEndAngle) / 2;
      var middleRadius = (coordinate.innerRadius + coordinate.outerRadius) / 2;
      var tooltipPayload = [{
        name,
        value: val,
        payload: entryWithCellInfo,
        dataKey,
        type: tooltipType,
        color: sectorColor,
        fill: sectorColor,
        graphicalItemId: pieSettings.id
      }];
      var tooltipPosition = polarToCartesian(coordinate.cx, coordinate.cy, middleRadius, midAngle);
      prev = _objectSpread(_objectSpread(_objectSpread(_objectSpread({}, pieSettings.presentationProps), {}, {
        percent,
        cornerRadius: typeof cornerRadius === 'string' ? parseFloat(cornerRadius) : cornerRadius,
        name,
        tooltipPayload,
        midAngle,
        middleRadius,
        tooltipPosition
      }, entryWithCellInfo), coordinate), {}, {
        value: val,
        dataKey,
        startAngle: tempStartAngle,
        endAngle: tempEndAngle,
        payload: entryWithCellInfo,
        paddingAngle: val !== 0 ? mathSign(deltaAngle) * paddingAngle : 0
      });
      return prev;
    });
  }
  return sectors;
}
function PieLabelListProvider(_ref5) {
  var {
    showLabels,
    sectors,
    children
  } = _ref5;
  var labelListEntries = useMemo(() => {
    if (!showLabels || !sectors) {
      return [];
    }
    return sectors.map(entry => ({
      value: entry.value,
      payload: entry.payload,
      clockWise: false,
      parentViewBox: undefined,
      viewBox: {
        cx: entry.cx,
        cy: entry.cy,
        innerRadius: entry.innerRadius,
        outerRadius: entry.outerRadius,
        startAngle: entry.startAngle,
        endAngle: entry.endAngle,
        clockWise: false
      },
      fill: entry.fill
    }));
  }, [sectors, showLabels]);
  return /*#__PURE__*/React.createElement(PolarLabelListContextProvider, {
    value: showLabels ? labelListEntries : undefined
  }, children);
}
function SectorsWithAnimation(_ref6) {
  var {
    props,
    previousSectorsRef,
    id
  } = _ref6;
  var {
    sectors,
    isAnimationActive,
    animationBegin,
    animationDuration,
    animationEasing,
    activeShape,
    inactiveShape,
    onAnimationStart,
    onAnimationEnd
  } = props;
  var animationId = useAnimationId(props, 'recharts-pie-');
  var prevSectors = previousSectorsRef.current;
  var [isAnimating, setIsAnimating] = useState(false);
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
  return /*#__PURE__*/React.createElement(PieLabelListProvider, {
    showLabels: !isAnimating,
    sectors: sectors
  }, /*#__PURE__*/React.createElement(JavascriptAnimate, {
    animationId: animationId,
    begin: animationBegin,
    duration: animationDuration,
    isActive: isAnimationActive,
    easing: animationEasing,
    onAnimationStart: handleAnimationStart,
    onAnimationEnd: handleAnimationEnd,
    key: animationId
  }, t => {
    var _first$startAngle;
    var stepData = [];
    var first = sectors && sectors[0];
    var curAngle = (_first$startAngle = first === null || first === void 0 ? void 0 : first.startAngle) !== null && _first$startAngle !== void 0 ? _first$startAngle : 0;
    sectors === null || sectors === void 0 || sectors.forEach((entry, index) => {
      var prev = prevSectors && prevSectors[index];
      var paddingAngle = index > 0 ? get(entry, 'paddingAngle', 0) : 0;
      if (prev) {
        var angle = interpolate(prev.endAngle - prev.startAngle, entry.endAngle - entry.startAngle, t);
        var latest = _objectSpread(_objectSpread({}, entry), {}, {
          startAngle: curAngle + paddingAngle,
          endAngle: curAngle + angle + paddingAngle
        });
        stepData.push(latest);
        curAngle = latest.endAngle;
      } else {
        var {
          endAngle,
          startAngle
        } = entry;
        var deltaAngle = interpolate(0, endAngle - startAngle, t);
        var _latest = _objectSpread(_objectSpread({}, entry), {}, {
          startAngle: curAngle + paddingAngle,
          endAngle: curAngle + deltaAngle + paddingAngle
        });
        stepData.push(_latest);
        curAngle = _latest.endAngle;
      }
    });

    // eslint-disable-next-line no-param-reassign
    previousSectorsRef.current = stepData;
    return /*#__PURE__*/React.createElement(Layer, null, /*#__PURE__*/React.createElement(PieSectors, {
      sectors: stepData,
      activeShape: activeShape,
      inactiveShape: inactiveShape,
      allOtherPieProps: props,
      shape: props.shape,
      id: id
    }));
  }), /*#__PURE__*/React.createElement(PieLabelList, {
    showLabels: !isAnimating,
    sectors: sectors,
    props: props
  }), props.children);
}
export var defaultPieProps = {
  animationBegin: 400,
  animationDuration: 1500,
  animationEasing: 'ease',
  cx: '50%',
  cy: '50%',
  dataKey: 'value',
  endAngle: 360,
  fill: '#808080',
  hide: false,
  innerRadius: 0,
  isAnimationActive: 'auto',
  label: false,
  labelLine: true,
  legendType: 'rect',
  minAngle: 0,
  nameKey: 'name',
  outerRadius: '80%',
  paddingAngle: 0,
  rootTabIndex: 0,
  startAngle: 0,
  stroke: '#fff',
  zIndex: DefaultZIndexes.area
};
function PieImpl(props) {
  var {
      id
    } = props,
    propsWithoutId = _objectWithoutProperties(props, _excluded3);
  var {
    hide,
    className,
    rootTabIndex
  } = props;
  var cells = useMemo(() => findAllByType(props.children, Cell), [props.children]);
  var sectors = useAppSelector(state => selectPieSectors(state, id, cells));
  var previousSectorsRef = useRef(null);
  var layerClass = clsx('recharts-pie', className);
  if (hide || sectors == null) {
    previousSectorsRef.current = null;
    return /*#__PURE__*/React.createElement(Layer, {
      tabIndex: rootTabIndex,
      className: layerClass
    });
  }
  return /*#__PURE__*/React.createElement(ZIndexLayer, {
    zIndex: props.zIndex
  }, /*#__PURE__*/React.createElement(SetPieTooltipEntrySettings, {
    dataKey: props.dataKey,
    nameKey: props.nameKey,
    sectors: sectors,
    stroke: props.stroke,
    strokeWidth: props.strokeWidth,
    fill: props.fill,
    name: props.name,
    hide: props.hide,
    tooltipType: props.tooltipType,
    id: id,
    activeShape: props.activeShape
  }), /*#__PURE__*/React.createElement(Layer, {
    tabIndex: rootTabIndex,
    className: layerClass
  }, /*#__PURE__*/React.createElement(SectorsWithAnimation, {
    props: _objectSpread(_objectSpread({}, propsWithoutId), {}, {
      sectors
    }),
    previousSectorsRef: previousSectorsRef,
    id: id
  })));
}
/**
 * @consumes PolarChartContext
 * @provides LabelListContext
 * @provides CellReader
 */
function PieFn(outsideProps) {
  var props = resolveDefaultProps(outsideProps, defaultPieProps);
  var {
      id: externalId
    } = props,
    propsWithoutId = _objectWithoutProperties(props, _excluded4);
  var presentationProps = svgPropertiesNoEvents(propsWithoutId);
  return /*#__PURE__*/React.createElement(RegisterGraphicalItemId, {
    id: externalId,
    type: "pie"
  }, id => /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement(SetPolarGraphicalItem, {
    type: "pie",
    id: id,
    data: propsWithoutId.data,
    dataKey: propsWithoutId.dataKey,
    hide: propsWithoutId.hide,
    angleAxisId: 0,
    radiusAxisId: 0,
    name: propsWithoutId.name,
    nameKey: propsWithoutId.nameKey,
    tooltipType: propsWithoutId.tooltipType,
    legendType: propsWithoutId.legendType,
    fill: propsWithoutId.fill,
    cx: propsWithoutId.cx,
    cy: propsWithoutId.cy,
    startAngle: propsWithoutId.startAngle,
    endAngle: propsWithoutId.endAngle,
    paddingAngle: propsWithoutId.paddingAngle,
    minAngle: propsWithoutId.minAngle,
    innerRadius: propsWithoutId.innerRadius,
    outerRadius: propsWithoutId.outerRadius,
    cornerRadius: propsWithoutId.cornerRadius,
    presentationProps: presentationProps,
    maxRadius: props.maxRadius
  }), /*#__PURE__*/React.createElement(SetPiePayloadLegend, _extends({}, propsWithoutId, {
    id: id
  })), /*#__PURE__*/React.createElement(PieImpl, _extends({}, propsWithoutId, {
    id: id
  }))));
}
export var Pie = PieFn;
// @ts-expect-error we need to set the displayName for debugging purposes
Pie.displayName = 'Pie';