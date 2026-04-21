function _extends() { return _extends = Object.assign ? Object.assign.bind() : function (n) { for (var e = 1; e < arguments.length; e++) { var t = arguments[e]; for (var r in t) ({}).hasOwnProperty.call(t, r) && (n[r] = t[r]); } return n; }, _extends.apply(null, arguments); }
function ownKeys(e, r) { var t = Object.keys(e); if (Object.getOwnPropertySymbols) { var o = Object.getOwnPropertySymbols(e); r && (o = o.filter(function (r) { return Object.getOwnPropertyDescriptor(e, r).enumerable; })), t.push.apply(t, o); } return t; }
function _objectSpread(e) { for (var r = 1; r < arguments.length; r++) { var t = null != arguments[r] ? arguments[r] : {}; r % 2 ? ownKeys(Object(t), !0).forEach(function (r) { _defineProperty(e, r, t[r]); }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(t)) : ownKeys(Object(t)).forEach(function (r) { Object.defineProperty(e, r, Object.getOwnPropertyDescriptor(t, r)); }); } return e; }
function _defineProperty(e, r, t) { return (r = _toPropertyKey(r)) in e ? Object.defineProperty(e, r, { value: t, enumerable: !0, configurable: !0, writable: !0 }) : e[r] = t, e; }
function _toPropertyKey(t) { var i = _toPrimitive(t, "string"); return "symbol" == typeof i ? i : i + ""; }
function _toPrimitive(t, r) { if ("object" != typeof t || !t) return t; var e = t[Symbol.toPrimitive]; if (void 0 !== e) { var i = e.call(t, r || "default"); if ("object" != typeof i) return i; throw new TypeError("@@toPrimitive must return a primitive value."); } return ("string" === r ? String : Number)(t); }
import * as React from 'react';
import { useState } from 'react';
import { scaleLinear } from 'victory-vendor/d3-scale';
import { clsx } from 'clsx';
import get from 'es-toolkit/compat/get';
import { Surface } from '../container/Surface';
import { Layer } from '../container/Layer';
import { Sector } from '../shape/Sector';
import { Text } from '../component/Text';
import { polarToCartesian } from '../util/PolarUtils';
import { ReportChartMargin, ReportChartSize, useChartHeight, useChartWidth } from '../context/chartLayoutContext';
import { TooltipPortalContext } from '../context/tooltipPortalContext';
import { RechartsWrapper } from './RechartsWrapper';
import { mouseLeaveItem, setActiveClickItemIndex, setActiveMouseOverItemIndex } from '../state/tooltipSlice';
import { SetTooltipEntrySettings } from '../state/SetTooltipEntrySettings';
import { RechartsStoreProvider } from '../state/RechartsStoreProvider';
import { ReportEventSettings } from '../state/ReportEventSettings';
import { useAppDispatch } from '../state/hooks';
import { RegisterGraphicalItemId } from '../context/RegisterGraphicalItemId';
import { resolveDefaultProps } from '../util/resolveDefaultProps';
import { initialEventSettingsState } from '../state/eventSettingsSlice';

/**
 * We require tooltipIndex on each node internally to track which node is active in the tooltip.
 * This is not required from the outside user - we can calculate it as we traverse the tree.
 */

var defaultTextProps = {
  fontWeight: 'bold',
  paintOrder: 'stroke fill',
  fontSize: '.75rem',
  stroke: '#FFF',
  fill: 'black',
  pointerEvents: 'none'
};
function getMaxDepthOf(node) {
  if (!node.children || node.children.length === 0) return 1;

  // Calculate depth for each child and find the maximum
  var childDepths = node.children.map(d => getMaxDepthOf(d));
  return 1 + Math.max(...childDepths);
}
var SetSunburstTooltipEntrySettings = /*#__PURE__*/React.memo(_ref => {
  var {
    dataKey,
    nameKey,
    data,
    stroke,
    fill,
    positions,
    id
  } = _ref;
  var tooltipEntrySettings = {
    dataDefinedOnItem: data.children,
    getPosition: index => positions.get(index),
    // Sunburst does not support many of the properties as other charts do so there's plenty of defaults here
    settings: {
      stroke,
      strokeWidth: undefined,
      fill,
      nameKey,
      dataKey,
      // if there is a nameKey use it, otherwise make the name of the tooltip the dataKey itself
      name: nameKey ? undefined : dataKey,
      hide: false,
      type: undefined,
      color: fill,
      unit: '',
      graphicalItemId: id
    }
  };
  return /*#__PURE__*/React.createElement(SetTooltipEntrySettings, {
    tooltipEntrySettings: tooltipEntrySettings
  });
});

// Why is margin not a sunburst prop? No clue. Probably it should be
var defaultSunburstMargin = {
  top: 0,
  right: 0,
  bottom: 0,
  left: 0
};
export var payloadSearcher = (data, activeIndex) => {
  if (activeIndex == null) {
    return undefined;
  }
  return get(data, activeIndex);
};
var addToSunburstNodeIndex = function addToSunburstNodeIndex(indexInChildrenArr) {
  var activeTooltipIndexSoFar = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : '';
  return "".concat(activeTooltipIndexSoFar, "children[").concat(indexInChildrenArr, "]");
};
var preloadedState = {
  options: {
    validateTooltipEventTypes: ['item'],
    defaultTooltipEventType: 'item',
    chartName: 'Sunburst',
    tooltipPayloadSearcher: payloadSearcher,
    eventEmitter: undefined
  }
};
export var defaultSunburstChartProps = _objectSpread({
  padding: 2,
  dataKey: 'value',
  nameKey: 'name',
  ringPadding: 2,
  innerRadius: 50,
  fill: '#333',
  stroke: '#FFF',
  textOptions: defaultTextProps,
  startAngle: 0,
  endAngle: 360,
  responsive: false
}, initialEventSettingsState);
var SunburstChartImpl = _ref2 => {
  var {
    className,
    data,
    children,
    padding,
    dataKey,
    nameKey,
    ringPadding,
    innerRadius,
    fill,
    stroke,
    textOptions,
    outerRadius: outerRadiusFromProps,
    cx: cxFromProps,
    cy: cyFromProps,
    startAngle,
    endAngle,
    onClick,
    onMouseEnter,
    onMouseLeave,
    id
  } = _ref2;
  var dispatch = useAppDispatch();
  var width = useChartWidth();
  var height = useChartHeight();
  if (width == null || height == null) {
    return null;
  }
  var outerRadius = outerRadiusFromProps !== null && outerRadiusFromProps !== void 0 ? outerRadiusFromProps : Math.min(width, height) / 2;
  var cx = cxFromProps !== null && cxFromProps !== void 0 ? cxFromProps : width / 2;
  var cy = cyFromProps !== null && cyFromProps !== void 0 ? cyFromProps : height / 2;
  var rScale = scaleLinear([0, data[dataKey]], [0, endAngle]);
  var treeDepth = getMaxDepthOf(data);
  var thickness = (outerRadius - innerRadius) / treeDepth;
  var sectors = [];
  var positions = new Map([]);

  // event handlers
  function handleMouseEnter(node, e) {
    if (onMouseEnter) onMouseEnter(node, e);
    dispatch(setActiveMouseOverItemIndex({
      activeIndex: node.tooltipIndex,
      activeDataKey: dataKey,
      activeCoordinate: positions.get(node.name),
      activeGraphicalItemId: id
    }));
  }
  function handleMouseLeave(node, e) {
    if (onMouseLeave) onMouseLeave(node, e);
    dispatch(mouseLeaveItem());
  }
  function handleClick(node) {
    if (onClick) onClick(node);
    dispatch(setActiveClickItemIndex({
      activeIndex: node.tooltipIndex,
      activeDataKey: dataKey,
      activeCoordinate: positions.get(node.name),
      activeGraphicalItemId: id
    }));
  }

  // recursively add nodes for each data point and its children
  function drawArcs(childNodes, options) {
    var depth = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : 1;
    var {
      radius,
      innerR,
      initialAngle,
      childColor,
      nestedActiveTooltipIndex
    } = options;
    var currentAngle = initialAngle;
    if (!childNodes) return; // base case: no children of this node

    childNodes.forEach((d, i) => {
      var _ref3, _d$fill;
      var currentTooltipIndex = depth === 1 ? "[".concat(i, "]") : addToSunburstNodeIndex(i, nestedActiveTooltipIndex);
      var nodeWithIndex = _objectSpread(_objectSpread({}, d), {}, {
        tooltipIndex: currentTooltipIndex
      });
      var arcLength = rScale(d[dataKey]);
      var start = currentAngle;
      // color priority - if there's a color on the individual point use that, otherwise use parent color or default
      var fillColor = (_ref3 = (_d$fill = d === null || d === void 0 ? void 0 : d.fill) !== null && _d$fill !== void 0 ? _d$fill : childColor) !== null && _ref3 !== void 0 ? _ref3 : fill;
      var {
        x: textX,
        y: textY
      } = polarToCartesian(0, 0, innerR + radius / 2, -(start + arcLength - arcLength / 2));
      currentAngle += arcLength;
      sectors.push(/*#__PURE__*/React.createElement("g", {
        key: "sunburst-sector-".concat(d.name, "-").concat(i)
      }, /*#__PURE__*/React.createElement(Sector, {
        onClick: () => handleClick(nodeWithIndex),
        onMouseEnter: e => handleMouseEnter(nodeWithIndex, e),
        onMouseLeave: e => handleMouseLeave(nodeWithIndex, e),
        fill: fillColor,
        stroke: stroke,
        strokeWidth: padding,
        startAngle: start,
        endAngle: start + arcLength,
        innerRadius: innerR,
        outerRadius: innerR + radius,
        cx: cx,
        cy: cy
      }), /*#__PURE__*/React.createElement(Text, _extends({}, textOptions, {
        alignmentBaseline: "middle",
        textAnchor: "middle",
        x: textX + cx,
        y: cy - textY
      }), d[dataKey])));
      var {
        x: tooltipX,
        y: tooltipY
      } = polarToCartesian(cx, cy, innerR + radius / 2, start);
      positions.set(d.name, {
        x: tooltipX,
        y: tooltipY
      });
      return drawArcs(d.children, {
        radius,
        innerR: innerR + radius + ringPadding,
        initialAngle: start,
        childColor: fillColor,
        nestedActiveTooltipIndex: currentTooltipIndex
      }, depth + 1);
    });
  }
  drawArcs(data.children, {
    radius: thickness,
    innerR: innerRadius,
    initialAngle: startAngle
  });
  var layerClass = clsx('recharts-sunburst', className);
  return /*#__PURE__*/React.createElement(Surface, {
    width: width,
    height: height
  }, /*#__PURE__*/React.createElement(Layer, {
    className: layerClass
  }, sectors), /*#__PURE__*/React.createElement(SetSunburstTooltipEntrySettings, {
    dataKey: dataKey,
    nameKey: nameKey,
    data: data,
    stroke: stroke,
    fill: fill,
    positions: positions,
    id: id
  }), children);
};

/**
 * The sunburst is a hierarchical chart, similar to a {@link Treemap}, plotted in polar coordinates.
 * Sunburst charts effectively convey the hierarchical relationships and proportions within each level.
 * It is easy to see all the middle layers in the hierarchy, which might get lost in other visualizations.
 * For some datasets, the radial layout may be more visually appealing and intuitive than a traditional {@link Treemap}.
 *
 * @consumes ResponsiveContainerContext
 * @provides TooltipEntrySettings
 */
export var SunburstChart = outsideProps => {
  var props = resolveDefaultProps(outsideProps, defaultSunburstChartProps);
  var {
    className,
    width,
    height,
    responsive,
    style,
    id: externalId,
    throttleDelay,
    throttledEvents
  } = props;
  var [tooltipPortal, setTooltipPortal] = useState(null);
  return /*#__PURE__*/React.createElement(RechartsStoreProvider, {
    preloadedState: preloadedState,
    reduxStoreName: className !== null && className !== void 0 ? className : 'SunburstChart'
  }, /*#__PURE__*/React.createElement(ReportChartSize, {
    width: width,
    height: height
  }), /*#__PURE__*/React.createElement(ReportChartMargin, {
    margin: defaultSunburstMargin
  }), /*#__PURE__*/React.createElement(ReportEventSettings, {
    throttleDelay: throttleDelay,
    throttledEvents: throttledEvents
  }), /*#__PURE__*/React.createElement(TooltipPortalContext.Provider, {
    value: tooltipPortal
  }, /*#__PURE__*/React.createElement(RechartsWrapper, {
    className: className,
    width: width,
    height: height,
    responsive: responsive,
    style: style,
    ref: node => {
      if (tooltipPortal == null && node != null) {
        setTooltipPortal(node);
      }
    },
    onMouseEnter: undefined,
    onMouseLeave: undefined,
    onClick: undefined,
    onMouseMove: undefined,
    onMouseDown: undefined,
    onMouseUp: undefined,
    onContextMenu: undefined,
    onDoubleClick: undefined,
    onTouchStart: undefined,
    onTouchMove: undefined,
    onTouchEnd: undefined
  }, /*#__PURE__*/React.createElement(RegisterGraphicalItemId, {
    id: externalId,
    type: "sunburst"
  }, id => /*#__PURE__*/React.createElement(SunburstChartImpl, _extends({}, props, {
    id: id
  }))))));
};