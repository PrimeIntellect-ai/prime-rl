var _excluded = ["contextPayload"];
function _extends() { return _extends = Object.assign ? Object.assign.bind() : function (n) { for (var e = 1; e < arguments.length; e++) { var t = arguments[e]; for (var r in t) ({}).hasOwnProperty.call(t, r) && (n[r] = t[r]); } return n; }, _extends.apply(null, arguments); }
function ownKeys(e, r) { var t = Object.keys(e); if (Object.getOwnPropertySymbols) { var o = Object.getOwnPropertySymbols(e); r && (o = o.filter(function (r) { return Object.getOwnPropertyDescriptor(e, r).enumerable; })), t.push.apply(t, o); } return t; }
function _objectSpread(e) { for (var r = 1; r < arguments.length; r++) { var t = null != arguments[r] ? arguments[r] : {}; r % 2 ? ownKeys(Object(t), !0).forEach(function (r) { _defineProperty(e, r, t[r]); }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(t)) : ownKeys(Object(t)).forEach(function (r) { Object.defineProperty(e, r, Object.getOwnPropertyDescriptor(t, r)); }); } return e; }
function _defineProperty(e, r, t) { return (r = _toPropertyKey(r)) in e ? Object.defineProperty(e, r, { value: t, enumerable: !0, configurable: !0, writable: !0 }) : e[r] = t, e; }
function _toPropertyKey(t) { var i = _toPrimitive(t, "string"); return "symbol" == typeof i ? i : i + ""; }
function _toPrimitive(t, r) { if ("object" != typeof t || !t) return t; var e = t[Symbol.toPrimitive]; if (void 0 !== e) { var i = e.call(t, r || "default"); if ("object" != typeof i) return i; throw new TypeError("@@toPrimitive must return a primitive value."); } return ("string" === r ? String : Number)(t); }
function _objectWithoutProperties(e, t) { if (null == e) return {}; var o, r, i = _objectWithoutPropertiesLoose(e, t); if (Object.getOwnPropertySymbols) { var n = Object.getOwnPropertySymbols(e); for (r = 0; r < n.length; r++) o = n[r], -1 === t.indexOf(o) && {}.propertyIsEnumerable.call(e, o) && (i[o] = e[o]); } return i; }
function _objectWithoutPropertiesLoose(r, e) { if (null == r) return {}; var t = {}; for (var n in r) if ({}.hasOwnProperty.call(r, n)) { if (-1 !== e.indexOf(n)) continue; t[n] = r[n]; } return t; }
import * as React from 'react';
import { useEffect } from 'react';
import { createPortal } from 'react-dom';
import { useLegendPortal } from '../context/legendPortalContext';
import { DefaultLegendContent } from './DefaultLegendContent';
import { getUniqPayload } from '../util/payload/getUniqPayload';
import { useLegendPayload } from '../context/legendPayloadContext';
import { useElementOffset } from '../util/useElementOffset';
import { useChartHeight, useChartWidth, useMargin } from '../context/chartLayoutContext';
import { setLegendSettings, setLegendSize } from '../state/legendSlice';
import { useAppDispatch } from '../state/hooks';
import { resolveDefaultProps } from '../util/resolveDefaultProps';
import { propsAreEqual } from '../util/propsAreEqual';
function defaultUniqBy(entry) {
  return entry.value;
}
function LegendContent(props) {
  var {
      contextPayload
    } = props,
    otherProps = _objectWithoutProperties(props, _excluded);
  var finalPayload = getUniqPayload(contextPayload, props.payloadUniqBy, defaultUniqBy);
  var contentProps = _objectSpread(_objectSpread({}, otherProps), {}, {
    payload: finalPayload
  });
  if (/*#__PURE__*/React.isValidElement(props.content)) {
    return /*#__PURE__*/React.cloneElement(props.content, contentProps);
  }
  if (typeof props.content === 'function') {
    return /*#__PURE__*/React.createElement(props.content, contentProps);
  }
  return /*#__PURE__*/React.createElement(DefaultLegendContent, contentProps);
}
function getDefaultPosition(style, props, margin, chartWidth, chartHeight, box) {
  var {
    layout,
    align,
    verticalAlign
  } = props;
  var hPos, vPos;
  if (!style || (style.left === undefined || style.left === null) && (style.right === undefined || style.right === null)) {
    if (align === 'center' && layout === 'vertical') {
      hPos = {
        left: ((chartWidth || 0) - box.width) / 2
      };
    } else {
      hPos = align === 'right' ? {
        right: margin && margin.right || 0
      } : {
        left: margin && margin.left || 0
      };
    }
  }
  if (!style || (style.top === undefined || style.top === null) && (style.bottom === undefined || style.bottom === null)) {
    if (verticalAlign === 'middle') {
      vPos = {
        top: ((chartHeight || 0) - box.height) / 2
      };
    } else {
      vPos = verticalAlign === 'bottom' ? {
        bottom: margin && margin.bottom || 0
      } : {
        top: margin && margin.top || 0
      };
    }
  }
  return _objectSpread(_objectSpread({}, hPos), vPos);
}
function LegendSettingsDispatcher(props) {
  var dispatch = useAppDispatch();
  useEffect(() => {
    dispatch(setLegendSettings(props));
  }, [dispatch, props]);
  return null;
}
function LegendSizeDispatcher(props) {
  var dispatch = useAppDispatch();
  useEffect(() => {
    dispatch(setLegendSize(props));
    return () => {
      dispatch(setLegendSize({
        width: 0,
        height: 0
      }));
    };
  }, [dispatch, props]);
  return null;
}
function getWidthOrHeight(layout, height, width, maxWidth) {
  if (layout === 'vertical' && height != null) {
    return {
      height
    };
  }
  if (layout === 'horizontal') {
    return {
      width: width || maxWidth
    };
  }
  return null;
}
export var legendDefaultProps = {
  align: 'center',
  iconSize: 14,
  inactiveColor: '#ccc',
  itemSorter: 'value',
  layout: 'horizontal',
  verticalAlign: 'bottom'
};

/**
 * @consumes CartesianChartContext
 * @consumes PolarChartContext
 */
function LegendImpl(outsideProps) {
  var props = resolveDefaultProps(outsideProps, legendDefaultProps);
  var contextPayload = useLegendPayload();
  var legendPortalFromContext = useLegendPortal();
  var margin = useMargin();
  var {
    width: widthFromProps,
    height: heightFromProps,
    wrapperStyle,
    portal: portalFromProps
  } = props;
  // The contextPayload is not used directly inside the hook, but we need the onBBoxUpdate call
  // when the payload changes, therefore it's here as a dependency.
  var [lastBoundingBox, updateBoundingBox] = useElementOffset([contextPayload]);
  var chartWidth = useChartWidth();
  var chartHeight = useChartHeight();
  if (chartWidth == null || chartHeight == null) {
    return null;
  }
  var maxWidth = chartWidth - ((margin === null || margin === void 0 ? void 0 : margin.left) || 0) - ((margin === null || margin === void 0 ? void 0 : margin.right) || 0);
  var widthOrHeight = getWidthOrHeight(props.layout, heightFromProps, widthFromProps, maxWidth);
  // if the user supplies their own portal, only use their defined wrapper styles
  var outerStyle = portalFromProps ? wrapperStyle : _objectSpread(_objectSpread({
    position: 'absolute',
    width: (widthOrHeight === null || widthOrHeight === void 0 ? void 0 : widthOrHeight.width) || widthFromProps || 'auto',
    height: (widthOrHeight === null || widthOrHeight === void 0 ? void 0 : widthOrHeight.height) || heightFromProps || 'auto'
  }, getDefaultPosition(wrapperStyle, props, margin, chartWidth, chartHeight, lastBoundingBox)), wrapperStyle);
  var legendPortal = portalFromProps !== null && portalFromProps !== void 0 ? portalFromProps : legendPortalFromContext;
  if (legendPortal == null || contextPayload == null) {
    return null;
  }
  var legendElement = /*#__PURE__*/React.createElement("div", {
    className: "recharts-legend-wrapper",
    style: outerStyle,
    ref: updateBoundingBox
  }, /*#__PURE__*/React.createElement(LegendSettingsDispatcher, {
    layout: props.layout,
    align: props.align,
    verticalAlign: props.verticalAlign,
    itemSorter: props.itemSorter
  }), !portalFromProps && /*#__PURE__*/React.createElement(LegendSizeDispatcher, {
    width: lastBoundingBox.width,
    height: lastBoundingBox.height
  }), /*#__PURE__*/React.createElement(LegendContent, _extends({}, props, widthOrHeight, {
    margin: margin,
    chartWidth: chartWidth,
    chartHeight: chartHeight,
    contextPayload: contextPayload
  })));
  return /*#__PURE__*/createPortal(legendElement, legendPortal);
}
export var Legend = /*#__PURE__*/React.memo(LegendImpl, propsAreEqual);
Legend.displayName = 'Legend';