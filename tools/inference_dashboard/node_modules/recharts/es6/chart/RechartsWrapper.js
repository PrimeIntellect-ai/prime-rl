function ownKeys(e, r) { var t = Object.keys(e); if (Object.getOwnPropertySymbols) { var o = Object.getOwnPropertySymbols(e); r && (o = o.filter(function (r) { return Object.getOwnPropertyDescriptor(e, r).enumerable; })), t.push.apply(t, o); } return t; }
function _objectSpread(e) { for (var r = 1; r < arguments.length; r++) { var t = null != arguments[r] ? arguments[r] : {}; r % 2 ? ownKeys(Object(t), !0).forEach(function (r) { _defineProperty(e, r, t[r]); }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(t)) : ownKeys(Object(t)).forEach(function (r) { Object.defineProperty(e, r, Object.getOwnPropertyDescriptor(t, r)); }); } return e; }
function _defineProperty(e, r, t) { return (r = _toPropertyKey(r)) in e ? Object.defineProperty(e, r, { value: t, enumerable: !0, configurable: !0, writable: !0 }) : e[r] = t, e; }
function _toPropertyKey(t) { var i = _toPrimitive(t, "string"); return "symbol" == typeof i ? i : i + ""; }
function _toPrimitive(t, r) { if ("object" != typeof t || !t) return t; var e = t[Symbol.toPrimitive]; if (void 0 !== e) { var i = e.call(t, r || "default"); if ("object" != typeof i) return i; throw new TypeError("@@toPrimitive must return a primitive value."); } return ("string" === r ? String : Number)(t); }
function _extends() { return _extends = Object.assign ? Object.assign.bind() : function (n) { for (var e = 1; e < arguments.length; e++) { var t = arguments[e]; for (var r in t) ({}).hasOwnProperty.call(t, r) && (n[r] = t[r]); } return n; }, _extends.apply(null, arguments); }
import * as React from 'react';
import { forwardRef, useCallback, useEffect, useRef, useState } from 'react';
import { clsx } from 'clsx';
import { mouseLeaveChart } from '../state/tooltipSlice';
import { useAppDispatch } from '../state/hooks';
import { mouseClickAction, mouseMoveAction } from '../state/mouseEventsMiddleware';
import { useSynchronisedEventsFromOtherCharts } from '../synchronisation/useChartSynchronisation';
import { focusAction, keyDownAction, blurAction } from '../state/keyboardEventsMiddleware';
import { useReportScale } from '../util/useReportScale';
import { externalEventAction } from '../state/externalEventsMiddleware';
import { touchEventAction } from '../state/touchEventsMiddleware';
import { TooltipPortalContext } from '../context/tooltipPortalContext';
import { LegendPortalContext } from '../context/legendPortalContext';
import { ReportChartSize } from '../context/chartLayoutContext';
import { useResponsiveContainerContext } from '../component/ResponsiveContainer';
var EventSynchronizer = () => {
  useSynchronisedEventsFromOtherCharts();
  return null;
};
function getNumberOrZero(value) {
  if (typeof value === 'number') {
    return value;
  }
  if (typeof value === 'string') {
    var parsed = parseFloat(value);
    if (!Number.isNaN(parsed)) {
      return parsed;
    }
  }
  return 0;
}
var ResponsiveDiv = /*#__PURE__*/forwardRef((props, ref) => {
  var _props$style, _props$style2;
  var observerRef = useRef(null);
  var [sizes, setSizes] = useState({
    containerWidth: getNumberOrZero((_props$style = props.style) === null || _props$style === void 0 ? void 0 : _props$style.width),
    containerHeight: getNumberOrZero((_props$style2 = props.style) === null || _props$style2 === void 0 ? void 0 : _props$style2.height)
  });
  var setContainerSize = useCallback((newWidth, newHeight) => {
    setSizes(prevState => {
      var roundedWidth = Math.round(newWidth);
      var roundedHeight = Math.round(newHeight);
      if (prevState.containerWidth === roundedWidth && prevState.containerHeight === roundedHeight) {
        return prevState;
      }
      return {
        containerWidth: roundedWidth,
        containerHeight: roundedHeight
      };
    });
  }, []);
  var innerRef = useCallback(node => {
    // 1. First, call the external ref if it was provided
    if (typeof ref === 'function') {
      ref(node);
    }

    // 2. Disconnect any previously active ResizeObserver instance to prevent memory leaks
    if (observerRef.current != null) {
      observerRef.current.disconnect();
      observerRef.current = null;
    }

    // 3. Initiate a new ResizeObserver on the valid DOM node
    if (node != null && typeof ResizeObserver !== 'undefined') {
      var {
        width: containerWidth,
        height: containerHeight
      } = node.getBoundingClientRect();
      setContainerSize(containerWidth, containerHeight);
      var callback = entries => {
        var entry = entries[0];
        if (entry == null) {
          return;
        }
        var {
          width,
          height
        } = entry.contentRect;
        setContainerSize(width, height);
      };
      var observer = new ResizeObserver(callback);
      observer.observe(node);
      observerRef.current = observer;
    }
  }, [ref, setContainerSize]);
  useEffect(() => {
    return () => {
      var observer = observerRef.current;
      if (observer != null) {
        observer.disconnect();
      }
    };
  }, [setContainerSize]);
  return /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement(ReportChartSize, {
    width: sizes.containerWidth,
    height: sizes.containerHeight
  }), /*#__PURE__*/React.createElement("div", _extends({
    ref: innerRef
  }, props)));
});
var ReadSizeOnceDiv = /*#__PURE__*/forwardRef((props, ref) => {
  var {
    width,
    height
  } = props;
  var [sizes, setSizes] = useState({
    containerWidth: getNumberOrZero(width),
    containerHeight: getNumberOrZero(height)
  });
  var setContainerSize = useCallback((newWidth, newHeight) => {
    setSizes(prevState => {
      var roundedWidth = Math.round(newWidth);
      var roundedHeight = Math.round(newHeight);
      if (prevState.containerWidth === roundedWidth && prevState.containerHeight === roundedHeight) {
        return prevState;
      }
      return {
        containerWidth: roundedWidth,
        containerHeight: roundedHeight
      };
    });
  }, []);
  var innerRef = useCallback(node => {
    if (typeof ref === 'function') {
      ref(node);
    }
    if (node != null) {
      var {
        width: containerWidth,
        height: containerHeight
      } = node.getBoundingClientRect();
      setContainerSize(containerWidth, containerHeight);
    }
  }, [ref, setContainerSize]);
  return /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement(ReportChartSize, {
    width: sizes.containerWidth,
    height: sizes.containerHeight
  }), /*#__PURE__*/React.createElement("div", _extends({
    ref: innerRef
  }, props)));
});
var StaticDiv = /*#__PURE__*/forwardRef((props, ref) => {
  var {
    width,
    height
  } = props;
  return /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement(ReportChartSize, {
    width: width,
    height: height
  }), /*#__PURE__*/React.createElement("div", _extends({
    ref: ref
  }, props)));
});
var NonResponsiveDiv = /*#__PURE__*/forwardRef((props, ref) => {
  var {
    width,
    height
  } = props;
  // When width or height are percentages or CSS short names, read size from DOM once
  if (typeof width === 'string' || typeof height === 'string') {
    return /*#__PURE__*/React.createElement(ReadSizeOnceDiv, _extends({}, props, {
      ref: ref
    }));
  }
  // When both are numbers, use them directly
  if (typeof width === 'number' && typeof height === 'number') {
    return /*#__PURE__*/React.createElement(StaticDiv, _extends({}, props, {
      width: width,
      height: height,
      ref: ref
    }));
  }
  // When width/height are undefined, render wrapper div without reporting size
  // This results in no SVG being rendered (intentional for backwards compatibility)
  return /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement(ReportChartSize, {
    width: width,
    height: height
  }), /*#__PURE__*/React.createElement("div", _extends({
    ref: ref
  }, props)));
});
function getWrapperDivComponent(responsive) {
  return responsive ? ResponsiveDiv : NonResponsiveDiv;
}
export var RechartsWrapper = /*#__PURE__*/forwardRef((props, ref) => {
  var {
    children,
    className,
    height: heightFromProps,
    onClick,
    onContextMenu,
    onDoubleClick,
    onMouseDown,
    onMouseEnter,
    onMouseLeave,
    onMouseMove,
    onMouseUp,
    onTouchEnd,
    onTouchMove,
    onTouchStart,
    style,
    width: widthFromProps,
    responsive,
    dispatchTouchEvents = true
  } = props;
  var containerRef = useRef(null);
  var dispatch = useAppDispatch();
  var [tooltipPortal, setTooltipPortal] = useState(null);
  var [legendPortal, setLegendPortal] = useState(null);
  var setScaleRef = useReportScale();
  var responsiveContainerCalculations = useResponsiveContainerContext();
  var width = (responsiveContainerCalculations === null || responsiveContainerCalculations === void 0 ? void 0 : responsiveContainerCalculations.width) > 0 ? responsiveContainerCalculations.width : widthFromProps;
  var height = (responsiveContainerCalculations === null || responsiveContainerCalculations === void 0 ? void 0 : responsiveContainerCalculations.height) > 0 ? responsiveContainerCalculations.height : heightFromProps;
  var innerRef = useCallback(node => {
    setScaleRef(node);
    if (typeof ref === 'function') {
      ref(node);
    }
    setTooltipPortal(node);
    setLegendPortal(node);
    if (node != null) {
      containerRef.current = node;
    }
  }, [setScaleRef, ref, setTooltipPortal, setLegendPortal]);
  var myOnClick = useCallback(e => {
    dispatch(mouseClickAction(e));
    dispatch(externalEventAction({
      handler: onClick,
      reactEvent: e
    }));
  }, [dispatch, onClick]);
  var myOnMouseEnter = useCallback(e => {
    dispatch(mouseMoveAction(e));
    dispatch(externalEventAction({
      handler: onMouseEnter,
      reactEvent: e
    }));
  }, [dispatch, onMouseEnter]);
  var myOnMouseLeave = useCallback(e => {
    dispatch(mouseLeaveChart());
    dispatch(externalEventAction({
      handler: onMouseLeave,
      reactEvent: e
    }));
  }, [dispatch, onMouseLeave]);
  var myOnMouseMove = useCallback(e => {
    dispatch(mouseMoveAction(e));
    dispatch(externalEventAction({
      handler: onMouseMove,
      reactEvent: e
    }));
  }, [dispatch, onMouseMove]);
  var onFocus = useCallback(() => {
    dispatch(focusAction());
  }, [dispatch]);
  var onBlur = useCallback(() => {
    dispatch(blurAction());
  }, [dispatch]);
  var onKeyDown = useCallback(e => {
    dispatch(keyDownAction(e.key));
  }, [dispatch]);
  var myOnContextMenu = useCallback(e => {
    dispatch(externalEventAction({
      handler: onContextMenu,
      reactEvent: e
    }));
  }, [dispatch, onContextMenu]);
  var myOnDoubleClick = useCallback(e => {
    dispatch(externalEventAction({
      handler: onDoubleClick,
      reactEvent: e
    }));
  }, [dispatch, onDoubleClick]);
  var myOnMouseDown = useCallback(e => {
    dispatch(externalEventAction({
      handler: onMouseDown,
      reactEvent: e
    }));
  }, [dispatch, onMouseDown]);
  var myOnMouseUp = useCallback(e => {
    dispatch(externalEventAction({
      handler: onMouseUp,
      reactEvent: e
    }));
  }, [dispatch, onMouseUp]);
  var myOnTouchStart = useCallback(e => {
    dispatch(externalEventAction({
      handler: onTouchStart,
      reactEvent: e
    }));
  }, [dispatch, onTouchStart]);

  /*
   * onTouchMove is special because it behaves different from mouse events.
   * Mouse events have 'enter' + 'leave' combo that notify us when the mouse is over
   * a certain element. Touch events don't have that; touch only gives us
   * start (finger down), end (finger up) and move (finger moving).
   * So we need to figure out which element the user is touching
   * ourselves. Fortunately, there's a convenient method for that:
   * https://developer.mozilla.org/en-US/docs/Web/API/Document/elementFromPoint
   */
  var myOnTouchMove = useCallback(e => {
    if (dispatchTouchEvents) {
      dispatch(touchEventAction(e));
    }
    dispatch(externalEventAction({
      handler: onTouchMove,
      reactEvent: e
    }));
  }, [dispatch, dispatchTouchEvents, onTouchMove]);
  var myOnTouchEnd = useCallback(e => {
    dispatch(externalEventAction({
      handler: onTouchEnd,
      reactEvent: e
    }));
  }, [dispatch, onTouchEnd]);
  var WrapperDiv = getWrapperDivComponent(responsive);
  return /*#__PURE__*/React.createElement(TooltipPortalContext.Provider, {
    value: tooltipPortal
  }, /*#__PURE__*/React.createElement(LegendPortalContext.Provider, {
    value: legendPortal
  }, /*#__PURE__*/React.createElement(WrapperDiv, {
    width: width !== null && width !== void 0 ? width : style === null || style === void 0 ? void 0 : style.width,
    height: height !== null && height !== void 0 ? height : style === null || style === void 0 ? void 0 : style.height,
    className: clsx('recharts-wrapper', className),
    style: _objectSpread({
      position: 'relative',
      cursor: 'default',
      width,
      height
    }, style),
    onClick: myOnClick,
    onContextMenu: myOnContextMenu,
    onDoubleClick: myOnDoubleClick,
    onFocus: onFocus,
    onBlur: onBlur,
    onKeyDown: onKeyDown,
    onMouseDown: myOnMouseDown,
    onMouseEnter: myOnMouseEnter,
    onMouseLeave: myOnMouseLeave,
    onMouseMove: myOnMouseMove,
    onMouseUp: myOnMouseUp,
    onTouchEnd: myOnTouchEnd,
    onTouchMove: myOnTouchMove,
    onTouchStart: myOnTouchStart,
    ref: innerRef
  }, /*#__PURE__*/React.createElement(EventSynchronizer, null), children)));
});