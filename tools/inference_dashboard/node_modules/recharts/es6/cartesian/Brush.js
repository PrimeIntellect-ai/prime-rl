function _extends() { return _extends = Object.assign ? Object.assign.bind() : function (n) { for (var e = 1; e < arguments.length; e++) { var t = arguments[e]; for (var r in t) ({}).hasOwnProperty.call(t, r) && (n[r] = t[r]); } return n; }, _extends.apply(null, arguments); }
function ownKeys(e, r) { var t = Object.keys(e); if (Object.getOwnPropertySymbols) { var o = Object.getOwnPropertySymbols(e); r && (o = o.filter(function (r) { return Object.getOwnPropertyDescriptor(e, r).enumerable; })), t.push.apply(t, o); } return t; }
function _objectSpread(e) { for (var r = 1; r < arguments.length; r++) { var t = null != arguments[r] ? arguments[r] : {}; r % 2 ? ownKeys(Object(t), !0).forEach(function (r) { _defineProperty(e, r, t[r]); }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(t)) : ownKeys(Object(t)).forEach(function (r) { Object.defineProperty(e, r, Object.getOwnPropertyDescriptor(t, r)); }); } return e; }
function _defineProperty(e, r, t) { return (r = _toPropertyKey(r)) in e ? Object.defineProperty(e, r, { value: t, enumerable: !0, configurable: !0, writable: !0 }) : e[r] = t, e; }
function _toPropertyKey(t) { var i = _toPrimitive(t, "string"); return "symbol" == typeof i ? i : i + ""; }
function _toPrimitive(t, r) { if ("object" != typeof t || !t) return t; var e = t[Symbol.toPrimitive]; if (void 0 !== e) { var i = e.call(t, r || "default"); if ("object" != typeof i) return i; throw new TypeError("@@toPrimitive must return a primitive value."); } return ("string" === r ? String : Number)(t); }
import * as React from 'react';
import { Children, PureComponent, useCallback, useContext, useEffect } from 'react';
import { clsx } from 'clsx';
import { scalePoint } from 'victory-vendor/d3-scale';
import range from 'es-toolkit/compat/range';
import { Layer } from '../container/Layer';
import { Text } from '../component/Text';
import { getValueByDataKey } from '../util/ChartUtils';
import { isNumber, isNotNil } from '../util/DataUtils';
import { generatePrefixStyle } from '../util/CssPrefixUtils';
import { useChartData, useDataIndex } from '../context/chartDataContext';
import { BrushUpdateDispatchContext } from '../context/brushUpdateContext';
import { useAppDispatch, useAppSelector } from '../state/hooks';
import { setDataStartEndIndexes } from '../state/chartDataSlice';
import { setBrushSettings } from '../state/brushSlice';
import { PanoramaContextProvider } from '../context/PanoramaContext';
import { selectBrushDimensions } from '../state/selectors/brushSelectors';
import { useBrushChartSynchronisation } from '../synchronisation/useChartSynchronisation';
import { resolveDefaultProps } from '../util/resolveDefaultProps';
import { svgPropertiesNoEvents } from '../util/svgPropertiesNoEvents';

// Why is this tickFormatter different from the other TickFormatters? This one allows to return numbers too for some reason.

function DefaultTraveller(props) {
  var {
    x,
    y,
    width,
    height,
    stroke
  } = props;
  var lineY = Math.floor(y + height / 2) - 1;
  return /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("rect", {
    x: x,
    y: y,
    width: width,
    height: height,
    fill: stroke,
    stroke: "none"
  }), /*#__PURE__*/React.createElement("line", {
    x1: x + 1,
    y1: lineY,
    x2: x + width - 1,
    y2: lineY,
    fill: "none",
    stroke: "#fff"
  }), /*#__PURE__*/React.createElement("line", {
    x1: x + 1,
    y1: lineY + 2,
    x2: x + width - 1,
    y2: lineY + 2,
    fill: "none",
    stroke: "#fff"
  }));
}
function Traveller(props) {
  var {
    travellerProps,
    travellerType
  } = props;
  if (/*#__PURE__*/React.isValidElement(travellerType)) {
    // @ts-expect-error element cloning disagrees with the types (and it should)
    return /*#__PURE__*/React.cloneElement(travellerType, travellerProps);
  }
  if (typeof travellerType === 'function') {
    return travellerType(travellerProps);
  }
  return /*#__PURE__*/React.createElement(DefaultTraveller, travellerProps);
}
function getNameFromUnknown(value) {
  if (isNotNil(value) && typeof value === 'object' && 'name' in value && typeof value.name === 'string') {
    return value.name;
  }
  return undefined;
}
function getAriaLabel(data, startIndex, endIndex) {
  var start = getNameFromUnknown(data[startIndex]);
  var end = getNameFromUnknown(data[endIndex]);
  return "Min value: ".concat(start, ", Max value: ").concat(end);
}
function TravellerLayer(_ref) {
  var {
    otherProps,
    travellerX,
    id,
    onMouseEnter,
    onMouseLeave,
    onMouseDown,
    onTouchStart,
    onTravellerMoveKeyboard,
    onFocus,
    onBlur
  } = _ref;
  var {
    y,
    x: xFromProps,
    travellerWidth,
    height,
    traveller,
    ariaLabel,
    data,
    startIndex,
    endIndex
  } = otherProps;
  var x = Math.max(travellerX, xFromProps);
  var travellerProps = _objectSpread(_objectSpread({}, svgPropertiesNoEvents(otherProps)), {}, {
    x,
    y,
    width: travellerWidth,
    height
  });
  var ariaLabelBrush = ariaLabel || getAriaLabel(data, startIndex, endIndex);
  return /*#__PURE__*/React.createElement(Layer, {
    tabIndex: 0,
    role: "slider",
    "aria-label": ariaLabelBrush,
    "aria-valuenow": travellerX,
    className: "recharts-brush-traveller",
    onMouseEnter: onMouseEnter,
    onMouseLeave: onMouseLeave,
    onMouseDown: onMouseDown,
    onTouchStart: onTouchStart,
    onKeyDown: e => {
      if (!['ArrowLeft', 'ArrowRight'].includes(e.key)) {
        return;
      }
      e.preventDefault();
      e.stopPropagation();
      onTravellerMoveKeyboard(e.key === 'ArrowRight' ? 1 : -1, id);
    },
    onFocus: onFocus,
    onBlur: onBlur,
    style: {
      cursor: 'col-resize'
    }
  }, /*#__PURE__*/React.createElement(Traveller, {
    travellerType: traveller,
    travellerProps: travellerProps
  }));
}
/*
 * This one cannot be a React Component because React is not happy with it returning only string | number.
 * React wants a full React.JSX.Element but that is not compatible with Text component.
 */
function getTextOfTick(props) {
  var {
    index,
    data,
    tickFormatter,
    dataKey
  } = props;
  var text = getValueByDataKey(data[index], dataKey, index);
  return typeof tickFormatter === 'function' ? tickFormatter(text, index) : text;
}
function getIndexInRange(valueRange, x) {
  var len = valueRange.length;
  var start = 0;
  var end = len - 1;
  while (end - start > 1) {
    var middle = Math.floor((start + end) / 2);
    var middleValue = valueRange[middle];
    if (middleValue != null && middleValue > x) {
      end = middle;
    } else {
      start = middle;
    }
  }
  var endValue = valueRange[end];
  return endValue != null && x >= endValue ? end : start;
}
function getIndex(_ref2) {
  var {
    startX,
    endX,
    scaleValues,
    gap,
    data
  } = _ref2;
  var lastIndex = data.length - 1;
  var min = Math.min(startX, endX);
  var max = Math.max(startX, endX);
  var minIndex = getIndexInRange(scaleValues, min);
  var maxIndex = getIndexInRange(scaleValues, max);
  return {
    startIndex: minIndex - minIndex % gap,
    endIndex: maxIndex === lastIndex ? lastIndex : maxIndex - maxIndex % gap
  };
}
function Background(_ref3) {
  var {
    x,
    y,
    width,
    height,
    fill,
    stroke
  } = _ref3;
  return /*#__PURE__*/React.createElement("rect", {
    stroke: stroke,
    fill: fill,
    x: x,
    y: y,
    width: width,
    height: height
  });
}
function BrushText(_ref4) {
  var {
    startIndex,
    endIndex,
    y,
    height,
    travellerWidth,
    stroke,
    tickFormatter,
    dataKey,
    data,
    startX,
    endX
  } = _ref4;
  var offset = 5;
  var attrs = {
    pointerEvents: 'none',
    fill: stroke
  };
  return /*#__PURE__*/React.createElement(Layer, {
    className: "recharts-brush-texts"
  }, /*#__PURE__*/React.createElement(Text, _extends({
    textAnchor: "end",
    verticalAnchor: "middle",
    x: Math.min(startX, endX) - offset,
    y: y + height / 2
  }, attrs), getTextOfTick({
    index: startIndex,
    tickFormatter,
    dataKey,
    data
  })), /*#__PURE__*/React.createElement(Text, _extends({
    textAnchor: "start",
    verticalAnchor: "middle",
    x: Math.max(startX, endX) + travellerWidth + offset,
    y: y + height / 2
  }, attrs), getTextOfTick({
    index: endIndex,
    tickFormatter,
    dataKey,
    data
  })));
}
function Slide(_ref5) {
  var {
    y,
    height,
    stroke,
    travellerWidth,
    startX,
    endX,
    onMouseEnter,
    onMouseLeave,
    onMouseDown,
    onTouchStart
  } = _ref5;
  var x = Math.min(startX, endX) + travellerWidth;
  var width = Math.max(Math.abs(endX - startX) - travellerWidth, 0);
  return /*#__PURE__*/React.createElement("rect", {
    className: "recharts-brush-slide",
    onMouseEnter: onMouseEnter,
    onMouseLeave: onMouseLeave,
    onMouseDown: onMouseDown,
    onTouchStart: onTouchStart,
    style: {
      cursor: 'move'
    },
    stroke: "none",
    fill: stroke,
    fillOpacity: 0.2,
    x: x,
    y: y,
    width: width,
    height: height
  });
}
function Panorama(_ref6) {
  var {
    x,
    y,
    width,
    height,
    data,
    children,
    padding
  } = _ref6;
  var isPanoramic = React.Children.count(children) === 1;
  if (!isPanoramic) {
    return null;
  }
  var chartElement = Children.only(children);
  if (!chartElement) {
    return null;
  }
  return /*#__PURE__*/React.cloneElement(chartElement, {
    x,
    y,
    width,
    height,
    margin: padding,
    compact: true,
    data
  });
}
var createScale = _ref7 => {
  var {
    data,
    startIndex,
    endIndex,
    x,
    width,
    travellerWidth
  } = _ref7;
  if (!data || !data.length) {
    return {};
  }
  var len = data.length;
  var scale = scalePoint().domain(range(0, len)).range([x, x + width - travellerWidth]);
  var scaleValues = scale.domain().map(entry => scale(entry)).filter(isNotNil);
  return {
    isTextActive: false,
    isSlideMoving: false,
    isTravellerMoving: false,
    isTravellerFocused: false,
    startX: scale(startIndex),
    endX: scale(endIndex),
    scale,
    scaleValues
  };
};
var isTouch = e => e.changedTouches && !!e.changedTouches.length;
class BrushWithState extends PureComponent {
  constructor(props) {
    super(props);
    _defineProperty(this, "handleDrag", e => {
      if (this.leaveTimer) {
        clearTimeout(this.leaveTimer);
        this.leaveTimer = null;
      }
      if (this.state.isTravellerMoving) {
        this.handleTravellerMove(e);
      } else if (this.state.isSlideMoving) {
        this.handleSlideDrag(e);
      }
    });
    _defineProperty(this, "handleTouchMove", e => {
      var _e$changedTouches;
      var touch = (_e$changedTouches = e.changedTouches) === null || _e$changedTouches === void 0 ? void 0 : _e$changedTouches[0];
      if (touch != null) {
        this.handleDrag(touch);
      }
    });
    _defineProperty(this, "handleDragEnd", () => {
      this.setState({
        isTravellerMoving: false,
        isSlideMoving: false
      }, () => {
        var {
          endIndex,
          onDragEnd,
          startIndex
        } = this.props;
        onDragEnd === null || onDragEnd === void 0 || onDragEnd({
          endIndex,
          startIndex
        });
      });
      this.detachDragEndListener();
    });
    _defineProperty(this, "handleLeaveWrapper", () => {
      if (this.state.isTravellerMoving || this.state.isSlideMoving) {
        this.leaveTimer = window.setTimeout(this.handleDragEnd, this.props.leaveTimeOut);
      }
    });
    _defineProperty(this, "handleEnterSlideOrTraveller", () => {
      this.setState({
        isTextActive: true
      });
    });
    _defineProperty(this, "handleLeaveSlideOrTraveller", () => {
      this.setState({
        isTextActive: false
      });
    });
    _defineProperty(this, "handleSlideDragStart", e => {
      var event = isTouch(e) ? e.changedTouches[0] : e;
      if (event == null) {
        return;
      }
      this.setState({
        isTravellerMoving: false,
        isSlideMoving: true,
        slideMoveStartX: event.pageX
      });
      this.attachDragEndListener();
    });
    _defineProperty(this, "handleTravellerMoveKeyboard", (direction, id) => {
      var {
        data,
        gap,
        startIndex,
        endIndex
      } = this.props;
      // scaleValues are a list of coordinates. For example: [65, 250, 435, 620, 805, 990].
      var {
        scaleValues,
        startX,
        endX
      } = this.state;
      if (scaleValues == null) {
        return;
      }

      // unless we search for the closest scaleValue to the current coordinate
      // we need to move travelers via index when using the keyboard
      var currentIndex = -1;
      if (id === 'startX') {
        currentIndex = startIndex;
      } else if (id === 'endX') {
        currentIndex = endIndex;
      }
      if (currentIndex < 0 || currentIndex >= data.length) {
        return;
      }
      var newIndex = currentIndex + direction;
      if (newIndex === -1 || newIndex >= scaleValues.length) {
        return;
      }
      var newScaleValue = scaleValues[newIndex];
      if (newScaleValue == null) {
        return;
      }

      // Prevent travellers from being on top of each other or overlapping
      if (id === 'startX' && newScaleValue >= endX || id === 'endX' && newScaleValue <= startX) {
        return;
      }
      this.setState(
      // @ts-expect-error not sure why typescript is not happy with this, partial update is fine in React
      {
        [id]: newScaleValue
      }, () => {
        this.props.onChange(getIndex({
          startX: this.state.startX,
          endX: this.state.endX,
          data,
          gap,
          scaleValues
        }));
      });
    });
    this.travellerDragStartHandlers = {
      startX: this.handleTravellerDragStart.bind(this, 'startX'),
      endX: this.handleTravellerDragStart.bind(this, 'endX')
    };
    this.state = {
      brushMoveStartX: 0,
      movingTravellerId: undefined,
      endX: 0,
      startX: 0,
      slideMoveStartX: 0
    };
  }
  static getDerivedStateFromProps(nextProps, prevState) {
    var {
      data,
      width,
      x,
      travellerWidth,
      startIndex,
      endIndex,
      startIndexControlledFromProps,
      endIndexControlledFromProps
    } = nextProps;
    if (data !== prevState.prevData) {
      return _objectSpread({
        prevData: data,
        prevTravellerWidth: travellerWidth,
        prevX: x,
        prevWidth: width
      }, data && data.length ? createScale({
        data,
        width,
        x,
        travellerWidth,
        startIndex,
        endIndex
      }) : {
        scale: undefined,
        scaleValues: undefined
      });
    }
    var prevScale = prevState.scale;
    if (prevScale && (width !== prevState.prevWidth || x !== prevState.prevX || travellerWidth !== prevState.prevTravellerWidth)) {
      prevScale.range([x, x + width - travellerWidth]);
      var scaleValues = prevScale.domain().map(entry => prevScale(entry)).filter(value => value != null);
      return {
        prevData: data,
        prevTravellerWidth: travellerWidth,
        prevX: x,
        prevWidth: width,
        startX: prevScale(nextProps.startIndex),
        endX: prevScale(nextProps.endIndex),
        scaleValues
      };
    }
    if (prevState.scale && !prevState.isSlideMoving && !prevState.isTravellerMoving && !prevState.isTravellerFocused && !prevState.isTextActive) {
      /*
       * If the startIndex or endIndex are controlled from the outside,
       * we need to keep the startX and end up to date.
       * Also we do not want to do that while user is interacting in the brush,
       * because this will trigger re-render and interrupt the drag&drop.
       */
      if (startIndexControlledFromProps != null && prevState.prevStartIndexControlledFromProps !== startIndexControlledFromProps) {
        return {
          startX: prevState.scale(startIndexControlledFromProps),
          prevStartIndexControlledFromProps: startIndexControlledFromProps
        };
      }
      if (endIndexControlledFromProps != null && prevState.prevEndIndexControlledFromProps !== endIndexControlledFromProps) {
        return {
          endX: prevState.scale(endIndexControlledFromProps),
          prevEndIndexControlledFromProps: endIndexControlledFromProps
        };
      }
    }
    return null;
  }
  componentWillUnmount() {
    if (this.leaveTimer) {
      clearTimeout(this.leaveTimer);
      this.leaveTimer = null;
    }
    this.detachDragEndListener();
  }
  attachDragEndListener() {
    window.addEventListener('mouseup', this.handleDragEnd, true);
    window.addEventListener('touchend', this.handleDragEnd, true);
    window.addEventListener('mousemove', this.handleDrag, true);
  }
  detachDragEndListener() {
    window.removeEventListener('mouseup', this.handleDragEnd, true);
    window.removeEventListener('touchend', this.handleDragEnd, true);
    window.removeEventListener('mousemove', this.handleDrag, true);
  }
  handleSlideDrag(e) {
    var {
      slideMoveStartX,
      startX,
      endX,
      scaleValues
    } = this.state;
    if (scaleValues == null) {
      return;
    }
    var {
      x,
      width,
      travellerWidth,
      startIndex,
      endIndex,
      onChange,
      data,
      gap
    } = this.props;
    var delta = e.pageX - slideMoveStartX;
    if (delta > 0) {
      delta = Math.min(delta, x + width - travellerWidth - endX, x + width - travellerWidth - startX);
    } else if (delta < 0) {
      delta = Math.max(delta, x - startX, x - endX);
    }
    var newIndex = getIndex({
      startX: startX + delta,
      endX: endX + delta,
      data,
      gap,
      scaleValues
    });
    if ((newIndex.startIndex !== startIndex || newIndex.endIndex !== endIndex) && onChange) {
      onChange(newIndex);
    }
    this.setState({
      startX: startX + delta,
      endX: endX + delta,
      slideMoveStartX: e.pageX
    });
  }
  handleTravellerDragStart(id, e) {
    var event = isTouch(e) ? e.changedTouches[0] : e;
    if (event == null) {
      return;
    }
    this.setState({
      isSlideMoving: false,
      isTravellerMoving: true,
      movingTravellerId: id,
      brushMoveStartX: event.pageX
    });
    this.attachDragEndListener();
  }
  handleTravellerMove(e) {
    var {
      brushMoveStartX,
      movingTravellerId,
      endX,
      startX,
      scaleValues
    } = this.state;
    if (movingTravellerId == null || scaleValues == null) {
      return;
    }
    var prevValue = this.state[movingTravellerId];
    var {
      x,
      width,
      travellerWidth,
      onChange,
      gap,
      data
    } = this.props;
    var params = {
      startX: this.state.startX,
      endX: this.state.endX,
      data,
      gap,
      scaleValues
    };
    var delta = e.pageX - brushMoveStartX;
    if (delta > 0) {
      delta = Math.min(delta, x + width - travellerWidth - prevValue);
    } else if (delta < 0) {
      delta = Math.max(delta, x - prevValue);
    }
    params[movingTravellerId] = prevValue + delta;
    var newIndex = getIndex(params);
    var {
      startIndex,
      endIndex
    } = newIndex;
    var isFullGap = () => {
      var lastIndex = data.length - 1;
      if (movingTravellerId === 'startX' && (endX > startX ? startIndex % gap === 0 : endIndex % gap === 0) || endX < startX && endIndex === lastIndex || movingTravellerId === 'endX' && (endX > startX ? endIndex % gap === 0 : startIndex % gap === 0) || endX > startX && endIndex === lastIndex) {
        return true;
      }
      return false;
    };
    this.setState(
    // @ts-expect-error not sure why typescript is not happy with this, partial update is fine in React
    {
      [movingTravellerId]: prevValue + delta,
      brushMoveStartX: e.pageX
    }, () => {
      if (onChange) {
        if (isFullGap()) {
          onChange(newIndex);
        }
      }
    });
  }
  render() {
    var {
      data,
      className,
      children,
      x,
      y,
      dy,
      width,
      height,
      alwaysShowText,
      fill,
      stroke,
      startIndex,
      endIndex,
      travellerWidth,
      tickFormatter,
      dataKey,
      padding
    } = this.props;
    var {
      startX,
      endX,
      isTextActive,
      isSlideMoving,
      isTravellerMoving,
      isTravellerFocused
    } = this.state;
    if (!data || !data.length || !isNumber(x) || !isNumber(y) || !isNumber(width) || !isNumber(height) || width <= 0 || height <= 0) {
      return null;
    }
    var layerClass = clsx('recharts-brush', className);
    var style = generatePrefixStyle('userSelect', 'none');
    var calculatedY = y + (dy !== null && dy !== void 0 ? dy : 0);
    return /*#__PURE__*/React.createElement(Layer, {
      className: layerClass,
      onMouseLeave: this.handleLeaveWrapper,
      onTouchMove: this.handleTouchMove,
      style: style
    }, /*#__PURE__*/React.createElement(Background, {
      x: x,
      y: calculatedY,
      width: width,
      height: height,
      fill: fill,
      stroke: stroke
    }), /*#__PURE__*/React.createElement(PanoramaContextProvider, null, /*#__PURE__*/React.createElement(Panorama, {
      x: x,
      y: calculatedY,
      width: width,
      height: height,
      data: data,
      padding: padding
    }, children)), /*#__PURE__*/React.createElement(Slide, {
      y: calculatedY,
      height: height,
      stroke: stroke,
      travellerWidth: travellerWidth,
      startX: startX,
      endX: endX,
      onMouseEnter: this.handleEnterSlideOrTraveller,
      onMouseLeave: this.handleLeaveSlideOrTraveller,
      onMouseDown: this.handleSlideDragStart,
      onTouchStart: this.handleSlideDragStart
    }), /*#__PURE__*/React.createElement(TravellerLayer, {
      travellerX: startX,
      id: "startX",
      otherProps: _objectSpread(_objectSpread({}, this.props), {}, {
        y: calculatedY
      }),
      onMouseEnter: this.handleEnterSlideOrTraveller,
      onMouseLeave: this.handleLeaveSlideOrTraveller,
      onMouseDown: this.travellerDragStartHandlers.startX,
      onTouchStart: this.travellerDragStartHandlers.startX,
      onTravellerMoveKeyboard: this.handleTravellerMoveKeyboard,
      onFocus: () => {
        this.setState({
          isTravellerFocused: true
        });
      },
      onBlur: () => {
        this.setState({
          isTravellerFocused: false
        });
      }
    }), /*#__PURE__*/React.createElement(TravellerLayer, {
      travellerX: endX,
      id: "endX",
      otherProps: _objectSpread(_objectSpread({}, this.props), {}, {
        y: calculatedY
      }),
      onMouseEnter: this.handleEnterSlideOrTraveller,
      onMouseLeave: this.handleLeaveSlideOrTraveller,
      onMouseDown: this.travellerDragStartHandlers.endX,
      onTouchStart: this.travellerDragStartHandlers.endX,
      onTravellerMoveKeyboard: this.handleTravellerMoveKeyboard,
      onFocus: () => {
        this.setState({
          isTravellerFocused: true
        });
      },
      onBlur: () => {
        this.setState({
          isTravellerFocused: false
        });
      }
    }), (isTextActive || isSlideMoving || isTravellerMoving || isTravellerFocused || alwaysShowText) && /*#__PURE__*/React.createElement(BrushText, {
      startIndex: startIndex,
      endIndex: endIndex,
      y: calculatedY,
      height: height,
      travellerWidth: travellerWidth,
      stroke: stroke,
      tickFormatter: tickFormatter,
      dataKey: dataKey,
      data: data,
      startX: startX,
      endX: endX
    }));
  }
}
function BrushInternal(props) {
  var dispatch = useAppDispatch();
  var chartData = useChartData();
  var dataIndexes = useDataIndex();
  var onChangeFromContext = useContext(BrushUpdateDispatchContext);
  var onChangeFromProps = props.onChange;
  var {
    startIndex: startIndexFromProps,
    endIndex: endIndexFromProps
  } = props;
  useEffect(() => {
    // start and end index can be controlled from props, and we need them to stay up-to-date in the Redux state too
    dispatch(setDataStartEndIndexes({
      startIndex: startIndexFromProps,
      endIndex: endIndexFromProps
    }));
  }, [dispatch, endIndexFromProps, startIndexFromProps]);
  useBrushChartSynchronisation();
  var onChange = useCallback(nextState => {
    if (dataIndexes == null) {
      return;
    }
    var {
      startIndex,
      endIndex
    } = dataIndexes;
    if (nextState.startIndex !== startIndex || nextState.endIndex !== endIndex) {
      onChangeFromContext === null || onChangeFromContext === void 0 || onChangeFromContext(nextState);
      onChangeFromProps === null || onChangeFromProps === void 0 || onChangeFromProps(nextState);
      dispatch(setDataStartEndIndexes(nextState));
    }
  }, [onChangeFromProps, onChangeFromContext, dispatch, dataIndexes]);
  var brushDimensions = useAppSelector(selectBrushDimensions);
  if (brushDimensions == null || dataIndexes == null || chartData == null || !chartData.length) {
    return null;
  }
  var {
    startIndex,
    endIndex
  } = dataIndexes;
  var {
    x,
    y,
    width
  } = brushDimensions;
  var contextProperties = {
    data: chartData,
    x,
    y,
    width,
    startIndex,
    endIndex,
    onChange
  };
  return /*#__PURE__*/React.createElement(BrushWithState, _extends({}, props, contextProperties, {
    startIndexControlledFromProps: startIndexFromProps !== null && startIndexFromProps !== void 0 ? startIndexFromProps : undefined,
    endIndexControlledFromProps: endIndexFromProps !== null && endIndexFromProps !== void 0 ? endIndexFromProps : undefined
  }));
}
function BrushSettingsDispatcher(props) {
  var dispatch = useAppDispatch();
  useEffect(() => {
    dispatch(setBrushSettings(props));
    return () => {
      dispatch(setBrushSettings(null));
    };
  }, [dispatch, props]);
  return null;
}
export var defaultBrushProps = {
  height: 40,
  travellerWidth: 5,
  gap: 1,
  fill: '#fff',
  stroke: '#666',
  padding: {
    top: 1,
    right: 1,
    bottom: 1,
    left: 1
  },
  leaveTimeOut: 1000,
  alwaysShowText: false
};

/**
 * Renders a scrollbar that allows the user to zoom and pan in the chart along its XAxis.
 * It also allows you to render a small overview of the chart inside the brush that is always visible
 * and shows the full data set so that the user can see where they are zoomed in.
 *
 * If a chart is synchronized with other charts using the `syncId` prop on the chart,
 * the brush will also synchronize the zooming and panning between all synchronized charts.
 *
 * @see {@link https://recharts.github.io/en-US/examples/BrushBarChart/ BarChart with Brush}
 * @see {@link https://recharts.github.io/en-US/examples/SynchronizedLineChart/ Synchronized Brush}
 *
 * @consumes CartesianChartContext
 */
export function Brush(outsideProps) {
  var props = resolveDefaultProps(outsideProps, defaultBrushProps);
  return /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement(BrushSettingsDispatcher, {
    height: props.height,
    x: props.x,
    y: props.y,
    width: props.width,
    padding: props.padding
  }), /*#__PURE__*/React.createElement(BrushInternal, props));
}
Brush.displayName = 'Brush';