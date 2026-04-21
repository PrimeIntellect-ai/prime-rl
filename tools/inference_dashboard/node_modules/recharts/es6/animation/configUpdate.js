function ownKeys(e, r) { var t = Object.keys(e); if (Object.getOwnPropertySymbols) { var o = Object.getOwnPropertySymbols(e); r && (o = o.filter(function (r) { return Object.getOwnPropertyDescriptor(e, r).enumerable; })), t.push.apply(t, o); } return t; }
function _objectSpread(e) { for (var r = 1; r < arguments.length; r++) { var t = null != arguments[r] ? arguments[r] : {}; r % 2 ? ownKeys(Object(t), !0).forEach(function (r) { _defineProperty(e, r, t[r]); }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(t)) : ownKeys(Object(t)).forEach(function (r) { Object.defineProperty(e, r, Object.getOwnPropertyDescriptor(t, r)); }); } return e; }
function _defineProperty(e, r, t) { return (r = _toPropertyKey(r)) in e ? Object.defineProperty(e, r, { value: t, enumerable: !0, configurable: !0, writable: !0 }) : e[r] = t, e; }
function _toPropertyKey(t) { var i = _toPrimitive(t, "string"); return "symbol" == typeof i ? i : i + ""; }
function _toPrimitive(t, r) { if ("object" != typeof t || !t) return t; var e = t[Symbol.toPrimitive]; if (void 0 !== e) { var i = e.call(t, r || "default"); if ("object" != typeof i) return i; throw new TypeError("@@toPrimitive must return a primitive value."); } return ("string" === r ? String : Number)(t); }
import { getIntersectionKeys, mapObject } from './util';
export var alpha = (begin, end, k) => begin + (end - begin) * k;
var needContinue = _ref => {
  var {
    from,
    to
  } = _ref;
  return from !== to;
};
/*
 * @description: cal new from value and velocity in each stepper
 * @return: { [styleProperty]: { from, to, velocity } }
 */
var calStepperVals = (easing, preVals, steps) => {
  var nextStepVals = mapObject((key, val) => {
    if (needContinue(val)) {
      var [newX, newV] = easing(val.from, val.to, val.velocity);
      return _objectSpread(_objectSpread({}, val), {}, {
        from: newX,
        velocity: newV
      });
    }
    return val;
  }, preVals);
  if (steps < 1) {
    return mapObject((key, val) => {
      if (needContinue(val) && nextStepVals[key] != null) {
        return _objectSpread(_objectSpread({}, val), {}, {
          velocity: alpha(val.velocity, nextStepVals[key].velocity, steps),
          from: alpha(val.from, nextStepVals[key].from, steps)
        });
      }
      return val;
    }, preVals);
  }
  return calStepperVals(easing, nextStepVals, steps - 1);
};
function createStepperUpdate(from, to, easing, interKeys, render, timeoutController) {
  var preTime;
  var stepperStyle = interKeys.reduce((res, key) => _objectSpread(_objectSpread({}, res), {}, {
    [key]: {
      from: from[key],
      velocity: 0,
      to: to[key]
    }
  }), {});
  var getCurrStyle = () => mapObject((key, val) => val.from, stepperStyle);
  var shouldStopAnimation = () => !Object.values(stepperStyle).filter(needContinue).length;
  var stopAnimation = null;
  var stepperUpdate = now => {
    if (!preTime) {
      preTime = now;
    }
    var deltaTime = now - preTime;
    var steps = deltaTime / easing.dt;
    stepperStyle = calStepperVals(easing, stepperStyle, steps);
    // get union set and add compatible prefix
    render(_objectSpread(_objectSpread(_objectSpread({}, from), to), getCurrStyle()));
    preTime = now;
    if (!shouldStopAnimation()) {
      stopAnimation = timeoutController.setTimeout(stepperUpdate);
    }
  };

  // return start animation method
  return () => {
    stopAnimation = timeoutController.setTimeout(stepperUpdate);

    // return stop animation method
    return () => {
      var _stopAnimation;
      (_stopAnimation = stopAnimation) === null || _stopAnimation === void 0 || _stopAnimation();
    };
  };
}
function createTimingUpdate(from, to, easing, duration, interKeys, render, timeoutController) {
  var stopAnimation = null;
  var timingStyle = interKeys.reduce((res, key) => {
    var fromElement = from[key];
    var toElement = to[key];
    if (fromElement == null || toElement == null) {
      return res;
    }
    return _objectSpread(_objectSpread({}, res), {}, {
      [key]: [fromElement, toElement]
    });
  }, {});
  var beginTime;
  var timingUpdate = now => {
    if (!beginTime) {
      beginTime = now;
    }
    var t = (now - beginTime) / duration;
    var currStyle = mapObject((key, val) => alpha(...val, easing(t)), timingStyle);

    // get union set and add compatible prefix
    render(_objectSpread(_objectSpread(_objectSpread({}, from), to), currStyle));
    if (t < 1) {
      stopAnimation = timeoutController.setTimeout(timingUpdate);
    } else {
      var finalStyle = mapObject((key, val) => alpha(...val, easing(1)), timingStyle);
      render(_objectSpread(_objectSpread(_objectSpread({}, from), to), finalStyle));
    }
  };

  // return start animation method
  return () => {
    stopAnimation = timeoutController.setTimeout(timingUpdate);

    // return stop animation method
    return () => {
      var _stopAnimation2;
      (_stopAnimation2 = stopAnimation) === null || _stopAnimation2 === void 0 || _stopAnimation2();
    };
  };
}

// configure update function
// eslint-disable-next-line import/no-default-export
export default (from, to, easing, duration, render, timeoutController) => {
  var interKeys = getIntersectionKeys(from, to);
  if (easing == null) {
    // no animation, just set to final state
    return () => {
      render(_objectSpread(_objectSpread({}, from), to));
      return () => {};
    };
  }
  return easing.isStepper === true ? createStepperUpdate(from, to, easing, interKeys, render, timeoutController) : createTimingUpdate(from, to, easing, duration, interKeys, render, timeoutController);
};