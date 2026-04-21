export var ACCURACY = 1e-4;
var cubicBezierFactor = (c1, c2) => [0, 3 * c1, 3 * c2 - 6 * c1, 3 * c1 - 3 * c2 + 1];
var evaluatePolynomial = (params, t) => params.map((param, i) => param * t ** i).reduce((pre, curr) => pre + curr);
var cubicBezier = (c1, c2) => t => {
  var params = cubicBezierFactor(c1, c2);
  return evaluatePolynomial(params, t);
};
var derivativeCubicBezier = (c1, c2) => t => {
  var params = cubicBezierFactor(c1, c2);
  var newParams = [...params.map((param, i) => param * i).slice(1), 0];
  return evaluatePolynomial(newParams, t);
};
var parseCubicBezier = easing => {
  var _easingParts$;
  var easingParts = easing.split('(');
  if (easingParts.length !== 2 || easingParts[0] !== 'cubic-bezier') {
    return null;
  }
  var numbers = (_easingParts$ = easingParts[1]) === null || _easingParts$ === void 0 || (_easingParts$ = _easingParts$.split(')')[0]) === null || _easingParts$ === void 0 ? void 0 : _easingParts$.split(',');
  if (numbers == null || numbers.length !== 4) {
    return null;
  }
  var coords = numbers.map(x => parseFloat(x));
  return [coords[0], coords[1], coords[2], coords[3]];
};
var getBezierCoordinates = function getBezierCoordinates() {
  for (var _len = arguments.length, args = new Array(_len), _key = 0; _key < _len; _key++) {
    args[_key] = arguments[_key];
  }
  if (args.length === 1) {
    switch (args[0]) {
      case 'linear':
        return [0.0, 0.0, 1.0, 1.0];
      case 'ease':
        return [0.25, 0.1, 0.25, 1.0];
      case 'ease-in':
        return [0.42, 0.0, 1.0, 1.0];
      case 'ease-out':
        return [0.42, 0.0, 0.58, 1.0];
      case 'ease-in-out':
        return [0.0, 0.0, 0.58, 1.0];
      default:
        {
          var easing = parseCubicBezier(args[0]);
          if (easing) {
            return easing;
          }
        }
    }
  }
  if (args.length === 4) {
    return args;
  }

  // Fallback for invalid inputs. The previous implementation was buggy and would lead to NaN.
  // Returning linear easing is a safe default.
  return [0.0, 0.0, 1.0, 1.0];
};
var createBezierEasing = (x1, y1, x2, y2) => {
  var curveX = cubicBezier(x1, x2);
  var curveY = cubicBezier(y1, y2);
  var derCurveX = derivativeCubicBezier(x1, x2);
  var rangeValue = value => {
    if (value > 1) {
      return 1;
    }
    if (value < 0) {
      return 0;
    }
    return value;
  };
  var bezier = _t => {
    var t = _t > 1 ? 1 : _t;
    var x = t;
    for (var i = 0; i < 8; ++i) {
      var evalT = curveX(x) - t;
      var derVal = derCurveX(x);
      if (Math.abs(evalT - t) < ACCURACY || derVal < ACCURACY) {
        return curveY(x);
      }
      x = rangeValue(x - evalT / derVal);
    }
    return curveY(x);
  };
  bezier.isStepper = false;
  return bezier;
};

// calculate cubic-bezier using Newton's method
export var configBezier = function configBezier() {
  return createBezierEasing(...getBezierCoordinates(...arguments));
};
export var configSpring = function configSpring() {
  var config = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : {};
  var {
    stiff = 100,
    damping = 8,
    dt = 17
  } = config;
  var stepper = (currX, destX, currV) => {
    var FSpring = -(currX - destX) * stiff;
    var FDamping = currV * damping;
    var newV = currV + (FSpring - FDamping) * dt / 1000;
    var newX = currV * dt / 1000 + currX;
    if (Math.abs(newX - destX) < ACCURACY && Math.abs(newV) < ACCURACY) {
      return [destX, 0];
    }
    return [newX, newV];
  };
  stepper.isStepper = true;
  stepper.dt = dt;
  return stepper;
};
export var configEasing = easing => {
  if (typeof easing === 'string') {
    switch (easing) {
      case 'ease':
      case 'ease-in-out':
      case 'ease-out':
      case 'ease-in':
      case 'linear':
        return configBezier(easing);
      case 'spring':
        return configSpring();
      default:
        if (easing.split('(')[0] === 'cubic-bezier') {
          return configBezier(easing);
        }
    }
  }
  if (typeof easing === 'function') {
    return easing;
  }
  return null;
};