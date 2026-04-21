var _DecimalCSS;
function _defineProperty(e, r, t) { return (r = _toPropertyKey(r)) in e ? Object.defineProperty(e, r, { value: t, enumerable: !0, configurable: !0, writable: !0 }) : e[r] = t, e; }
function _toPropertyKey(t) { var i = _toPrimitive(t, "string"); return "symbol" == typeof i ? i : i + ""; }
function _toPrimitive(t, r) { if ("object" != typeof t || !t) return t; var e = t[Symbol.toPrimitive]; if (void 0 !== e) { var i = e.call(t, r || "default"); if ("object" != typeof i) return i; throw new TypeError("@@toPrimitive must return a primitive value."); } return ("string" === r ? String : Number)(t); }
import { isNan } from './DataUtils';
var MULTIPLY_OR_DIVIDE_REGEX = /(-?\d+(?:\.\d+)?[a-zA-Z%]*)([*/])(-?\d+(?:\.\d+)?[a-zA-Z%]*)/;
var ADD_OR_SUBTRACT_REGEX = /(-?\d+(?:\.\d+)?[a-zA-Z%]*)([+-])(-?\d+(?:\.\d+)?[a-zA-Z%]*)/;
var CSS_LENGTH_UNIT_REGEX = /^(px|cm|vh|vw|em|rem|%|mm|in|pt|pc|ex|ch|vmin|vmax|Q)$/;
var NUM_SPLIT_REGEX = /(-?\d+(?:\.\d+)?)([a-zA-Z%]+)?/;
var CONVERSION_RATES = {
  cm: 96 / 2.54,
  mm: 96 / 25.4,
  pt: 96 / 72,
  pc: 96 / 6,
  in: 96,
  Q: 96 / (2.54 * 40),
  px: 1
};
var FIXED_CSS_LENGTH_UNITS = ['cm', 'mm', 'pt', 'pc', 'in', 'Q', 'px'];
function isSupportedUnit(unit) {
  return FIXED_CSS_LENGTH_UNITS.includes(unit);
}
var STR_NAN = 'NaN';
function convertToPx(value, unit) {
  return value * CONVERSION_RATES[unit];
}
class DecimalCSS {
  static parse(str) {
    var _NUM_SPLIT_REGEX$exec;
    var [, numStr, unit] = (_NUM_SPLIT_REGEX$exec = NUM_SPLIT_REGEX.exec(str)) !== null && _NUM_SPLIT_REGEX$exec !== void 0 ? _NUM_SPLIT_REGEX$exec : [];
    if (numStr == null) {
      return DecimalCSS.NaN;
    }
    return new DecimalCSS(parseFloat(numStr), unit !== null && unit !== void 0 ? unit : '');
  }
  constructor(num, unit) {
    this.num = num;
    this.unit = unit;
    this.num = num;
    this.unit = unit;
    if (isNan(num)) {
      this.unit = '';
    }
    if (unit !== '' && !CSS_LENGTH_UNIT_REGEX.test(unit)) {
      this.num = NaN;
      this.unit = '';
    }
    if (isSupportedUnit(unit)) {
      this.num = convertToPx(num, unit);
      this.unit = 'px';
    }
  }
  add(other) {
    if (this.unit !== other.unit) {
      return new DecimalCSS(NaN, '');
    }
    return new DecimalCSS(this.num + other.num, this.unit);
  }
  subtract(other) {
    if (this.unit !== other.unit) {
      return new DecimalCSS(NaN, '');
    }
    return new DecimalCSS(this.num - other.num, this.unit);
  }
  multiply(other) {
    if (this.unit !== '' && other.unit !== '' && this.unit !== other.unit) {
      return new DecimalCSS(NaN, '');
    }
    return new DecimalCSS(this.num * other.num, this.unit || other.unit);
  }
  divide(other) {
    if (this.unit !== '' && other.unit !== '' && this.unit !== other.unit) {
      return new DecimalCSS(NaN, '');
    }
    return new DecimalCSS(this.num / other.num, this.unit || other.unit);
  }
  toString() {
    return "".concat(this.num).concat(this.unit);
  }
  isNaN() {
    return isNan(this.num);
  }
}
_DecimalCSS = DecimalCSS;
_defineProperty(DecimalCSS, "NaN", new _DecimalCSS(NaN, ''));
function calculateArithmetic(expr) {
  if (expr == null || expr.includes(STR_NAN)) {
    return STR_NAN;
  }
  var newExpr = expr;
  while (newExpr.includes('*') || newExpr.includes('/')) {
    var _MULTIPLY_OR_DIVIDE_R;
    var [, leftOperand, operator, rightOperand] = (_MULTIPLY_OR_DIVIDE_R = MULTIPLY_OR_DIVIDE_REGEX.exec(newExpr)) !== null && _MULTIPLY_OR_DIVIDE_R !== void 0 ? _MULTIPLY_OR_DIVIDE_R : [];
    var lTs = DecimalCSS.parse(leftOperand !== null && leftOperand !== void 0 ? leftOperand : '');
    var rTs = DecimalCSS.parse(rightOperand !== null && rightOperand !== void 0 ? rightOperand : '');
    var result = operator === '*' ? lTs.multiply(rTs) : lTs.divide(rTs);
    if (result.isNaN()) {
      return STR_NAN;
    }
    newExpr = newExpr.replace(MULTIPLY_OR_DIVIDE_REGEX, result.toString());
  }
  while (newExpr.includes('+') || /.-\d+(?:\.\d+)?/.test(newExpr)) {
    var _ADD_OR_SUBTRACT_REGE;
    var [, _leftOperand, _operator, _rightOperand] = (_ADD_OR_SUBTRACT_REGE = ADD_OR_SUBTRACT_REGEX.exec(newExpr)) !== null && _ADD_OR_SUBTRACT_REGE !== void 0 ? _ADD_OR_SUBTRACT_REGE : [];
    var _lTs = DecimalCSS.parse(_leftOperand !== null && _leftOperand !== void 0 ? _leftOperand : '');
    var _rTs = DecimalCSS.parse(_rightOperand !== null && _rightOperand !== void 0 ? _rightOperand : '');
    var _result = _operator === '+' ? _lTs.add(_rTs) : _lTs.subtract(_rTs);
    if (_result.isNaN()) {
      return STR_NAN;
    }
    newExpr = newExpr.replace(ADD_OR_SUBTRACT_REGEX, _result.toString());
  }
  return newExpr;
}
var PARENTHESES_REGEX = /\(([^()]*)\)/;
function calculateParentheses(expr) {
  var newExpr = expr;
  var match;
  // eslint-disable-next-line no-cond-assign
  while ((match = PARENTHESES_REGEX.exec(newExpr)) != null) {
    var [, parentheticalExpression] = match;
    newExpr = newExpr.replace(PARENTHESES_REGEX, calculateArithmetic(parentheticalExpression));
  }
  return newExpr;
}
function evaluateExpression(expression) {
  var newExpr = expression.replace(/\s+/g, '');
  newExpr = calculateParentheses(newExpr);
  newExpr = calculateArithmetic(newExpr);
  return newExpr;
}
export function safeEvaluateExpression(expression) {
  try {
    return evaluateExpression(expression);
  } catch (_unused) {
    return STR_NAN;
  }
}
export function reduceCSSCalc(expression) {
  var result = safeEvaluateExpression(expression.slice(5, -1));
  if (result === STR_NAN) {
    return '';
  }
  return result;
}