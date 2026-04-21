function ownKeys(e, r) { var t = Object.keys(e); if (Object.getOwnPropertySymbols) { var o = Object.getOwnPropertySymbols(e); r && (o = o.filter(function (r) { return Object.getOwnPropertyDescriptor(e, r).enumerable; })), t.push.apply(t, o); } return t; }
function _objectSpread(e) { for (var r = 1; r < arguments.length; r++) { var t = null != arguments[r] ? arguments[r] : {}; r % 2 ? ownKeys(Object(t), !0).forEach(function (r) { _defineProperty(e, r, t[r]); }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(t)) : ownKeys(Object(t)).forEach(function (r) { Object.defineProperty(e, r, Object.getOwnPropertyDescriptor(t, r)); }); } return e; }
function _defineProperty(e, r, t) { return (r = _toPropertyKey(r)) in e ? Object.defineProperty(e, r, { value: t, enumerable: !0, configurable: !0, writable: !0 }) : e[r] = t, e; }
function _toPropertyKey(t) { var i = _toPrimitive(t, "string"); return "symbol" == typeof i ? i : i + ""; }
function _toPrimitive(t, r) { if ("object" != typeof t || !t) return t; var e = t[Symbol.toPrimitive]; if (void 0 !== e) { var i = e.call(t, r || "default"); if ("object" != typeof i) return i; throw new TypeError("@@toPrimitive must return a primitive value."); } return ("string" === r ? String : Number)(t); }
/*
 * @description: convert camel case to dash case
 * string => string
 */
export var getDashCase = name => name.replace(/([A-Z])/g, v => "-".concat(v.toLowerCase()));
export var getTransitionVal = (props, duration, easing) => props.map(prop => "".concat(getDashCase(prop), " ").concat(duration, "ms ").concat(easing)).join(',');

/**
 * Finds the intersection of keys between two objects
 * @param {object} preObj previous object
 * @param {object} nextObj next object
 * @returns an array of keys that exist in both objects
 */
export var getIntersectionKeys = (preObj, nextObj) => [Object.keys(preObj), Object.keys(nextObj)].reduce((a, b) => a.filter(c => b.includes(c)));

/**
 * Maps an object to another object
 * @param {function} fn function to map
 * @param {object} obj object to map
 * @returns mapped object
 */
export var mapObject = (fn, obj) => Object.keys(obj).reduce((res, key) => _objectSpread(_objectSpread({}, res), {}, {
  [key]: fn(key, obj[key])
}), {});