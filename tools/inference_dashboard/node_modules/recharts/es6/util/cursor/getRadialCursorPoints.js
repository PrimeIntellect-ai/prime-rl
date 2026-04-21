import { polarToCartesian } from '../PolarUtils';
/**
 * Only applicable for radial layouts
 * @param {Object} activeCoordinate ChartCoordinate
 * @returns {Object} RadialCursorPoints
 */
export function getRadialCursorPoints(activeCoordinate) {
  var {
    cx,
    cy,
    radius,
    startAngle,
    endAngle
  } = activeCoordinate;
  var startPoint = polarToCartesian(cx, cy, radius, startAngle);
  var endPoint = polarToCartesian(cx, cy, radius, endAngle);
  return {
    points: [startPoint, endPoint],
    cx,
    cy,
    radius,
    startAngle,
    endAngle
  };
}