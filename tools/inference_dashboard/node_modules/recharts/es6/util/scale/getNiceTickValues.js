/**
 * @fileOverview calculate tick values of scale
 * @author xile611, arcthur
 * @date 2015-09-17
 */
import Decimal from 'decimal.js-light';
import { getDigitCount, rangeStep } from './util/arithmetic';
/**
 * Calculate a interval of a minimum value and a maximum value
 *
 * @param  {Number} min       The minimum value
 * @param  {Number} max       The maximum value
 * @return {Array} An interval
 */
export var getValidInterval = _ref => {
  var [min, max] = _ref;
  var [validMin, validMax] = [min, max];

  // exchange
  if (min > max) {
    [validMin, validMax] = [max, min];
  }
  return [validMin, validMax];
};

/**
 * Calculate the step which is easy to understand between ticks, like 10, 20, 25
 *
 * @param  roughStep        The rough step calculated by dividing the difference by the tickCount
 * @param  allowDecimals    Allow the ticks to be decimals or not
 * @param  correctionFactor A correction factor
 * @return The step which is easy to understand between two ticks
 */
export var getAdaptiveStep = (roughStep, allowDecimals, correctionFactor) => {
  if (roughStep.lte(0)) {
    return new Decimal(0);
  }
  var digitCount = getDigitCount(roughStep.toNumber());
  // The ratio between the rough step and the smallest number which has a bigger
  // order of magnitudes than the rough step
  var digitCountValue = new Decimal(10).pow(digitCount);
  var stepRatio = roughStep.div(digitCountValue);
  // When an integer and a float multiplied, the accuracy of result may be wrong
  var stepRatioScale = digitCount !== 1 ? 0.05 : 0.1;
  var amendStepRatio = new Decimal(Math.ceil(stepRatio.div(stepRatioScale).toNumber())).add(correctionFactor).mul(stepRatioScale);
  var formatStep = amendStepRatio.mul(digitCountValue);
  return allowDecimals ? new Decimal(formatStep.toNumber()) : new Decimal(Math.ceil(formatStep.toNumber()));
};
/**
 * The snap125 step algorithm snaps to nice numbers (1, 2, 2.5, 5) at each
 * order of magnitude, producing human-friendly tick intervals like
 * 0, 5, 10, 15, 20 instead of 0, 4, 8, 12, 16.
 *
 * This is opt-in and can be enabled via the `niceTicks` prop on axis components.
 *
 * @param  roughStep        The rough step calculated by dividing the difference by the tickCount
 * @param  allowDecimals    Allow the ticks to be decimals or not
 * @param  correctionFactor A correction factor
 * @return The step which is easy to understand between two ticks
 */
export var getSnap125Step = (roughStep, allowDecimals, correctionFactor) => {
  var _NICE_STEPS$niceIdx;
  if (roughStep.lte(0)) {
    return new Decimal(0);
  }
  var NICE_STEPS = [1, 2, 2.5, 5];
  var roughNum = roughStep.toNumber();
  var exponent = Math.floor(new Decimal(roughNum).abs().log(10).toNumber());
  var magnitude = new Decimal(10).pow(exponent);

  // normalized is in the range [1, 10)
  var normalized = roughStep.div(magnitude).toNumber();

  // Find the smallest nice step >= normalized (ceiling)
  var niceIdx = NICE_STEPS.findIndex(s => s >= normalized - 1e-10);
  if (niceIdx === -1) {
    // normalized > 5 (e.g. 7.3), move to next order of magnitude
    magnitude = magnitude.mul(10);
    niceIdx = 0;
  }

  // Apply correction factor by stepping through the nice number sequence
  niceIdx += correctionFactor;
  if (niceIdx >= NICE_STEPS.length) {
    var extraMag = Math.floor(niceIdx / NICE_STEPS.length);
    niceIdx %= NICE_STEPS.length;
    magnitude = magnitude.mul(new Decimal(10).pow(extraMag));
  }
  var niceStep = (_NICE_STEPS$niceIdx = NICE_STEPS[niceIdx]) !== null && _NICE_STEPS$niceIdx !== void 0 ? _NICE_STEPS$niceIdx : 1;
  var formatStep = new Decimal(niceStep).mul(magnitude);
  return allowDecimals ? formatStep : new Decimal(Math.ceil(formatStep.toNumber()));
};

/**
 * calculate the ticks when the minimum value equals to the maximum value
 *
 * @param  value         The minimum value which is also the maximum value
 * @param  tickCount     The count of ticks
 * @param  allowDecimals Allow the ticks to be decimals or not
 * @return array of ticks
 */
export var getTickOfSingleValue = (value, tickCount, allowDecimals) => {
  var step = new Decimal(1);
  // calculate the middle value of ticks
  var middle = new Decimal(value);
  if (!middle.isint() && allowDecimals) {
    var absVal = Math.abs(value);
    if (absVal < 1) {
      // The step should be a float number when the difference is smaller than 1
      step = new Decimal(10).pow(getDigitCount(value) - 1);
      middle = new Decimal(Math.floor(middle.div(step).toNumber())).mul(step);
    } else if (absVal > 1) {
      // Return the maximum integer which is smaller than 'value' when 'value' is greater than 1
      middle = new Decimal(Math.floor(value));
    }
  } else if (value === 0) {
    middle = new Decimal(Math.floor((tickCount - 1) / 2));
  } else if (!allowDecimals) {
    middle = new Decimal(Math.floor(value));
  }
  var middleIndex = Math.floor((tickCount - 1) / 2);
  var ticks = [];
  for (var i = 0; i < tickCount; i++) {
    ticks.push(middle.add(new Decimal(i - middleIndex).mul(step)).toNumber());
  }
  return ticks;
};

/**
 * Calculate the step
 *
 * @param  min              The minimum value of an interval
 * @param  max              The maximum value of an interval
 * @param  tickCount        The count of ticks
 * @param  allowDecimals    Allow the ticks to be decimals or not
 * @param  correctionFactor A correction factor
 * @return The step, minimum value of ticks, maximum value of ticks
 */
var _calculateStep = function calculateStep(min, max, tickCount, allowDecimals) {
  var correctionFactor = arguments.length > 4 && arguments[4] !== undefined ? arguments[4] : 0;
  var stepFn = arguments.length > 5 && arguments[5] !== undefined ? arguments[5] : getAdaptiveStep;
  // dirty hack (for recharts' test)
  if (!Number.isFinite((max - min) / (tickCount - 1))) {
    return {
      step: new Decimal(0),
      tickMin: new Decimal(0),
      tickMax: new Decimal(0)
    };
  }

  // The step which is easy to understand between two ticks
  var step = stepFn(new Decimal(max).sub(min).div(tickCount - 1), allowDecimals, correctionFactor);

  // A medial value of ticks
  var middle;

  // When 0 is inside the interval, 0 should be a tick
  if (min <= 0 && max >= 0) {
    middle = new Decimal(0);
  } else {
    // calculate the middle value
    middle = new Decimal(min).add(max).div(2);
    // minus modulo value
    middle = middle.sub(new Decimal(middle).mod(step));
  }
  var belowCount = Math.ceil(middle.sub(min).div(step).toNumber());
  var upCount = Math.ceil(new Decimal(max).sub(middle).div(step).toNumber());
  var scaleCount = belowCount + upCount + 1;
  if (scaleCount > tickCount) {
    // When more ticks need to cover the interval, step should be bigger.
    return _calculateStep(min, max, tickCount, allowDecimals, correctionFactor + 1, stepFn);
  }
  if (scaleCount < tickCount) {
    // When less ticks can cover the interval, we should add some additional ticks
    upCount = max > 0 ? upCount + (tickCount - scaleCount) : upCount;
    belowCount = max > 0 ? belowCount : belowCount + (tickCount - scaleCount);
  }
  return {
    step,
    tickMin: middle.sub(new Decimal(belowCount).mul(step)),
    tickMax: middle.add(new Decimal(upCount).mul(step))
  };
};

/**
 * Calculate the ticks of an interval. Ticks can appear outside the interval
 * if it makes them more rounded and nice.
 *
 * @param tuple of [min,max] min: The minimum value, max: The maximum value
 * @param tickCount     The count of ticks
 * @param allowDecimals Allow the ticks to be decimals or not
 * @param niceTicksMode The algorithm to use for calculating nice ticks.
 * @return array of ticks
 */
export { _calculateStep as calculateStep };
export var getNiceTickValues = function getNiceTickValues(_ref2) {
  var [min, max] = _ref2;
  var tickCount = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : 6;
  var allowDecimals = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : true;
  var niceTicksMode = arguments.length > 3 && arguments[3] !== undefined ? arguments[3] : 'auto';
  // More than two ticks should be return
  var count = Math.max(tickCount, 2);
  var [cormin, cormax] = getValidInterval([min, max]);
  if (cormin === -Infinity || cormax === Infinity) {
    var _values = cormax === Infinity ? [cormin, ...Array(tickCount - 1).fill(Infinity)] : [...Array(tickCount - 1).fill(-Infinity), cormax];
    return min > max ? _values.reverse() : _values;
  }
  if (cormin === cormax) {
    return getTickOfSingleValue(cormin, tickCount, allowDecimals);
  }
  var stepFn = niceTicksMode === 'snap125' ? getSnap125Step : getAdaptiveStep;

  // Get the step between two ticks
  var {
    step,
    tickMin,
    tickMax
  } = _calculateStep(cormin, cormax, count, allowDecimals, 0, stepFn);
  var values = rangeStep(tickMin, tickMax.add(new Decimal(0.1).mul(step)), step);
  return min > max ? values.reverse() : values;
};

/**
 * Calculate the ticks of an interval.
 * Ticks will be constrained to the interval [min, max] even if it makes them less rounded and nice.
 *
 * @param tuple of [min,max] min: The minimum value, max: The maximum value
 * @param tickCount     The count of ticks. This function may return less than tickCount ticks if the interval is too small.
 * @param allowDecimals Allow the ticks to be decimals or not
 * @param niceTicksMode          The algorithm to use for calculating nice ticks. See {@link NiceTicksAlgorithm}.
 * @return array of ticks
 */
export var getTickValuesFixedDomain = function getTickValuesFixedDomain(_ref3, tickCount) {
  var [min, max] = _ref3;
  var allowDecimals = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : true;
  var niceTicksMode = arguments.length > 3 && arguments[3] !== undefined ? arguments[3] : 'auto';
  // More than two ticks should be return
  var [cormin, cormax] = getValidInterval([min, max]);
  if (cormin === -Infinity || cormax === Infinity) {
    return [min, max];
  }
  if (cormin === cormax) {
    return [cormin];
  }
  var stepFn = niceTicksMode === 'snap125' ? getSnap125Step : getAdaptiveStep;
  var count = Math.max(tickCount, 2);
  var step = stepFn(new Decimal(cormax).sub(cormin).div(count - 1), allowDecimals, 0);
  var values = [...rangeStep(new Decimal(cormin), new Decimal(cormax), step), cormax];
  if (allowDecimals === false) {
    /*
     * allowDecimals is false means that we want to have integer ticks.
     * The step is guaranteed to be an integer in the code above which is great start
     * but when the first step is not an integer, it will start stepping from a decimal value anyway.
     * So we need to round all the values to integers after the fact.
     */
    values = values.map(value => Math.round(value));
  }
  return min > max ? values.reverse() : values;
};