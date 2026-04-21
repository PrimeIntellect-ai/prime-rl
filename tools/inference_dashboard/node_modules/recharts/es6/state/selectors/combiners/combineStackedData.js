import { getStackSeriesIdentifier } from '../../../util/stacks/getStackSeriesIdentifier';
export var combineStackedData = (stackGroups, barSettings) => {
  var stackSeriesIdentifier = getStackSeriesIdentifier(barSettings);
  if (!stackGroups || stackSeriesIdentifier == null || barSettings == null) {
    return undefined;
  }
  var {
    stackId
  } = barSettings;
  if (stackId == null) {
    return undefined;
  }
  var stackGroup = stackGroups[stackId];
  if (!stackGroup) {
    return undefined;
  }
  var {
    stackedData
  } = stackGroup;
  if (!stackedData) {
    return undefined;
  }
  return stackedData.find(sd => sd.key === stackSeriesIdentifier);
};