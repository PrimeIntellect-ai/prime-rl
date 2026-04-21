import { isStacked } from '../../types/StackedGraphicalItem';
import { getPercentValue, isNullish } from '../../../util/DataUtils';
var getBarSize = (globalSize, totalSize, selfSize) => {
  var barSize = selfSize !== null && selfSize !== void 0 ? selfSize : globalSize;
  if (isNullish(barSize)) {
    return undefined;
  }
  return getPercentValue(barSize, totalSize, 0);
};
export var combineBarSizeList = (allBars, globalSize, totalSize) => {
  var initialValue = {};
  var stackedBars = allBars.filter(isStacked);
  var unstackedBars = allBars.filter(b => b.stackId == null);
  var groupByStack = stackedBars.reduce((acc, bar) => {
    var s = acc[bar.stackId];
    if (s == null) {
      s = [];
    }
    s.push(bar);
    acc[bar.stackId] = s;
    return acc;
  }, initialValue);
  var stackedSizeList = Object.entries(groupByStack).map(_ref => {
    var _bars$;
    var [stackId, bars] = _ref;
    var dataKeys = bars.map(b => b.dataKey);
    var barSize = getBarSize(globalSize, totalSize, (_bars$ = bars[0]) === null || _bars$ === void 0 ? void 0 : _bars$.barSize);
    return {
      stackId,
      dataKeys,
      barSize
    };
  });
  var unstackedSizeList = unstackedBars.map(b => {
    var dataKeys = [b.dataKey].filter(dk => dk != null);
    var barSize = getBarSize(globalSize, totalSize, b.barSize);
    return {
      stackId: undefined,
      dataKeys,
      barSize
    };
  });
  return [...stackedSizeList, ...unstackedSizeList];
};