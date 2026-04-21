import { useEffect, useState } from 'react';
import { useAppDispatch, useAppSelector } from '../state/hooks';
import { selectContainerScale } from '../state/selectors/containerSelectors';
import { setScale } from '../state/layoutSlice';
import { isWellBehavedNumber } from './isWellBehavedNumber';
export function useReportScale() {
  var dispatch = useAppDispatch();
  var [ref, setRef] = useState(null);
  var scale = useAppSelector(selectContainerScale);
  useEffect(() => {
    if (ref == null) {
      return;
    }
    var rect = ref.getBoundingClientRect();
    var newScale = rect.width / ref.offsetWidth;
    if (isWellBehavedNumber(newScale) && newScale !== scale) {
      dispatch(setScale(newScale));
    }
  }, [ref, dispatch, scale]);
  return setRef;
}