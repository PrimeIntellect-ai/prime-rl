import { useCallback, useEffect, useRef, useState } from 'react';
import { noop } from '../util/DataUtils';
import { resolveDefaultProps } from '../util/resolveDefaultProps';
import { useAnimationManager } from './useAnimationManager';
import { getTransitionVal } from './util';
import { Global } from '../util/Global';
import { usePrefersReducedMotion } from '../util/usePrefersReducedMotion';
var defaultProps = {
  begin: 0,
  duration: 1000,
  easing: 'ease',
  isActive: true,
  canBegin: true,
  onAnimationEnd: () => {},
  onAnimationStart: () => {}
};
export function CSSTransitionAnimate(outsideProps) {
  var props = resolveDefaultProps(outsideProps, defaultProps);
  var {
    animationId,
    from,
    to,
    attributeName,
    isActive: isActiveProp,
    canBegin,
    duration,
    easing,
    begin,
    onAnimationEnd,
    onAnimationStart: onAnimationStartFromProps,
    children
  } = props;
  var prefersReducedMotion = usePrefersReducedMotion();
  var isActive = isActiveProp === 'auto' ? !Global.isSsr && !prefersReducedMotion : isActiveProp;
  var animationManager = useAnimationManager(animationId + attributeName, props.animationManager);
  var [style, setStyle] = useState(() => {
    if (!isActive) {
      return to;
    }
    return from;
  });
  var initialized = useRef(false);
  var onAnimationStart = useCallback(() => {
    setStyle(from);
    onAnimationStartFromProps();
  }, [from, onAnimationStartFromProps]);
  useEffect(() => {
    if (!isActive || !canBegin) {
      return noop;
    }
    initialized.current = true;
    var unsubscribe = animationManager.subscribe(setStyle);
    animationManager.start([onAnimationStart, begin, to, duration, onAnimationEnd]);
    return () => {
      animationManager.stop();
      if (unsubscribe) {
        unsubscribe();
      }
      onAnimationEnd();
    };
  }, [isActive, canBegin, duration, easing, begin, onAnimationStart, onAnimationEnd, animationManager, to, from]);
  if (!isActive) {
    /*
     * With isActive=false, the component always renders with the final style, immediately,
     * and ignores all other props.
     * Also there is no transition applied.
     */
    return children({
      [attributeName]: to
    });
  }
  if (!canBegin) {
    return children({
      [attributeName]: from
    });
  }
  if (initialized.current) {
    var transition = getTransitionVal([attributeName], duration, easing);
    return children({
      transition,
      [attributeName]: style
    });
  }
  return children({
    [attributeName]: from
  });
}