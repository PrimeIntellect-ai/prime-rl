import { useEffect, useRef, useState } from 'react';
import { noop } from '../util/DataUtils';
import { resolveDefaultProps } from '../util/resolveDefaultProps';
import configUpdate from './configUpdate';
import { configEasing } from './easing';
import { useAnimationManager } from './useAnimationManager';
import { Global } from '../util/Global';
import { usePrefersReducedMotion } from '../util/usePrefersReducedMotion';
var defaultJavascriptAnimateProps = {
  begin: 0,
  duration: 1000,
  easing: 'ease',
  isActive: true,
  canBegin: true,
  onAnimationEnd: () => {},
  onAnimationStart: () => {}
};
var from = {
  t: 0
};
var to = {
  t: 1
};
export function JavascriptAnimate(outsideProps) {
  var props = resolveDefaultProps(outsideProps, defaultJavascriptAnimateProps);
  var {
    isActive: isActiveProp,
    canBegin,
    duration,
    easing,
    begin,
    onAnimationEnd,
    onAnimationStart,
    children
  } = props;
  var prefersReducedMotion = usePrefersReducedMotion();
  var isActive = isActiveProp === 'auto' ? !Global.isSsr && !prefersReducedMotion : isActiveProp;
  var animationManager = useAnimationManager(props.animationId, props.animationManager);
  var [style, setStyle] = useState(isActive ? from : to);
  var stopJSAnimation = useRef(null);
  useEffect(() => {
    if (!isActive) {
      setStyle(to);
    }
  }, [isActive]);
  useEffect(() => {
    if (!isActive || !canBegin) {
      return noop;
    }
    var startAnimation = configUpdate(from, to, configEasing(easing), duration, setStyle, animationManager.getTimeoutController());
    var onAnimationActive = () => {
      stopJSAnimation.current = startAnimation();
    };
    animationManager.start([onAnimationStart, begin, onAnimationActive, duration, onAnimationEnd]);
    return () => {
      animationManager.stop();
      if (stopJSAnimation.current) {
        stopJSAnimation.current();
      }
      onAnimationEnd();
    };
  }, [isActive, canBegin, duration, easing, begin, onAnimationStart, onAnimationEnd, animationManager]);
  return children(style.t);
}