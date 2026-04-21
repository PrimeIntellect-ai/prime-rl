import { createContext, useContext, useMemo } from 'react';
import { createDefaultAnimationManager } from './createDefaultAnimationManager';
export var AnimationManagerContext = /*#__PURE__*/createContext(createDefaultAnimationManager);
export function useAnimationManager(animationId, animationManagerFromProps) {
  var contextAnimationManager = useContext(AnimationManagerContext);
  return useMemo(() => animationManagerFromProps !== null && animationManagerFromProps !== void 0 ? animationManagerFromProps : contextAnimationManager(animationId), [animationId, animationManagerFromProps, contextAnimationManager]);
}