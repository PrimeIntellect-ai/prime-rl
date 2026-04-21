import { createAnimateManager } from './AnimationManager';
import { RequestAnimationFrameTimeoutController } from './timeoutController';
export function createDefaultAnimationManager() {
  return createAnimateManager(new RequestAnimationFrameTimeoutController());
}