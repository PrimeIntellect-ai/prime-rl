import { AnimationManager } from './AnimationManager';
export type AnimationManagerFactory = (animationId: string) => AnimationManager;
export declare const AnimationManagerContext: import("react").Context<AnimationManagerFactory>;
export declare function useAnimationManager(animationId: string, animationManagerFromProps: AnimationManager | undefined): AnimationManager;
