import * as React from 'react';
import { EasingInput } from './easing';
import { AnimationManager } from './AnimationManager';
type JavascriptAnimateProps = {
    animationId: string;
    animationManager?: AnimationManager;
    duration?: number;
    begin?: number;
    easing?: EasingInput;
    isActive?: boolean | 'auto';
    canBegin?: boolean;
    onAnimationStart?: () => void;
    onAnimationEnd?: () => void;
    children: (time: number) => React.ReactNode;
};
export declare function JavascriptAnimate(outsideProps: JavascriptAnimateProps): React.ReactNode;
export {};
