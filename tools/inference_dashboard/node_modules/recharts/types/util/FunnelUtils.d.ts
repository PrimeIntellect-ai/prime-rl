import * as React from 'react';
import { Props as FunnelProps, FunnelTrapezoidItem } from '../cartesian/Funnel';
export type FunnelTrapezoidProps = {
    option: FunnelProps['activeShape'];
    isActive: boolean;
} & FunnelTrapezoidItem;
export declare function FunnelTrapezoid(props: FunnelTrapezoidProps): React.JSX.Element;
