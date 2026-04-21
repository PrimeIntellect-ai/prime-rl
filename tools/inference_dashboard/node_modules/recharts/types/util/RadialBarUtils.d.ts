import * as React from 'react';
import { RadialBarDataItem, RadialBarProps } from '../polar/RadialBar';
export declare function parseCornerRadius(cornerRadius: string | number | undefined): number | undefined;
export interface RadialBarSectorProps extends RadialBarDataItem {
    index: number;
    option: RadialBarProps['activeShape'];
    isActive: boolean;
}
export declare function RadialBarSector(props: RadialBarSectorProps): React.JSX.Element;
