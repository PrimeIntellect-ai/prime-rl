import * as React from 'react';
import { ActiveShape, SymbolType } from './types';
import { ScatterPointItem } from '../cartesian/Scatter';
import { DATA_ITEM_GRAPHICAL_ITEM_ID_ATTRIBUTE_NAME } from './Constants';
import { GraphicalItemId } from '../state/graphicalItemsSlice';
export type ScatterShapeProps = ScatterPointItem & {
    index: number;
    [DATA_ITEM_GRAPHICAL_ITEM_ID_ATTRIBUTE_NAME]: GraphicalItemId;
};
export declare function ScatterSymbol({ option, isActive, ...props }: {
    option: ActiveShape<ScatterShapeProps> | SymbolType;
    isActive: boolean;
} & ScatterShapeProps): React.JSX.Element;
