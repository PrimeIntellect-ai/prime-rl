import * as React from 'react';
import { ReactElement } from 'react';
/**
 * This is an abstraction for rendering a user defined prop for a customized shape in several forms.
 *
 * <Shape /> is the root and will handle taking in:
 *  - an object of svg properties
 *  - a boolean
 *  - a render prop(inline function that returns jsx)
 *  - a React element
 *
 * <ShapeSelector /> is a subcomponent of <Shape /> and used to match a component
 * to the value of props.shapeType that is passed to the root.
 *
 */
type ShapeType = 'trapezoid' | 'rectangle' | 'sector' | 'symbols' | 'curve';
export type ShapeProps<OptionType, ExtraProps> = {
    shapeType: ShapeType;
    option: OptionType;
    isActive?: boolean;
    index?: string | number;
    activeClassName?: string;
    inActiveClassName?: string;
} & ExtraProps;
export declare function getPropsFromShapeOption<T>(option: ReactElement<T> | T): T;
export declare function Shape<OptionType, ExtraProps, ShapePropsType extends React.JSX.IntrinsicAttributes>({ option, shapeType, activeClassName, inActiveClassName, ...props }: ShapeProps<OptionType, ExtraProps>): React.JSX.Element;
export {};
