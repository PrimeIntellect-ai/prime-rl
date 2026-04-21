import { CartesianViewBoxRequired, TrapezoidViewBox } from '../util/types';
import { TextAnchor, TextVerticalAnchor } from '../component/Text';
export type CartesianLabelPosition = 'top' | 'left' | 'right' | 'bottom' | 'inside' | 'outside' | 'insideLeft' | 'insideRight' | 'insideTop' | 'insideBottom' | 'insideTopLeft' | 'insideBottomLeft' | 'insideTopRight' | 'insideBottomRight' | 'insideStart' | 'insideEnd' | 'end' | 'center' | 'centerTop' | 'centerBottom' | 'middle' | {
    x?: number | string;
    y?: number | string;
};
export type GetCartesianPositionOptions = {
    viewBox: TrapezoidViewBox | CartesianViewBoxRequired;
    parentViewBox?: CartesianViewBoxRequired;
    /**
     * The offset to the specified "position". Direction of the offset depends on the position.
     */
    offset?: number;
    /**
     * The position of the element relative to the view box.
     */
    position?: CartesianLabelPosition;
    /**
     * If true, the returned width and height will be clamped to keep the element within the parentViewBox.
     * This is useful for preventing labels from overflowing the chart area.
     */
    clamp?: boolean;
};
export type CartesianPosition = {
    x: number;
    y: number;
    horizontalAnchor: TextAnchor;
    verticalAnchor: TextVerticalAnchor;
    width?: number;
    height?: number;
};
/**
 * Calculates the position and alignment for a generic element in a Cartesian coordinate system.
 *
 * @param options - The options including viewBox, position, and offset.
 * @returns The calculated x, y, alignment and size.
 */
export declare const getCartesianPosition: (options: GetCartesianPositionOptions) => CartesianPosition;
