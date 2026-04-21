import * as React from 'react';
import { ReactNode, MouseEvent, ReactElement, CSSProperties } from 'react';
import { DataKey, LegendType, PresentationAttributesAdaptChildEvent, CartesianLayout } from '../util/types';
export type ContentType = ReactElement | ((props: Props) => ReactNode);
export type HorizontalAlignmentType = 'center' | 'left' | 'right';
export type VerticalAlignmentType = 'top' | 'bottom' | 'middle';
export type Formatter = (value: any, entry: LegendPayload, index: number) => ReactNode;
export interface LegendPayload {
    /**
     * This is the text that will be displayed in the legend in the DOM.
     * If undefined, the text will not be displayed, so the icon will be rendered without text.
     */
    value: string | undefined;
    type?: LegendType;
    color?: string;
    /**
     * Different graphical items put different information in the payload object
     * so double check in runtime what are you getting here.
     */
    payload?: object;
    formatter?: Formatter;
    inactive?: boolean;
    legendIcon?: ReactElement<SVGElement>;
    dataKey?: DataKey<any>;
}
interface DefaultLegendContentProps {
    /**
     * The size of icon in each legend item.
     * @defaultValue 14
     */
    iconSize?: number;
    /**
     * The type of icon in each legend item.
     */
    iconType?: LegendType;
    /**
     * The layout of legend items inside the legend container.
     * @defaultValue horizontal
     */
    layout?: CartesianLayout;
    /**
     * Horizontal alignment of the whole Legend container:
     *
     * - `left`: shows the Legend to the left of the chart, and chart width reduces automatically to make space for it.
     * - `right` shows the Legend to the right of the chart, and chart width reduces automatically.
     * - `center` shows the Legend in the middle of chart, and chart width remains unchanged.
     *
     * The exact behavior changes depending on 'verticalAlign' prop.
     *
     * @defaultValue center
     */
    align?: HorizontalAlignmentType;
    /**
     * Vertical alignment of the whole Legend container:
     *
     * - `bottom`: shows the Legend below chart, and chart height reduces automatically to make space for it.
     * - `top`: shows the Legend above chart, and chart height reduces automatically.
     * - `middle`:  shows the Legend in the middle of chart, covering other content, and chart height remains unchanged.
     * The exact behavior changes depending on `align` prop.
     *
     * @defaultValue middle
     */
    verticalAlign?: VerticalAlignmentType;
    /**
     * The color of the icon when the item is inactive.
     * @defaultValue #ccc
     */
    inactiveColor?: string;
    /**
     * Function to customize how content is serialized before rendering.
     *
     * This should return HTML elements, or strings.
     *
     * @example (value, entry, index) => <span style={{ color: 'red' }}>{value}</span>
     * @example https://codesandbox.io/s/legend-formatter-rmzp9
     */
    formatter?: Formatter;
    /**
     * The customized event handler of mouseenter on the items in this group
     * @example https://recharts.github.io/examples/LegendEffectOpacity
     */
    onMouseEnter?: (data: LegendPayload, index: number, event: MouseEvent<HTMLElement>) => void;
    /**
     * The customized event handler of mouseleave on the items in this group
     * @example https://recharts.github.io/examples/LegendEffectOpacity
     */
    onMouseLeave?: (data: LegendPayload, index: number, event: MouseEvent<HTMLElement>) => void;
    /**
     * The customized event handler of click on the items in this group
     */
    onClick?: (data: LegendPayload, index: number, event: MouseEvent<HTMLElement>) => void;
    /**
     * DefaultLegendContent.payload is omitted from Legend props.
     * A custom payload can be passed here if desired, or it can be passed from the Legend "content" callback.
     */
    payload?: ReadonlyArray<LegendPayload>;
    /**
     * The style of each text label which is a span element.
     * @defaultValue {}
     */
    labelStyle?: CSSProperties;
}
export type Props = DefaultLegendContentProps & Omit<PresentationAttributesAdaptChildEvent<any, ReactElement>, keyof DefaultLegendContentProps>;
export declare const defaultLegendContentDefaultProps: {
    readonly align: "center";
    readonly iconSize: 14;
    readonly inactiveColor: "#ccc";
    readonly layout: "horizontal";
    readonly verticalAlign: "middle";
    readonly labelStyle: {};
};
/**
 * This component is by default rendered inside the {@link Legend} component. You would not use it directly.
 *
 * You can use this component to customize the content of the legend,
 * or you can provide your own completely independent content.
 */
export declare const DefaultLegendContent: (outsideProps: Props) => React.JSX.Element | null;
export {};
