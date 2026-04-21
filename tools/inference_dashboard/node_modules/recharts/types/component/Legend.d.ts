import * as React from 'react';
import { CSSProperties } from 'react';
import { ContentType, LegendPayload, Props as DefaultLegendContentProps, VerticalAlignmentType } from './DefaultLegendContent';
import { UniqueOption } from '../util/payload/getUniqPayload';
import { ElementOffset } from '../util/useElementOffset';
export type LegendItemSorter = 'value' | 'dataKey' | ((item: LegendPayload) => number | string);
export type Props = Omit<DefaultLegendContentProps, 'payload' | 'ref' | 'verticalAlign'> & {
    /**
     * Renders the content of the legend.
     *
     * This should return HTML elements, not SVG elements.
     *
     * - If not set, the {@link DefaultLegendContent} component is used.
     * - If set to a React element, this element will be cloned and extra props will be passed in.
     * - If set to a function, the function will be called and should return HTML elements.
     *
     * @example <Legend content={CustomizedLegend} />
     * @example <Legend content={renderLegend} />
     */
    content?: ContentType;
    /**
     * CSS styles to be applied to the wrapper `div` element.
     */
    wrapperStyle?: CSSProperties;
    /**
     * Width of the legend.
     * Accept CSS style string values like `100%` or `fit-content`, or number values like `400`.
     */
    width?: number | string;
    /**
     * Height of the legend.
     * Accept CSS style string values like `100%` or `fit-content`, or number values like `400`.
     */
    height?: number | string;
    payloadUniqBy?: UniqueOption<LegendPayload>;
    onBBoxUpdate?: (box: ElementOffset | null) => void;
    /**
     * If portal is defined, then Legend will use this element as a target
     * for rendering using React Portal.
     *
     * If this is undefined then Legend renders inside the recharts-wrapper element.
     *
     * @see {@link https://react.dev/reference/react-dom/createPortal}
     */
    portal?: HTMLElement | null;
    /**
     * Sorts Legend items. Defaults to `value` which means it will sort alphabetically
     * by the label.
     *
     * If `null` is provided then the payload is not sorted. Be aware that without sort,
     * the order of items may change between renders!
     *
     * @defaultValue value
     */
    itemSorter?: LegendItemSorter | null;
    /**
     * The alignment of the whole Legend container:
     *
     * - `bottom`: shows the Legend below chart, and chart height reduces automatically to make space for it.
     * - `top`: shows the Legend above chart, and chart height reduces automatically.
     * - `middle`:  shows the Legend in the middle of chart, covering other content, and chart height remains unchanged.
     * The exact behavior changes depending on `align` prop.
     *
     * @defaultValue bottom
     */
    verticalAlign?: VerticalAlignmentType;
};
export declare const legendDefaultProps: {
    readonly align: "center";
    readonly iconSize: 14;
    readonly inactiveColor: "#ccc";
    readonly itemSorter: "value";
    readonly layout: "horizontal";
    readonly verticalAlign: "bottom";
};
/**
 * @consumes CartesianChartContext
 * @consumes PolarChartContext
 */
declare function LegendImpl(outsideProps: Props): React.ReactPortal | null;
export declare const Legend: React.MemoExoticComponent<typeof LegendImpl>;
export {};
