import * as React from 'react';
import { ReactElement } from 'react';
import { Series } from 'victory-vendor/d3-shape';
import { RadialBarSectorProps } from '../util/RadialBarUtils';
import { Props as SectorProps } from '../shape/Sector';
import { ImplicitLabelListType } from '../component/LabelList';
import { BarPositionPosition } from '../util/ChartUtils';
import { ActiveShape, AnimationDuration, AnimationTiming, DataConsumer, DataKey, LayoutType, LegendType, PolarViewBoxRequired, PresentationAttributesAdaptChildEvent, TickItem, TooltipType } from '../util/types';
import { TooltipTriggerInfo } from '../context/tooltipContext';
import { BaseAxisWithScale } from '../state/selectors/axisSelectors';
import { ChartData } from '../state/chartDataSlice';
import { AxisId } from '../state/cartesianAxisSlice';
import { ZIndexable } from '../zIndex/ZIndexLayer';
export type RadialBarDataItem = SectorProps & PolarViewBoxRequired & TooltipTriggerInfo & {
    value?: any;
    payload?: any;
    background?: SectorProps;
};
type RadialBarBackground = boolean | (ActiveShape<SectorProps> & ZIndexable);
interface InternalRadialBarProps<DataPointType = any, DataValueType = any> extends DataConsumer<DataPointType, DataValueType>, ZIndexable {
    activeShape?: ActiveShape<RadialBarSectorProps, SVGPathElement>;
    /**
     * @defaultValue 0
     */
    angleAxisId?: AxisId;
    /**
     * Specifies when the animation should begin, the unit of this option is ms.
     * @defaultValue 0
     */
    animationBegin?: number;
    /**
     * Specifies the duration of animation, the unit of this option is ms.
     * @defaultValue 1500
     */
    animationDuration?: AnimationDuration;
    /**
     * The type of easing function.
     * @defaultValue ease
     */
    animationEasing?: AnimationTiming;
    /**
     * Renders a background for each bar. Options:
     *  - `false`: no background;
     *  - `true`: renders default background;
     *  - `object`: the props of background rectangle;
     *  - `ReactElement`: a custom background element;
     *  - `function`: a render function of custom background.
     *
     * @defaultValue false
     */
    background?: RadialBarBackground;
    /**
     * The width or height of each bar. If the barSize is not specified, the size of the bar will be calculated by the barCategoryGap, barGap and the quantity of bar groups.
     */
    barSize?: number;
    className?: string;
    /**
     * @defaultValue false
     */
    cornerIsExternal?: boolean;
    /**
     * @defaultValue 0
     */
    cornerRadius?: string | number;
    /**
     * Calculated radial bar sectors
     */
    sectors: ReadonlyArray<RadialBarDataItem>;
    /**
     * @defaultValue false
     */
    forceCornerRadius?: boolean;
    /**
     * @defaultValue false
     */
    hide?: boolean;
    /**
     * If set false, animation of radial bars will be disabled.
     * If set "auto", the animation will be disabled in SSR and will respect the user's prefers-reduced-motion system preference for accessibility.
     * @defaultValue auto
     */
    isAnimationActive?: boolean | 'auto';
    /**
     * Renders one label for each data point. Options:
     * - `true`: renders default labels;
     * - `false`: no labels are rendered;
     * - `object`: the props of LabelList component;
     * - `ReactElement`: a custom label element;
     * - `function`: a render function of custom label.
     *
     * @defaultValue false
     */
    label?: ImplicitLabelListType;
    /**
     * The type of icon in legend.  If set to 'none', no legend item will be rendered.
     * @defaultValue rect
     */
    legendType?: LegendType;
    maxBarSize?: number;
    /**
     * @defaultValue 0
     */
    minPointSize?: number;
    /**
     * The customized event handler of animation end
     */
    onAnimationEnd?: () => void;
    /**
     * The customized event handler of animation start
     */
    onAnimationStart?: () => void;
    /**
     * The customized event handler of click in this chart.
     */
    onClick?: (data: RadialBarDataItem, index: number, e: React.MouseEvent<SVGGraphicsElement>) => void;
    /**
     * The customized event handler of mousedown in this chart.
     */
    onMouseDown?: (data: RadialBarDataItem, index: number, e: React.MouseEvent<SVGGraphicsElement>) => void;
    /**
     * The customized event handler of mouseup in this chart.
     */
    onMouseUp?: (data: RadialBarDataItem, index: number, e: React.MouseEvent<SVGGraphicsElement>) => void;
    /**
     * The customized event handler of mousemove in this chart.
     */
    onMouseMove?: (data: RadialBarDataItem, index: number, e: React.MouseEvent<SVGGraphicsElement>) => void;
    /**
     * The customized event handler of mouseover in this chart.
     */
    onMouseOver?: (data: RadialBarDataItem, index: number, e: React.MouseEvent<SVGGraphicsElement>) => void;
    /**
     * The customized event handler of mouseout in this chart.
     */
    onMouseOut?: (data: RadialBarDataItem, index: number, e: React.MouseEvent<SVGGraphicsElement>) => void;
    /**
     * The customized event handler of mouseenter in this chart.
     */
    onMouseEnter?: (data: RadialBarDataItem, index: number, e: React.MouseEvent<SVGGraphicsElement>) => void;
    /**
     * The customized event handler of mouseleave in this chart.
     */
    onMouseLeave?: (data: RadialBarDataItem, index: number, e: React.MouseEvent<SVGGraphicsElement>) => void;
    onTouchStart?: (data: RadialBarDataItem, index: number, e: React.TouchEvent<SVGGraphicsElement>) => void;
    onTouchMove?: (data: RadialBarDataItem, index: number, e: React.TouchEvent<SVGGraphicsElement>) => void;
    onTouchEnd?: (data: RadialBarDataItem, index: number, e: React.TouchEvent<SVGGraphicsElement>) => void;
    /**
     * @defaultValue 0
     */
    radiusAxisId?: AxisId;
    shape?: ActiveShape<RadialBarSectorProps, SVGPathElement>;
    stackId?: string | number;
    tooltipType?: TooltipType;
    /**
     * @defaultValue 300
     */
    zIndex?: number;
}
export type RadialBarProps<DataPointType = any, DataValueType = any> = Omit<PresentationAttributesAdaptChildEvent<RadialBarDataItem, SVGElement>, 'ref' | keyof InternalRadialBarProps<DataPointType, DataValueType>> & Omit<InternalRadialBarProps<DataPointType, DataValueType>, 'sectors'>;
export declare const defaultRadialBarProps: {
    readonly angleAxisId: 0;
    readonly animationBegin: 0;
    readonly animationDuration: 1500;
    readonly animationEasing: "ease";
    readonly background: false;
    readonly cornerIsExternal: false;
    readonly cornerRadius: 0;
    readonly forceCornerRadius: false;
    readonly hide: false;
    readonly isAnimationActive: "auto";
    readonly label: false;
    readonly legendType: "rect";
    readonly minPointSize: 0;
    readonly radiusAxisId: 0;
    readonly zIndex: 300;
};
export declare function computeRadialBarDataItems({ displayedData, stackedData, dataStartIndex, stackedDomain, dataKey, baseValue, layout, radiusAxis, radiusAxisTicks, bandSize, pos, angleAxis, minPointSize, cx, cy, angleAxisTicks, cells, startAngle: rootStartAngle, endAngle: rootEndAngle, }: {
    displayedData: ChartData;
    stackedData: Series<Record<number, number>, DataKey<any>> | undefined;
    dataStartIndex: number;
    stackedDomain: ReadonlyArray<unknown> | null;
    dataKey: DataKey<any> | undefined;
    baseValue: number | unknown;
    layout: LayoutType;
    radiusAxis: BaseAxisWithScale;
    radiusAxisTicks: ReadonlyArray<TickItem> | undefined;
    bandSize: number;
    pos: BarPositionPosition;
    angleAxis: BaseAxisWithScale;
    minPointSize: number;
    cx: number;
    cy: number;
    angleAxisTicks: ReadonlyArray<TickItem> | undefined;
    cells: ReadonlyArray<ReactElement> | undefined;
    startAngle: number;
    endAngle: number;
}): ReadonlyArray<RadialBarDataItem>;
/**
 * @consumes PolarChartContext
 * @provides LabelListContext
 * @provides CellReader
 */
export declare function RadialBar<DataPointType = any, DataValueType = any>(outsideProps: RadialBarProps<DataPointType, DataValueType>): React.JSX.Element;
export declare namespace RadialBar {
    var displayName: string;
}
export {};
