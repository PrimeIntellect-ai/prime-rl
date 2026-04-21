import { ChartOffsetInternal } from '../../util/types';
import { RechartsRootState } from '../store';
export declare const selectBrushHeight: (state: RechartsRootState) => number;
/**
 * For internal use only.
 *
 * @param root state
 * @return ChartOffsetInternal
 */
export declare const selectChartOffsetInternal: (state: RechartsRootState) => ChartOffsetInternal;
export declare const selectChartViewBox: ((state: RechartsRootState) => Required<import("../../util/types").CartesianViewBox>) & {
    clearCache: () => void;
    resultsCount: () => number;
    resetResultsCount: () => void;
} & {
    resultFunc: (resultFuncArgs_0: ChartOffsetInternal) => Required<import("../../util/types").CartesianViewBox>;
    memoizedResultFunc: ((resultFuncArgs_0: ChartOffsetInternal) => Required<import("../../util/types").CartesianViewBox>) & {
        clearCache: () => void;
        resultsCount: () => number;
        resetResultsCount: () => void;
    };
    lastResult: () => Required<import("../../util/types").CartesianViewBox>;
    dependencies: [(state: RechartsRootState) => ChartOffsetInternal];
    recomputations: () => number;
    resetRecomputations: () => void;
    dependencyRecomputations: () => number;
    resetDependencyRecomputations: () => void;
} & {
    argsMemoize: typeof import("reselect").weakMapMemoize;
    memoize: typeof import("reselect").weakMapMemoize;
};
export declare const selectAxisViewBox: ((state: RechartsRootState) => Required<import("../../util/types").CartesianViewBox>) & {
    clearCache: () => void;
    resultsCount: () => number;
    resetResultsCount: () => void;
} & {
    resultFunc: (resultFuncArgs_0: number, resultFuncArgs_1: number) => Required<import("../../util/types").CartesianViewBox>;
    memoizedResultFunc: ((resultFuncArgs_0: number, resultFuncArgs_1: number) => Required<import("../../util/types").CartesianViewBox>) & {
        clearCache: () => void;
        resultsCount: () => number;
        resetResultsCount: () => void;
    };
    lastResult: () => Required<import("../../util/types").CartesianViewBox>;
    dependencies: [(state: RechartsRootState) => number, (state: RechartsRootState) => number];
    recomputations: () => number;
    resetRecomputations: () => void;
    dependencyRecomputations: () => number;
    resetDependencyRecomputations: () => void;
} & {
    argsMemoize: typeof import("reselect").weakMapMemoize;
    memoize: typeof import("reselect").weakMapMemoize;
};
