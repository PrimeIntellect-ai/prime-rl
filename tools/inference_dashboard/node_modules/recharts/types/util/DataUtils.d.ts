import { NonEmptyArray, Percent } from './types';
export declare const mathSign: (value: number) => 0 | 1 | -1;
export declare const isNan: (value: unknown) => value is number;
export declare const isPercent: (value: string | number | undefined) => value is Percent;
export declare const isNumber: (value: unknown) => value is number;
export declare const isNumOrStr: (value: unknown) => value is number | string;
export declare const uniqueId: (prefix?: string) => string;
/**
 * Calculates the numeric value represented by a percent string or number, based on a total value.
 *
 * - If `percent` is not a number or string, returns `defaultValue`.
 * - If `percent` is a percent string but `totalValue` is null/undefined, returns `defaultValue`.
 * - If the result is NaN, returns `defaultValue`.
 * - If `validate` is true and the result exceeds `totalValue`, returns `totalValue`.
 *
 * @param percent - The percent value to convert. Can be a number (e.g. 25) or a string ending with '%' (e.g. '25%').
 *                  If a string, it must end with '%' to be treated as a percent; otherwise, it is parsed as a number.
 * @param totalValue - The total value to calculate the percent of. Required if `percent` is a percent string.
 * @param defaultValue - The value returned if `percent` is undefined, invalid, or cannot be converted to a number.
 * @param validate - If true, ensures the result does not exceed `totalValue` (when provided).
 * @returns The calculated value, or `defaultValue` for invalid input.
 */
export declare const getPercentValue: (percent: number | string | undefined, totalValue: number | undefined, defaultValue?: number, validate?: boolean) => number;
export declare const hasDuplicate: (ary: ReadonlyArray<unknown>) => boolean;
export declare function interpolate(start: unknown, end: number, t: number): number;
export declare function interpolate(start: unknown, end: null, t: number): null;
export declare function interpolate(start: unknown, end: number | null, t: number): number | null;
export declare function findEntryInArray<T>(ary: ReadonlyArray<T>, specifiedKey: number | string | ((entry: T) => unknown), specifiedValue: unknown): T | undefined;
type LinearRegressionResult = {
    xmin: number;
    xmax: number;
    a: number;
    b: number;
};
/**
 * The least square linear regression
 * @param {Array} data The array of points
 * @returns {Object} The domain of x, and the parameter of linear function
 */
export declare const getLinearRegression: (data: NonEmptyArray<{
    cx?: number;
    cy?: number;
}>) => LinearRegressionResult;
type Nullish = null | undefined;
/**
 * Checks if the value is null or undefined
 * @param value The value to check
 * @returns true if the value is null or undefined
 */
export declare const isNullish: (value: unknown) => value is Nullish;
/**
 * Uppercase the first letter of a string
 * @param {string} value The string to uppercase
 * @returns {string} The uppercased string
 */
export declare const upperFirst: (value: string) => string;
/**
 * Checks if the value is not null nor undefined.
 * @param value The value to check
 * @returns true if the value is not null nor undefined
 */
export declare function isNotNil<T>(value: T): value is NonNullable<T>;
/**
 * No-operation function that does nothing.
 * Useful as a placeholder or default callback function.
 */
export declare function noop(): undefined;
export {};
