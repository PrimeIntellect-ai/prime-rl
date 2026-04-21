var _ref;
import * as React from 'react';
import { uniqueId } from './DataUtils';

/**
 * Fallback for React.useId() for versions prior to React 18.
 * Generates a unique ID using a simple counter and a prefix.
 *
 * @returns A unique ID that remains consistent across renders.
 */
export var useIdFallback = () => {
  var [id] = React.useState(() => uniqueId('uid-'));
  return id;
};

/*
 * This weird syntax is used to avoid a build-time error in React 17 and earlier when building with Webpack.
 * See https://github.com/webpack/webpack/issues/14814
 */
export var useId = (_ref = React['useId'.toString()]) !== null && _ref !== void 0 ? _ref : useIdFallback;