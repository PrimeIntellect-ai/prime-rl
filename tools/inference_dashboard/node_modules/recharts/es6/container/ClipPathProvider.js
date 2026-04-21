import * as React from 'react';
import { createContext, useContext, useState } from 'react';
import { uniqueId } from '../util/DataUtils';
import { usePlotArea } from '../hooks';
var ClipPathIdContext = /*#__PURE__*/createContext(undefined);

/**
 * Generates a unique clip path ID for use in SVG elements,
 * and puts it in a context provider.
 *
 * To read the clip path ID, use the `useClipPathId` hook,
 * or render `<ClipPath>` component which will automatically use the ID from this context.
 *
 * @param props children - React children to be wrapped by the provider
 * @returns React Context Provider
 */
export var ClipPathProvider = _ref => {
  var {
    children
  } = _ref;
  var [clipPathId] = useState("".concat(uniqueId('recharts'), "-clip"));
  var plotArea = usePlotArea();
  if (plotArea == null) {
    return null;
  }
  var {
    x,
    y,
    width,
    height
  } = plotArea;
  return /*#__PURE__*/React.createElement(ClipPathIdContext.Provider, {
    value: clipPathId
  }, /*#__PURE__*/React.createElement("defs", null, /*#__PURE__*/React.createElement("clipPath", {
    id: clipPathId
  }, /*#__PURE__*/React.createElement("rect", {
    x: x,
    y: y,
    height: height,
    width: width
  }))), children);
};
export var useClipPathId = () => {
  return useContext(ClipPathIdContext);
};