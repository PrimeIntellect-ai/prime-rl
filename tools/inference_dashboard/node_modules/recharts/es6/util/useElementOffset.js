import { useCallback, useState } from 'react';
var EPS = 1;

/**
 * TODO this documentation does not reflect what this hook is doing, update it.
 * Stores the `offsetHeight`, `offsetLeft`, `offsetTop`, and `offsetWidth` of a DOM element.
 */

/**
 * Use this to listen to element layout changes.
 *
 * Very useful for reading actual sizes of DOM elements relative to the viewport.
 *
 * @param extraDependencies use this to trigger new DOM dimensions read when any of these change. Good for things like payload and label, that will re-render something down in the children array, but you want to read the layout box of a parent.
 * @returns [lastElementOffset, updateElementOffset] most recent value, and setter. Pass the setter to a DOM element ref like this: `<div ref={updateElementOffset}>`
 */
export function useElementOffset() {
  var extraDependencies = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : [];
  var [lastBoundingBox, setLastBoundingBox] = useState({
    height: 0,
    left: 0,
    top: 0,
    width: 0
  });
  var updateBoundingBox = useCallback(node => {
    if (node != null) {
      var rect = node.getBoundingClientRect();
      var box = {
        height: rect.height,
        left: rect.left,
        top: rect.top,
        width: rect.width
      };
      if (Math.abs(box.height - lastBoundingBox.height) > EPS || Math.abs(box.left - lastBoundingBox.left) > EPS || Math.abs(box.top - lastBoundingBox.top) > EPS || Math.abs(box.width - lastBoundingBox.width) > EPS) {
        setLastBoundingBox({
          height: box.height,
          left: box.left,
          top: box.top,
          width: box.width
        });
      }
    }
  },
  // eslint-disable-next-line react-hooks/exhaustive-deps
  [lastBoundingBox.width, lastBoundingBox.height, lastBoundingBox.top, lastBoundingBox.left, ...extraDependencies]);
  return [lastBoundingBox, updateBoundingBox];
}