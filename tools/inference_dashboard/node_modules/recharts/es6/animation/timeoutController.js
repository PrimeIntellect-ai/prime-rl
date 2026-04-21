/**
 * Callback type for the timeout function.
 * Receives current time in milliseconds as an argument.
 */

/**
 * A function that, when called, cancels the timeout.
 */

export class RequestAnimationFrameTimeoutController {
  setTimeout(callback) {
    var delay = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : 0;
    var startTime = performance.now();
    var requestId = null;
    var executeCallback = now => {
      if (now - startTime >= delay) {
        callback(now);
        // tests fail without the extra if, even when five lines below it's not needed
        // TODO finish transition to the mocked timeout controller and then remove this condition
      } else if (typeof requestAnimationFrame === 'function') {
        requestId = requestAnimationFrame(executeCallback);
      }
    };
    requestId = requestAnimationFrame(executeCallback);
    return () => {
      if (requestId != null) {
        cancelAnimationFrame(requestId);
      }
    };
  }
}