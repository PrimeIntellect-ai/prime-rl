import * as React from 'react';
import { RechartsRootState } from './store';
export declare const touchEventAction: import("@reduxjs/toolkit").ActionCreatorWithPayload<React.TouchEvent<HTMLDivElement>, string>;
export declare const touchEventMiddleware: import("@reduxjs/toolkit").ListenerMiddlewareInstance<RechartsRootState, import("@reduxjs/toolkit").ThunkDispatch<RechartsRootState, unknown, import("redux").AnyAction>, unknown>;
