import { createSlice } from '@reduxjs/toolkit';
import { castDraft } from 'immer';
export var initialEventSettingsState = {
  throttleDelay: 'raf',
  throttledEvents: ['mousemove', 'touchmove', 'pointermove', 'scroll', 'wheel']
};
var eventSettingsSlice = createSlice({
  name: 'eventSettings',
  initialState: initialEventSettingsState,
  reducers: {
    setEventSettings: (state, action) => {
      if (action.payload.throttleDelay != null) {
        state.throttleDelay = action.payload.throttleDelay;
      }
      if (action.payload.throttledEvents != null) {
        state.throttledEvents = castDraft(action.payload.throttledEvents);
      }
    }
  }
});
export var {
  setEventSettings
} = eventSettingsSlice.actions;
export var eventSettingsReducer = eventSettingsSlice.reducer;