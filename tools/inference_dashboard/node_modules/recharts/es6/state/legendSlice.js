import { createSlice, current, prepareAutoBatched } from '@reduxjs/toolkit';
import { castDraft } from 'immer';

/**
 * The properties inside this state update independently of each other and quite often.
 * When selecting, never select the whole state because you are going to get
 * unnecessary re-renders. Select only the properties you need.
 *
 * This is why this state type is not exported - don't use it directly.
 */

var initialState = {
  settings: {
    layout: 'horizontal',
    align: 'center',
    verticalAlign: 'middle',
    itemSorter: 'value'
  },
  size: {
    width: 0,
    height: 0
  },
  payload: []
};
var legendSlice = createSlice({
  name: 'legend',
  initialState,
  reducers: {
    setLegendSize(state, action) {
      state.size.width = action.payload.width;
      state.size.height = action.payload.height;
    },
    setLegendSettings(state, action) {
      state.settings.align = action.payload.align;
      state.settings.layout = action.payload.layout;
      state.settings.verticalAlign = action.payload.verticalAlign;
      state.settings.itemSorter = action.payload.itemSorter;
    },
    addLegendPayload: {
      reducer(state, action) {
        state.payload.push(castDraft(action.payload));
      },
      prepare: prepareAutoBatched()
    },
    replaceLegendPayload: {
      reducer(state, action) {
        var {
          prev,
          next
        } = action.payload;
        var index = current(state).payload.indexOf(castDraft(prev));
        if (index > -1) {
          state.payload[index] = castDraft(next);
        }
      },
      prepare: prepareAutoBatched()
    },
    removeLegendPayload: {
      reducer(state, action) {
        var index = current(state).payload.indexOf(castDraft(action.payload));
        if (index > -1) {
          state.payload.splice(index, 1);
        }
      },
      prepare: prepareAutoBatched()
    }
  }
});
export var {
  setLegendSize,
  setLegendSettings,
  addLegendPayload,
  replaceLegendPayload,
  removeLegendPayload
} = legendSlice.actions;
export var legendReducer = legendSlice.reducer;