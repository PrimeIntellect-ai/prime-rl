/**
 * @fileOverview this stores actually rendered ticks.
 *
 * What we do is that we have the domain -> ticks mapping in the cartesianSlice,
 * which is fine but the result then goes to CartesianAxis where we use DOM measurement
 * to decide which ticks to actually render.
 *
 * This renderedTickSlice stores those actually rendered ticks so that we can return them from a hook later.
 */
import { createSlice } from '@reduxjs/toolkit';
import { castDraft } from 'immer';
var initialState = {
  xAxis: {},
  yAxis: {}
};
export var renderedTicksSlice = createSlice({
  name: 'renderedTicks',
  initialState,
  reducers: {
    setRenderedTicks: (state, action) => {
      var {
        axisType,
        axisId,
        ticks
      } = action.payload;
      state[axisType][axisId] = castDraft(ticks);
    },
    removeRenderedTicks: (state, action) => {
      var {
        axisType,
        axisId
      } = action.payload;
      delete state[axisType][axisId];
    }
  }
});
export var {
  setRenderedTicks,
  removeRenderedTicks
} = renderedTicksSlice.actions;
export var renderedTicksReducer = renderedTicksSlice.reducer;