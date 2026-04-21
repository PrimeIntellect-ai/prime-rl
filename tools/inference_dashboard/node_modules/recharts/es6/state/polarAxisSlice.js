import { createSlice } from '@reduxjs/toolkit';
import { castDraft } from 'immer';
var initialState = {
  radiusAxis: {},
  angleAxis: {}
};
var polarAxisSlice = createSlice({
  name: 'polarAxis',
  initialState,
  reducers: {
    addRadiusAxis(state, action) {
      state.radiusAxis[action.payload.id] = castDraft(action.payload);
    },
    removeRadiusAxis(state, action) {
      delete state.radiusAxis[action.payload.id];
    },
    addAngleAxis(state, action) {
      state.angleAxis[action.payload.id] = castDraft(action.payload);
    },
    removeAngleAxis(state, action) {
      delete state.angleAxis[action.payload.id];
    }
  }
});
export var {
  addRadiusAxis,
  removeRadiusAxis,
  addAngleAxis,
  removeAngleAxis
} = polarAxisSlice.actions;
export var polarAxisReducer = polarAxisSlice.reducer;