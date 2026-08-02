/**
 * Wiring for the crisis-sim demo.
 *
 * This is the only module that listens for DOM events. It owns the small bit
 * of application state (which announcement is selected, which view is active,
 * and an in-memory cache of results already fetched) and delegates all network
 * work to `api.js` and all DOM work to `render.js`.
 */

import { fetchEvents, fetchEventResult } from './api.js';
import {
  populateEventSelect,
  renderComparisonPanel,
  renderEnsemblePanel,
  renderEventContext,
  renderNaivePanel,
  setHint,
  setRunBusy,
  setRunEnabled,
  setVisibleMode,
  showError,
  showLoading,
  showPlaceholder,
  showResult,
  showSelectMessage,
} from './render.js';

/** Purely cosmetic pause on a fresh fetch, so the reveal lands as a moment. */
const REVEAL_DELAY_MS = 400;

const selectEl = document.getElementById('event-select');
const runButton = document.getElementById('run-button');
const modeControl = document.getElementById('mode-control');

/** eventId -> full result payload. Populated on first successful fetch. */
const resultCache = new Map();

const state = {
  mode: currentModeFromDom(),
  /** The event whose payload is currently rendered into the panels, if any. */
  renderedEventId: null,
  /** Guards against an in-flight fetch overwriting a newer request. */
  requestToken: 0,
};

function currentModeFromDom() {
  const checked = document.querySelector('input[name="mode"]:checked');
  return checked ? checked.value : 'comparison';
}

function selectedEventId() {
  return selectEl.value || null;
}

function selectedEventLabel() {
  const option = selectEl.options[selectEl.selectedIndex];
  return option && option.value ? option.textContent : 'this announcement';
}

/** Draw all three views for a payload; `animate` is for fresh runs only. */
function renderPayload(eventId, payload, { animate }) {
  renderEventContext(payload);
  renderNaivePanel(payload);
  renderEnsemblePanel(payload);
  renderComparisonPanel(payload);
  state.renderedEventId = eventId;
  showResult();
  setVisibleMode(state.mode, { animate });
}

/** Prompt shown when the selected announcement has not been run yet. */
function promptForRun() {
  state.renderedEventId = null;
  showPlaceholder(
    selectedEventId()
      ? 'Press Run Simulation to see how this announcement lands.'
      : 'Select a real-world announcement above, then press Run Simulation.',
  );
}

async function handleRun() {
  const eventId = selectedEventId();
  if (!eventId) {
    setHint('Choose an announcement first.');
    selectEl.focus();
    return;
  }
  setHint('');

  // Cache hit: render immediately, with no delay and no network call.
  if (resultCache.has(eventId)) {
    renderPayload(eventId, resultCache.get(eventId), { animate: true });
    return;
  }

  const token = (state.requestToken += 1);
  setRunBusy(true);
  showLoading();

  try {
    const payload = await fetchEventResult(eventId);
    await new Promise((resolve) => setTimeout(resolve, REVEAL_DELAY_MS));
    resultCache.set(eventId, payload);
    if (token !== state.requestToken) return; // a newer request won
    renderPayload(eventId, payload, { animate: true });
  } catch (error) {
    if (token !== state.requestToken) return;
    state.renderedEventId = null;
    showError(error.message);
  } finally {
    if (token === state.requestToken) setRunBusy(false);
  }
}

function handleSelectChange() {
  const eventId = selectedEventId();
  setHint('');
  setRunEnabled(Boolean(eventId));

  // Selecting an announcement already run in this session re-renders straight
  // from the cache -- no network call, no loading state.
  if (eventId && resultCache.has(eventId)) {
    renderPayload(eventId, resultCache.get(eventId), { animate: false });
    return;
  }
  promptForRun();
}

function handleModeChange(event) {
  const target = event.target;
  if (!target || target.name !== 'mode') return;
  state.mode = target.value;

  // Every view is rendered up front, so switching is a pure visibility flip:
  // no fetch, no re-render, no flicker.
  if (state.renderedEventId) {
    setVisibleMode(state.mode, { animate: false });
    return;
  }
  promptForRun();
}

async function init() {
  runButton.addEventListener('click', handleRun);
  selectEl.addEventListener('change', handleSelectChange);
  modeControl.addEventListener('change', handleModeChange);

  setRunEnabled(false);
  try {
    const events = await fetchEvents();
    if (events.length === 0) {
      showSelectMessage('No announcements available');
      setHint('No announcements are available in this demo yet.');
      return;
    }
    populateEventSelect(events);
    setRunEnabled(false);
    setHint(`${events.length} real-world announcements ready to replay.`);
  } catch (error) {
    showSelectMessage('Announcements unavailable');
    setHint('');
    showError(error.message);
  }
}

init();
