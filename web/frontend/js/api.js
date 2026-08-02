/**
 * Network layer for the crisis-sim demo.
 *
 * This is the only module that knows about `/api/...` URLs. Everything else
 * receives plain JavaScript objects.
 */

const NETWORK_MESSAGE =
  "We couldn't reach the simulation service. Check that it is running and try again.";

/**
 * GET `path` and parse the JSON body.
 *
 * Throws an Error with a human-readable message on transport failure, a
 * non-2xx status, or an unparseable body -- callers surface `error.message`
 * directly to the user, so it must never contain raw technical detail.
 */
async function getJson(path) {
  let response;
  try {
    response = await fetch(path, { headers: { Accept: 'application/json' } });
  } catch (cause) {
    throw new Error(NETWORK_MESSAGE, { cause });
  }

  if (!response.ok) {
    if (response.status === 404) {
      throw new Error('That announcement is not available in this demo.');
    }
    throw new Error(
      `The simulation service returned an unexpected response (${response.status}).`,
    );
  }

  try {
    return await response.json();
  } catch (cause) {
    throw new Error('The simulation service returned an unreadable response.', {
      cause,
    });
  }
}

/**
 * Fetch the list of available announcements.
 *
 * @returns {Promise<Array<{id: string, company: string, headline: string,
 *   date: string, sector: string}>>} events in the order the API returned them.
 */
export async function fetchEvents() {
  const payload = await getJson('/api/events');
  return Array.isArray(payload?.events) ? payload.events : [];
}

/**
 * Fetch the full simulation result for one announcement.
 *
 * @param {string} eventId
 * @returns {Promise<{event: object, naive: object, ensemble: object,
 *   comparison: object}>}
 */
export async function fetchEventResult(eventId) {
  return getJson(`/api/events/${encodeURIComponent(eventId)}/result`);
}
