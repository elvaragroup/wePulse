/**
 * Pure DOM rendering for the crisis-sim demo.
 *
 * Every function here takes data that is already in memory and writes it into
 * the page. There are no `fetch` calls and no `/api/...` paths in this module.
 * All user-visible text goes through `textContent`, so API strings are never
 * interpreted as markup.
 */

/* ------------------------------------------------------------------ setup */

const dom = {
  select: document.getElementById('event-select'),
  runButton: document.getElementById('run-button'),
  hint: document.getElementById('console-hint'),
  empty: document.getElementById('state-empty'),
  emptyBody: document.getElementById('state-empty-body'),
  loading: document.getElementById('state-loading'),
  error: document.getElementById('state-error'),
  errorBody: document.getElementById('state-error-body'),
  result: document.getElementById('result'),
  context: document.getElementById('event-context'),
  panels: {
    naive: document.getElementById('panel-naive'),
    ensemble: document.getElementById('panel-ensemble'),
    comparison: document.getElementById('panel-comparison'),
  },
  modePanels: {
    naive: document.getElementById('mode-panel-naive'),
    ensemble: document.getElementById('mode-panel-ensemble'),
    comparison: document.getElementById('mode-panel-comparison'),
  },
};

const NO_BACKLASH_CATEGORY = 'none';

const PLATFORM_LABELS = {
  x: 'X',
  reddit: 'Reddit',
  linkedin: 'LinkedIn',
  press: 'Press',
};

const REACTION_LABELS = {
  outrage: 'Outrage',
  criticize: 'Objection',
  mild_concern: 'Mild concern',
  ignore: 'No reaction',
};

/** Display order for the reaction mix bar: loudest first. */
const REACTION_ORDER = ['outrage', 'criticize', 'mild_concern', 'ignore'];

/* -------------------------------------------------------------- utilities */

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = text;
  return node;
}

function clear(node) {
  node.replaceChildren();
}

/** `consumer_tech` -> `Consumer tech`; leaves already-friendly text alone. */
function humanize(value) {
  if (!value) return '';
  const words = String(value).replace(/[_-]+/g, ' ').trim().toLowerCase();
  return words.charAt(0).toUpperCase() + words.slice(1);
}

/** ISO `YYYY-MM-DD` -> `14 June 2026`, parsed as UTC so it never shifts a day. */
function formatDate(iso) {
  const match = /^(\d{4})-(\d{2})-(\d{2})/.exec(String(iso ?? ''));
  if (!match) return String(iso ?? '');
  const date = new Date(Date.UTC(+match[1], +match[2] - 1, +match[3]));
  if (Number.isNaN(date.getTime())) return String(iso);
  return date.toLocaleDateString(undefined, {
    day: 'numeric',
    month: 'long',
    year: 'numeric',
    timeZone: 'UTC',
  });
}

function formatConfidence(confidence) {
  if (typeof confidence !== 'number' || Number.isNaN(confidence)) return null;
  const percent = confidence * 100;
  if (percent > 0 && percent < 1) return '<1%';
  return `${Math.round(percent)}%`;
}

function pluralize(count, singular, plural) {
  return `${count} ${count === 1 ? singular : plural ?? `${singular}s`}`;
}

function platformLabel(platform) {
  return PLATFORM_LABELS[platform] ?? humanize(platform);
}

function reactionLabel(reaction) {
  return REACTION_LABELS[reaction] ?? humanize(reaction);
}

/**
 * Build an id -> label lookup from every category the payload mentions, so
 * quote tags can show the same friendly wording as the ranked lists rather
 * than a bare identifier.
 */
function categoryLabels(payload) {
  const lookup = new Map();
  const groups = [
    payload?.naive?.top_categories,
    payload?.ensemble?.top_categories,
    payload?.comparison?.agreed,
    payload?.comparison?.ensemble_only,
    payload?.comparison?.naive_only,
  ];
  groups.forEach((group) => {
    if (!Array.isArray(group)) return;
    group.forEach((category) => {
      if (category?.id && category?.label) lookup.set(category.id, category.label);
    });
  });
  return lookup;
}

/** Comma-joined category labels, e.g. "Privacy & data and Security". */
function joinLabels(categories) {
  const labels = categories.map((c) => c.label);
  if (labels.length <= 1) return labels.join('');
  if (labels.length === 2) return `${labels[0]} and ${labels[1]}`;
  return `${labels.slice(0, -1).join(', ')}, and ${labels[labels.length - 1]}`;
}

/* --------------------------------------------------------- shared pieces */

function buildVerdict(backlashPredicted, { small = false } = {}) {
  const badge = el(
    'div',
    `verdict ${backlashPredicted ? 'verdict--yes' : 'verdict--no'}${
      small ? ' verdict--sm' : ''
    }`,
  );
  badge.appendChild(el('span', 'verdict__dot'));
  badge.appendChild(
    el(
      'span',
      null,
      backlashPredicted ? 'Backlash predicted: Yes' : 'Backlash predicted: No',
    ),
  );
  return badge;
}

/**
 * Ranked concern list. When `scored` is true and confidence values exist, each
 * row gets a bar sized relative to the strongest concern in the list (so the
 * ranking is legible even when absolute values are small) plus the real
 * percentage alongside it.
 */
function buildConcernList(categories, { scored = false } = {}) {
  const list = el('ol', 'concerns');
  if (!Array.isArray(categories) || categories.length === 0) {
    list.appendChild(
      el('li', 'concern__none', 'No concern areas were surfaced.'),
    );
    return list;
  }

  const values = categories
    .map((c) => (typeof c.confidence === 'number' ? c.confidence : null))
    .filter((v) => v !== null);
  const showBars = scored && values.length > 0;
  const peak = showBars ? Math.max(...values) : 0;

  categories.forEach((category, index) => {
    const value = formatConfidence(category.confidence);
    const hasBar = showBars && typeof category.confidence === 'number';
    const row = el('li', `concern${hasBar ? ' concern--scored' : ''}`);

    row.appendChild(el('span', 'concern__rank', String(index + 1)));
    row.appendChild(el('span', 'concern__name', category.label));

    if (hasBar) {
      const track = el('span', 'concern__bar');
      const fill = el('span');
      const share = peak > 0 ? category.confidence / peak : 0;
      fill.style.width = `${Math.max(6, Math.round(share * 100))}%`;
      track.appendChild(fill);
      row.appendChild(track);
    }

    if (scored) {
      // An em dash keeps the column aligned for categories the ensemble
      // ranked but did not score (e.g. "No meaningful backlash").
      row.appendChild(
        el('span', `concern__value${value ? '' : ' concern__value--none'}`,
          value ?? '—'),
      );
    }
    list.appendChild(row);
  });

  return list;
}

function buildBlock(title, ...children) {
  const block = el('div', 'panel__block');
  block.appendChild(el('h4', 'panel__block-title', title));
  children.filter(Boolean).forEach((child) => block.appendChild(child));
  return block;
}

function buildPanelHead(title, subtitle, badge) {
  const head = el('div', 'panel__head');
  const titles = el('div', 'panel__titles');
  titles.appendChild(el('h3', 'panel__title', title));
  titles.appendChild(el('p', 'panel__subtitle', subtitle));
  head.appendChild(titles);
  if (badge) head.appendChild(badge);
  return head;
}

/* ------------------------------------------------------------ reaction mix */

function buildMixBar(counts) {
  const entries = REACTION_ORDER.map((key) => [key, Number(counts?.[key] ?? 0)]);
  const total = entries.reduce((sum, [, n]) => sum + n, 0);

  const wrap = el('div');
  if (total > 0) {
    const bar = el('div', 'mix-bar');
    entries.forEach(([key, count]) => {
      if (count <= 0) return;
      const seg = el('span', `mix-bar__seg mix-bar__seg--${key}`);
      seg.style.width = `${(count / total) * 100}%`;
      seg.title = `${reactionLabel(key)}: ${count}`;
      bar.appendChild(seg);
    });
    wrap.appendChild(bar);
  }

  const legend = el('div', 'mix-legend');
  entries.forEach(([key, count]) => {
    const item = el('span', 'mix-legend__item');
    item.appendChild(el('span', `mix-legend__swatch mix-bar__seg--${key}`));
    item.appendChild(el('span', 'mix-legend__count', String(count)));
    item.appendChild(el('span', null, reactionLabel(key).toLowerCase()));
    legend.appendChild(item);
  });
  wrap.appendChild(legend);
  return wrap;
}

/* ----------------------------------------------------------------- quotes */

function buildQuote(quote, labels) {
  const card = el('article', `quote quote--${quote.reaction}`);

  const head = el('div', 'quote__head');
  head.appendChild(el('span', 'quote__who', quote.archetype_label));
  head.appendChild(el('span', 'quote__platform', platformLabel(quote.platform)));
  head.appendChild(el('span', 'quote__reaction', reactionLabel(quote.reaction)));

  const filled = Math.min(
    5,
    Math.max(1, Math.round(Number(quote.intensity ?? 0) * 5)),
  );
  const dots = el('span', 'quote__intensity');
  dots.setAttribute('aria-label', `Intensity ${filled} out of 5`);
  for (let i = 0; i < 5; i += 1) {
    dots.appendChild(el('i', i < filled ? 'on' : null));
  }
  head.appendChild(dots);
  card.appendChild(head);

  card.appendChild(el('p', 'quote__text', quote.quote));

  const categories = Array.isArray(quote.categories) ? quote.categories : [];
  if (categories.length > 0) {
    const tags = el('div', 'quote__tags');
    categories.forEach((id) =>
      tags.appendChild(el('span', 'tag', labels?.get(id) ?? humanize(id))),
    );
    card.appendChild(tags);
  }

  return card;
}

function buildQuotes(quotes, limit, labels) {
  const wrap = el('div');
  if (!Array.isArray(quotes) || quotes.length === 0) {
    wrap.appendChild(
      el(
        'p',
        'empty-note',
        'Nobody in the simulated audience spoke up about this one — the room stayed quiet.',
      ),
    );
    return wrap;
  }

  const shown = typeof limit === 'number' ? quotes.slice(0, limit) : quotes;
  const list = el('div', 'quotes');
  shown.forEach((quote) => list.appendChild(buildQuote(quote, labels)));
  wrap.appendChild(list);

  if (shown.length < quotes.length) {
    wrap.appendChild(
      el(
        'p',
        'hint-note',
        `Showing ${shown.length} of ${quotes.length} reactions — switch to the Persona Ensemble view to read them all.`,
      ),
    );
  }
  return wrap;
}

/* ----------------------------------------------------------------- panels */

function buildNaivePanel(payload) {
  const card = el('section', 'card panel panel--naive');

  card.appendChild(
    buildPanelHead(
      'Naive AI',
      'One general-purpose model, asked once, answering on its own.',
      buildVerdict(payload.naive.backlash_predicted, { small: true }),
    ),
  );

  const block = buildBlock(
    'Top concern areas',
    buildConcernList(payload.naive.top_categories, { scored: false }),
  );
  block.appendChild(
    el(
      'p',
      'hint-note',
      'A single call returns a ranked hunch. There is no measure of how many people would care, or how strongly.',
    ),
  );
  card.appendChild(block);

  return card;
}

function buildEnsemblePanel(payload, { quoteLimit } = {}) {
  const ensemble = payload.ensemble;
  const card = el('section', 'card panel panel--ensemble');

  card.appendChild(
    buildPanelHead(
      'Persona Ensemble',
      'Thirty distinct people react independently, then we read the room.',
      buildVerdict(ensemble.backlash_predicted, { small: true }),
    ),
  );

  card.appendChild(
    buildBlock(
      'How the room reacted',
      el('p', 'mix-summary', ensemble.reaction_mix_summary),
      buildMixBar(ensemble.reaction_counts),
    ),
  );

  card.appendChild(
    buildBlock(
      'Top concern areas · confidence',
      buildConcernList(ensemble.top_categories, { scored: true }),
    ),
  );

  card.appendChild(
    buildBlock(
      'What our personas said',
      buildQuotes(ensemble.sample_quotes, quoteLimit, categoryLabels(payload)),
    ),
  );

  return card;
}

/* ------------------------------------------------------- side-by-side bits */

function buildVerdictStrip(payload) {
  const agreed = payload.comparison.backlash_agreement;
  const strip = el(
    'div',
    `verdict-strip${agreed ? '' : ' verdict-strip--disagree'}`,
  );

  const naive = el('span', 'verdict-strip__text');
  naive.appendChild(el('strong', null, 'Naive AI: '));
  naive.appendChild(
    document.createTextNode(
      payload.naive.backlash_predicted ? 'backlash' : 'no backlash',
    ),
  );

  const ensemble = el('span', 'verdict-strip__text');
  ensemble.appendChild(el('strong', null, 'Persona Ensemble: '));
  ensemble.appendChild(
    document.createTextNode(
      payload.ensemble.backlash_predicted ? 'backlash' : 'no backlash',
    ),
  );

  strip.appendChild(naive);
  strip.appendChild(ensemble);
  strip.appendChild(
    el(
      'span',
      'verdict-strip__text',
      agreed
        ? '· Both approaches reached the same headline verdict.'
        : '· The two approaches disagree on the headline verdict.',
    ),
  );
  return strip;
}

/**
 * The callout under the two panels. Three shapes, depending on the data:
 *  - real concerns only the ensemble found  -> the red "would have missed" box
 *  - only "no meaningful backlash"          -> a calm "false alarm" box
 *  - nothing at all                         -> a calm "same conclusion" box
 */
function buildCallout(payload) {
  const comparison = payload.comparison;
  const ensembleOnly = comparison.ensemble_only ?? [];
  const naiveOnly = comparison.naive_only ?? [];
  const missed = ensembleOnly.filter((c) => c.id !== NO_BACKLASH_CATEGORY);
  const overCalled = ensembleOnly.some((c) => c.id === NO_BACKLASH_CATEGORY);

  const box = el('div', 'callout');
  const eyebrow = el('p', 'callout__eyebrow', 'The difference');
  const title = el('h3', 'callout__title');
  const body = el('p', 'callout__body');
  box.append(eyebrow, title, body);

  if (missed.length > 0) {
    title.textContent = 'What a single AI call would have missed';
    body.textContent = `${pluralize(
      missed.length,
      'concern area',
    )} surfaced by the simulated audience never appeared in the single-call answer${
      overCalled
        ? ', which also failed to register how much of the room simply shrugged'
        : ''
    }.`;

    const list = el('div', 'missed');
    missed.forEach((category) => {
      const item = el('div', 'missed__item');
      item.appendChild(el('div', 'missed__name', category.label));
      const value = formatConfidence(category.confidence);
      item.appendChild(
        el(
          'div',
          'missed__value',
          value ? `${value} confidence` : 'raised by the audience',
        ),
      );
      list.appendChild(item);
    });
    box.appendChild(list);
  } else if (overCalled) {
    box.classList.add('callout--amber');
    title.textContent = 'Where a single AI call would have raised a false alarm';
    body.textContent =
      'The simulated audience mostly shrugged — “no meaningful backlash” was the ensemble’s leading finding, and the single-call answer never surfaced it.';
  } else {
    box.classList.add('callout--calm');
    title.textContent = 'Both approaches landed in the same place';
    body.textContent =
      'On this announcement the single-call answer covered the same concern areas the simulated audience raised. The ensemble adds who said it, how loudly, and how many stayed quiet.';
  }

  if (naiveOnly.length > 0) {
    const note = el('p', 'footnote');
    note.appendChild(el('strong', null, 'Also worth noting: '));
    note.appendChild(
      document.createTextNode(
        `the single AI call flagged ${joinLabels(naiveOnly)}, which nobody in the simulated audience actually raised.`,
      ),
    );
    box.appendChild(note);
  }

  return box;
}

/* ------------------------------------------------------- exported renderers */

/**
 * Fill the announcement `<select>` with `{company} — {headline}` options.
 *
 * @param {Array<object>} events
 */
export function populateEventSelect(events) {
  clear(dom.select);
  const placeholder = el('option', null, 'Select an announcement…');
  placeholder.value = '';
  dom.select.appendChild(placeholder);

  events.forEach((event) => {
    const option = el('option', null, `${event.company} — ${event.headline}`);
    option.value = event.id;
    dom.select.appendChild(option);
  });

  dom.select.disabled = events.length === 0;
}

/** Put a single non-selectable message in the `<select>` (load failure, etc.). */
export function showSelectMessage(message) {
  clear(dom.select);
  const option = el('option', null, message);
  option.value = '';
  dom.select.appendChild(option);
  dom.select.disabled = true;
}

/** Render the event context card (company, headline, meta, announcement). */
export function renderEventContext(payload) {
  const event = payload.event;
  const card = el('section', 'card event-card');

  card.appendChild(el('p', 'event-card__company', event.company));
  card.appendChild(el('h2', 'event-card__headline', event.headline));

  const meta = el('div', 'meta-row');
  meta.appendChild(el('span', 'chip', formatDate(event.date)));
  meta.appendChild(el('span', 'chip', humanize(event.sector)));
  if (event.source_url) {
    const link = el('a', 'chip chip--link', 'View source');
    link.href = event.source_url;
    link.target = '_blank';
    link.rel = 'noopener noreferrer';
    meta.appendChild(link);
  }
  card.appendChild(meta);

  if (event.announcement) {
    // Collapsed by default: the audience reaction is the point of the page,
    // and the source text is one click away when a prospect asks for it.
    const details = el('details', 'announcement');
    const summary = el(
      'summary',
      'announcement__summary',
      'Read the announcement',
    );
    details.appendChild(summary);
    details.appendChild(el('div', 'announcement__body', event.announcement));
    card.appendChild(details);
  }

  clear(dom.context);
  dom.context.appendChild(card);
  return card;
}

/** Render the Naive AI panel into its mode container. */
export function renderNaivePanel(payload) {
  const panel = buildNaivePanel(payload);
  clear(dom.panels.naive);
  dom.panels.naive.appendChild(panel);
  return panel;
}

/** Render the Persona Ensemble panel into its mode container. */
export function renderEnsemblePanel(payload) {
  const panel = buildEnsemblePanel(payload);
  clear(dom.panels.ensemble);
  dom.panels.ensemble.appendChild(panel);
  return panel;
}

/** Render the side-by-side comparison, including the difference callout. */
export function renderComparisonPanel(payload) {
  const wrap = el('div');
  wrap.appendChild(buildVerdictStrip(payload));

  const grid = el('div', 'compare-grid');
  grid.appendChild(buildNaivePanel(payload));
  grid.appendChild(buildEnsemblePanel(payload, { quoteLimit: 3 }));
  wrap.appendChild(grid);

  wrap.appendChild(buildCallout(payload));

  clear(dom.panels.comparison);
  dom.panels.comparison.appendChild(wrap);
  return wrap;
}

/**
 * Show one of the three result views.
 *
 * @param {'naive'|'ensemble'|'comparison'} mode
 * @param {{animate?: boolean}} [options] `animate` plays the reveal transition
 *   (used for a fresh run only, never for a cached mode switch).
 */
export function setVisibleMode(mode, { animate = false } = {}) {
  Object.entries(dom.modePanels).forEach(([key, node]) => {
    node.hidden = key !== mode;
  });
  const active = dom.modePanels[mode];
  if (!active) return;
  active.classList.remove('reveal');
  if (animate) {
    // Force a reflow so the animation restarts even on a repeated run.
    void active.offsetWidth;
    active.classList.add('reveal');
  }
}

/* ------------------------------------------------------------ page states */

function showOnly(which) {
  dom.empty.hidden = which !== 'empty';
  dom.loading.hidden = which !== 'loading';
  dom.error.hidden = which !== 'error';
  dom.result.hidden = which !== 'result';
}

/** Show the pre-run placeholder, optionally with a tailored message. */
export function showPlaceholder(message) {
  if (message) dom.emptyBody.textContent = message;
  showOnly('empty');
}

export function showLoading() {
  showOnly('loading');
}

export function showResult() {
  showOnly('result');
}

export function showError(message) {
  dom.errorBody.textContent = message;
  showOnly('error');
}

/** Toggle the Run button between idle and in-flight appearance. */
export function setRunBusy(isBusy) {
  dom.runButton.classList.toggle('is-busy', isBusy);
  dom.runButton.disabled = isBusy || !dom.select.value;
}

/** Enable/disable Run based on whether an announcement is selected. */
export function setRunEnabled(isEnabled) {
  dom.runButton.disabled = !isEnabled;
}

/** Small line of guidance under the controls; pass '' to clear it. */
export function setHint(message) {
  dom.hint.textContent = message ?? '';
}
