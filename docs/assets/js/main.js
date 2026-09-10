(() => {
  const state = {
    track: 'main',
    resultSet: 'public',
    publicRows: [],
    privateRows: [],
    privateResultsPublished: false,
    multipleTracksEnabled: false,
    competition: {}
  };

  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

  function parseCSV(text) {
    const rows = [];
    let row = [], cell = '', quoted = false;
    for (let i = 0; i < text.length; i++) {
      const char = text[i];
      if (quoted) {
        if (char === '"' && text[i + 1] === '"') { cell += '"'; i++; }
        else if (char === '"') quoted = false;
        else cell += char;
      } else {
        if (char === '"') quoted = true;
        else if (char === ',') { row.push(cell); cell = ''; }
        else if (char === '\n') { row.push(cell); rows.push(row); row = []; cell = ''; }
        else if (char !== '\r') cell += char;
      }
    }
    if (cell.length || row.length) { row.push(cell); rows.push(row); }
    if (!rows.length) return [];
    const headers = rows.shift().map(h => h.trim());
    return rows
      .filter(r => r.some(c => c.trim()))
      .map(r => Object.fromEntries(headers.map((h, i) => [h, (r[i] ?? '').trim()])));
  }

  const numberOrNull = value => value === '' || value === undefined ? null : Number(value);
  const formatPercent = value => value === null ? '—' : `${(value * 100).toFixed(1)}%`;
  const formatScore = value => value === null ? '—' : value.toFixed(3);
  const formatCost = value => value === null ? '—' : `$${value.toFixed(value < 1 ? 3 : 2)}`;
  const escapeHTML = value => String(value ?? '').replace(/[&<>'"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));

  function normalizedRows(rawRows) {
    return rawRows.map(row => ({
      ...row,
      tdsN: numberOrNull(row.tds),
      passN: numberOrNull(row.pass_at_1),
      ndcgN: numberOrNull(row.ndcg),
      clarifyN: numberOrNull(row.clarification_rate),
      overAskN: numberOrNull(row.over_asking_rate),
      costN: numberOrNull(row.avg_cost_usd)
    }));
  }

  function activeRows() {
    return state.resultSet === 'private' ? state.privateRows : state.publicRows;
  }

  function renderLeaderboard() {
    const body = $('#leaderboard-body');
    const rows = activeRows()
      .filter(row => !state.multipleTracksEnabled || row.track === state.track)
      .sort((a, b) => {
        if (a.tdsN === null && b.tdsN === null) return a.algorithm.localeCompare(b.algorithm);
        if (a.tdsN === null) return 1;
        if (b.tdsN === null) return -1;

        const tdsDifference = b.tdsN - a.tdsN;
        if (tdsDifference !== 0) return tdsDifference;

        if (a.ndcgN === null && b.ndcgN !== null) return 1;
        if (a.ndcgN !== null && b.ndcgN === null) return -1;
        if (a.ndcgN !== null && b.ndcgN !== null && a.ndcgN !== b.ndcgN) return b.ndcgN - a.ndcgN;

        // Exact TDS+nDCG ties are shown alphabetically only for deterministic display.
        // No reported metric below nDCG affects the competition ranking.
        return a.algorithm.localeCompare(b.algorithm);
      });

    $('#track-description').textContent = state.multipleTracksEnabled
      ? (state.track === 'single'
        ? 'Single-turn track · at most one clarification turn (confirm final organizer constraint).'
        : 'Multi-turn track · iterative clarification within the organizer-defined budget.')
      : 'One shared clarification track for all eligible systems.';

    if (!rows.length) {
      const file = state.resultSet === 'private' ? 'data/private-leaderboard.csv' : 'data/leaderboard.csv';
      body.innerHTML = `<tr><td colspan="10" class="empty-cell">${state.multipleTracksEnabled ? 'No entries yet for this track.' : 'No entries yet.'} Add a row to <code>${file}</code>.</td></tr>`;
      return;
    }

    let rank = 0;
    let rankedIndex = 0;
    let previousRankedRow = null;
    body.innerHTML = rows.map(row => {
      const isRanked = row.tdsN !== null;
      if (isRanked) {
        rankedIndex += 1;
        const exactTie = previousRankedRow
          && row.tdsN === previousRankedRow.tdsN
          && row.ndcgN === previousRankedRow.ndcgN;
        if (!exactTie) rank = rankedIndex;
        previousRankedRow = row;
      }
      const place = isRanked ? rank : '—';
      const note = row.status === 'example' ? '<span class="placeholder-badge">example row</span>' : '';
      const submission = row.submission_url
        ? `<a class="submission-link" href="${escapeHTML(row.submission_url)}" target="_blank" rel="noreferrer">View ↗</a>`
        : '—';
      return `<tr>
        <td class="rank ${rank <= 3 && isRanked ? 'top' : ''}">${place}</td>
        <td class="algorithm-cell"><strong>${escapeHTML(row.algorithm)}</strong><small>${escapeHTML(row.team || '')}</small>${note}</td>
        <td>${escapeHTML(row.model || '—')}</td>
        <td class="metric-value primary-value">${formatScore(row.tdsN)}</td>
        <td class="metric-value tie-break-value">${formatScore(row.ndcgN)}</td>
        <td class="metric-value">${formatPercent(row.passN)}</td>
        <td class="metric-value">${formatPercent(row.clarifyN)}</td>
        <td class="metric-value">${formatPercent(row.overAskN)}</td>
        <td class="cost-value">${formatCost(row.costN)}</td>
        <td>${submission}</td>
      </tr>`;
    }).join('');
  }

  function updateResultSetContent() {
    const isPrivate = state.resultSet === 'private';
    const privateAvailable = isPrivate && state.privateResultsPublished;

    $('#leaderboard-results').hidden = isPrivate && !state.privateResultsPublished;
    $('#private-results').hidden = !isPrivate || state.privateResultsPublished;

    const eyebrow = $('#leaderboard-eyebrow');
    const summary = $('#leaderboard-summary');
    const status = $('#result-status-text');

    if (isPrivate) {
      if (eyebrow) eyebrow.textContent = 'Final benchmark';
      if (summary) summary.innerHTML = privateAvailable
        ? 'Final rankings below are for the <strong>private test set</strong>.'
        : 'Private-test rankings are <strong>sealed until the competition ends</strong>.';
      if (status) status.textContent = privateAvailable ? 'Private test' : 'Private test · sealed';
    } else {
      if (eyebrow) eyebrow.textContent = 'Public benchmark';
      if (summary) summary.innerHTML = 'Rankings below are for the <strong>public validation set</strong>. Final private-test results remain hidden until the competition ends.';
      if (status) status.textContent = 'Public validation';
    }

    if (!isPrivate || state.privateResultsPublished) renderLeaderboard();
  }

  function setResultSet(resultSet) {
    state.resultSet = resultSet;
    $$('[data-result-set]').forEach(btn => {
      const active = btn.dataset.resultSet === resultSet;
      btn.classList.toggle('is-active', active);
      btn.setAttribute('aria-pressed', String(active));
    });
    updateResultSetContent();
  }

  function setTrack(track) {
    state.track = track;
    $$('[data-track]').forEach(btn => {
      const active = btn.dataset.track === track;
      btn.classList.toggle('is-active', active);
      btn.setAttribute('aria-pressed', String(active));
    });
    if (state.resultSet !== 'private' || state.privateResultsPublished) renderLeaderboard();
  }

  function configureTracks(data) {
    state.multipleTracksEnabled = data.multiple_tracks_enabled === true;
    state.track = state.multipleTracksEnabled ? 'single' : 'main';

    const trackControl = $('#track-control');
    if (trackControl) trackControl.hidden = !state.multipleTracksEnabled;

    const heroTrackCount = $('#hero-track-count');
    const heroTrackLabel = $('#hero-track-label');
    if (heroTrackCount) heroTrackCount.textContent = state.multipleTracksEnabled ? '2' : '1';
    if (heroTrackLabel) heroTrackLabel.textContent = state.multipleTracksEnabled ? 'competition tracks' : 'competition track';

    const mainTrackCard = $('#main-track-card');
    const multiTrackCards = $('#multi-track-cards');
    if (mainTrackCard) mainTrackCard.hidden = state.multipleTracksEnabled;
    if (multiTrackCards) multiTrackCards.hidden = !state.multipleTracksEnabled;

    $$('[data-track]').forEach(btn => {
      const active = btn.dataset.track === state.track;
      btn.classList.toggle('is-active', active);
      btn.setAttribute('aria-pressed', String(active));
    });
  }

  function configurePrivateResults(data) {
    state.privateResultsPublished = data.private_results_published === true;
    const lock = $('#private-lock');
    if (lock) lock.hidden = state.privateResultsPublished;
  }

  function applyCompetitionContent(data) {
    state.competition = data;
    configureTracks(data);
    configurePrivateResults(data);

    $$('[data-content]').forEach(el => {
      const key = el.dataset.content;
      if (Object.prototype.hasOwnProperty.call(data, key)) el.textContent = data[key];
    });

    const timeline = $('#timeline-list');
    if (Array.isArray(data.dates)) {
      timeline.innerHTML = data.dates.map(item => `<article class="timeline-item">
        <div class="timeline-date">${escapeHTML(item.date)}</div>
        <h3>${escapeHTML(item.label)}</h3>
        <p>${escapeHTML(item.description || '')}</p>
      </article>`).join('');
    }
  }

  async function fetchRows(path) {
    const response = await fetch(path, { cache: 'no-store' });
    if (!response.ok) throw new Error(`${path}: ${response.status}`);
    return normalizedRows(parseCSV(await response.text()));
  }

  async function loadData() {
    try {
      const [publicRows, competitionResponse] = await Promise.all([
        fetchRows('data/leaderboard.csv'),
        fetch('data/competition.json', { cache: 'no-store' })
      ]);
      if (!competitionResponse.ok) throw new Error(`competition.json: ${competitionResponse.status}`);

      state.publicRows = publicRows;
      const competition = await competitionResponse.json();
      applyCompetitionContent(competition);

      if (state.privateResultsPublished) {
        try {
          state.privateRows = await fetchRows('data/private-leaderboard.csv');
        } catch (error) {
          console.error(error);
          state.privateRows = [];
        }
      }

      setResultSet('public');
    } catch (error) {
      console.error(error);
      $('#leaderboard-body').innerHTML = '<tr><td colspan="10" class="empty-cell">Could not load leaderboard data. Check <code>data/leaderboard.csv</code> and <code>data/competition.json</code>, then serve the site through GitHub Pages or a local web server.</td></tr>';
    }
  }

  $$('[data-result-set]').forEach(btn => btn.addEventListener('click', () => setResultSet(btn.dataset.resultSet)));
  $$('[data-track]').forEach(btn => btn.addEventListener('click', () => setTrack(btn.dataset.track)));
  $('#year').textContent = new Date().getFullYear();
  loadData();
})();
