(() => {
  const state = { track: 'single', resultSet: 'public', rows: [] };

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
    const headers = rows.shift().map(h => h.trim());
    return rows.filter(r => r.some(c => c.trim())).map(r => Object.fromEntries(headers.map((h, i) => [h, (r[i] ?? '').trim()])));
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

  function renderLeaderboard() {
    const body = $('#leaderboard-body');
    const rows = state.rows
      .filter(row => row.track === state.track)
      .sort((a, b) => {
        if (a.tdsN === null && b.tdsN === null) return a.algorithm.localeCompare(b.algorithm);
        if (a.tdsN === null) return 1;
        if (b.tdsN === null) return -1;
        return b.tdsN - a.tdsN || (a.costN ?? Infinity) - (b.costN ?? Infinity);
      });

    $('#track-description').textContent = state.track === 'single'
      ? 'Single-turn track · at most one clarification turn (confirm final organizer constraint).'
      : 'Multi-turn track · iterative clarification within the organizer-defined budget.';

    if (!rows.length) {
      body.innerHTML = '<tr><td colspan="10" class="empty-cell">No entries yet for this track. Add a row to <code>data/leaderboard.csv</code>.</td></tr>';
      return;
    }

    let rank = 0;
    body.innerHTML = rows.map(row => {
      const isRanked = row.tdsN !== null;
      if (isRanked) rank += 1;
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
        <td class="metric-value">${formatPercent(row.passN)}</td>
        <td class="metric-value">${formatScore(row.ndcgN)}</td>
        <td class="metric-value">${formatPercent(row.clarifyN)}</td>
        <td class="metric-value">${formatPercent(row.overAskN)}</td>
        <td class="cost-value">${formatCost(row.costN)}</td>
        <td>${submission}</td>
      </tr>`;
    }).join('');
  }

  function setResultSet(resultSet) {
    state.resultSet = resultSet;
    $$('[data-result-set]').forEach(btn => {
      const active = btn.dataset.resultSet === resultSet;
      btn.classList.toggle('is-active', active);
      btn.setAttribute('aria-pressed', String(active));
    });
    $('#public-results').hidden = resultSet !== 'public';
    $('#private-results').hidden = resultSet !== 'private';
  }

  function setTrack(track) {
    state.track = track;
    $$('[data-track]').forEach(btn => {
      const active = btn.dataset.track === track;
      btn.classList.toggle('is-active', active);
      btn.setAttribute('aria-pressed', String(active));
    });
    renderLeaderboard();
  }

  function applyCompetitionContent(data) {
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

  async function loadData() {
    try {
      const [leaderboardResponse, competitionResponse] = await Promise.all([
        fetch('data/leaderboard.csv', { cache: 'no-store' }),
        fetch('data/competition.json', { cache: 'no-store' })
      ]);
      if (!leaderboardResponse.ok) throw new Error(`leaderboard.csv: ${leaderboardResponse.status}`);
      if (!competitionResponse.ok) throw new Error(`competition.json: ${competitionResponse.status}`);
      state.rows = normalizedRows(parseCSV(await leaderboardResponse.text()));
      applyCompetitionContent(await competitionResponse.json());
      renderLeaderboard();
    } catch (error) {
      console.error(error);
      $('#leaderboard-body').innerHTML = '<tr><td colspan="10" class="empty-cell">Could not load leaderboard data. Check <code>data/leaderboard.csv</code> and serve the site through GitHub Pages or a local web server.</td></tr>';
    }
  }

  $$('[data-result-set]').forEach(btn => btn.addEventListener('click', () => setResultSet(btn.dataset.resultSet)));
  $$('[data-track]').forEach(btn => btn.addEventListener('click', () => setTrack(btn.dataset.track)));
  $('#year').textContent = new Date().getFullYear();
  loadData();
})();
