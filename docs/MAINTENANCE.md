# Website maintenance

This site intentionally uses **plain HTML + CSS + JavaScript**. There is no build step, package manager, Jekyll theme, or JavaScript framework.

## The two files you will normally edit

### 1. Update leaderboard results

Edit `data/leaderboard.csv` in GitHub's web editor.

Columns:

- `track`: `single` or `multi`. This column is ignored while `multiple_tracks_enabled` is `false`, so the current site shows one combined leaderboard. Keeping the metadata now means you can expose two leaderboards later with one config change.
- `algorithm`: public method/system name.
- `model`: model(s) used by the algorithm.
- `tds`: Turn-Discounted Success as a decimal, e.g. `0.713`.
- `pass_at_1`: decimal in `[0, 1]`, e.g. `0.742`.
- `ndcg`: decimal in `[0, 1]`.
- `clarification_rate`: decimal in `[0, 1]`.
- `over_asking_rate`: decimal in `[0, 1]`; lower is better.
- `avg_cost_usd`: average cost per task in USD, e.g. `0.083`.
- `team`: team or organization name.
- `date`: result date, preferably `YYYY-MM-DD`.
- `submission_url`: PR, paper, code, or trajectory URL.
- `status`: leave blank for real entries. `example` adds an “example row” badge.

The page ranks by `tds` descending. If two entries have the same TDS, `ndcg` descending is the tie-breaker. Pass@1, clarification rate, over-asking rate, and cost never affect rank. Exact TDS+nDCG ties share the same displayed rank. Empty `tds` values appear at the bottom and are not ranked.

Example real row:

```csv
single,Example Clarifier,gpt-example,0.713,0.742,0.881,0.634,0.061,0.083,Example Team,2026-10-01,https://github.com/example/pr/123,
```

Delete the three initial `example` rows once you have real baseline or participant scores.

### 2. Update competition format, deadlines, and track wording

Edit `data/competition.json`.

This controls:

- `multiple_tracks_enabled`: set to `false` for one shared competition track (the current default), or `true` to expose separate Single-turn / Multi-turn selectors and rule cards;
- the short competition summary in the hero;
- the shared `main_track` description used in one-track mode;
- the future single-turn and multi-turn descriptions;
- the private-test placeholder text;
- all deadline/timeline cards.

With the CSV rows already tagged `single` or `multi`, switching `multiple_tracks_enabled` from `false` to `true` is enough to expose the two selectors and split the leaderboard. Switching it back to `false` recombines all rows into one leaderboard.

Keep valid JSON syntax: strings use double quotes and items are separated by commas.

## Less frequent edits

- `index.html`: competition rules, metrics, organizer names, navigation, links.
- `assets/css/style.css`: colors/layout.
- `assets/js/main.js`: optional track toggling, private-results placeholder, CSV loading, and TDS → nDCG ranking logic.

## Local preview

Opening `index.html` directly can block `fetch()` in some browsers. Run a tiny local server instead:

```bash
cd docs
python -m http.server 8000
```

Then open `http://localhost:8000`.

## GitHub Pages setup

In the repository:

1. Put this entire `docs/` directory on the default branch.
2. Go to **Settings → Pages**.
3. Under **Build and deployment**, choose **Deploy from a branch**.
4. Select the default branch and the **`/docs`** folder, then save.

No Actions workflow is required for this dependency-free setup.

## Publishing private results at the end

The current private-test view is intentionally a locked placeholder. A simple approach is to add a second CSV such as `data/private-leaderboard.csv` and update `main.js` to load it when the private result-set button is selected. Until then, the private scores cannot accidentally be exposed in the public site data.
