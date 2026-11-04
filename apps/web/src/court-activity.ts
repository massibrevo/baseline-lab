import type {Analysis, Observation, ShotSignal} from './contracts';

export type Player = 'near' | 'far';
export type CourtLayer = 'movement' | 'heatmap' | 'strokes';
export type PlayerFilter = Player | 'both';
type Point = [number, number];
type Link = {start: number; end: number; a: Point; b: Point; cell: string};
type PlayerIndex = {points: Observation[]; links: Link[]};
export type CourtMark = {t_s: number; player?: Player; point: Point; label: string; reviewed: boolean};
const ids: Player[] = ['near', 'far'];
const colors = {near: '#d8ef45', far: '#f0a482'};
const cache = new WeakMap<Analysis, Record<Player, PlayerIndex>>();
const safe = (s: unknown) => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'}[c]!));

function mappingAvailable(result: Analysis, frame: number) {
  return !!(result.frame_calibrations?.length ? result.frame_calibrations[frame] : result.calibration);
}
function supported(result: Analysis, p?: Observation): p is Observation & {court_m: Point} {
  return !!p && p.state === 'detected' && validCourtPoint(p.court_m) && mappingAvailable(result, p.frame);
}
function validCourtPoint(p?: Point | null): p is Point {
  return !!p && p.every(Number.isFinite) && p[0] >= -1.5 && p[0] <= 9.73 && p[1] >= -5 && p[1] <= 28.77;
}
function index(result: Analysis) {
  const previous = cache.get(result); if (previous) return previous;
  const data = Object.fromEntries(ids.map(id => {
    const points = result.tracks.find(t => t.id === id)?.observations ?? [];
    const links: Link[] = [];
    for (let i = 1; i < points.length; i++) {
      const a = points[i - 1], b = points[i], dt = b.t_s - a.t_s;
      if (!supported(result, a) || !supported(result, b) || dt <= 0 || dt > .1 + 1e-9) continue;
      if (Math.hypot(b.court_m[0] - a.court_m[0], b.court_m[1] - a.court_m[1]) / dt > 12) continue;
      // One-metre cells. Real timestamp duration, never frame count, is the weight.
      const cell = `${Math.floor((a.court_m[0] + b.court_m[0]) / 2)},${Math.floor((a.court_m[1] + b.court_m[1]) / 2)}`;
      links.push({start: a.t_s, end: b.t_s, a: a.court_m, b: b.court_m, cell});
    }
    return [id, {points, links}];
  })) as Record<Player, PlayerIndex>;
  cache.set(result, data); return data;
}

export function courtActivity(result: Analysis, frame: number, now: number, source: 'automatic' | 'reviewed') {
  const data = index(result);
  const shots = (source === 'automatic' ? result.automatic_shots ?? [] : (result.shots ?? []).filter(s => s.status === 'reviewed')).filter(s => s.t_s <= now);
  const players = ids.map(id => {
    const links = data[id].links.filter(l => l.end <= now);
    const cells = new Map<string, number>();
    for (const l of links) cells.set(l.cell, (cells.get(l.cell) ?? 0) + l.end - l.start);
    const supportedSeconds = links.reduce((n, l) => n + l.end - l.start, 0);
    const elapsed = Math.max(0, now - (result.metadata.timestamps_s[0] ?? 0));
    const current = data[id].points[frame];
    const sample = result.live_metrics?.[frame];
    const distance = id === 'near' ? sample?.near_observed_distance_m : sample?.far_observed_distance_m;
    const playerShots = shots.filter(s => s.player === id);
    return {id, cells, supportedSeconds, supportFraction: elapsed > 0 ? Math.min(1, supportedSeconds / elapsed) : 0,
      point: supported(result, current) ? current.court_m : null,
      links: links.filter(l => l.end >= now - 3), distance,
      shots: playerShots.filter(s => !s.recognition_basis?.startsWith('clip_boundary')).length,
      boundarySwings: playerShots.filter(s => s.recognition_basis?.startsWith('clip_boundary')).length};
  });
  const marks: CourtMark[] = [];
  for (const s of shots) {
    if (!ids.includes(s.player as Player) || s.recognition_basis?.startsWith('clip_boundary')) continue;
    const p = data[s.player as Player].points[s.frame];
    if (!supported(result, p) || Math.abs(p.t_s - s.t_s) > .1) continue;
    marks.push({t_s: s.t_s, player: s.player as Player, point: p.court_m,
      label: strokeLabel(s), reviewed: source === 'reviewed'});
  }
  const bounces: CourtMark[] = result.events.filter(e => e.t_s <= now && ['manual', 'confirmed'].includes(e.status) && validCourtPoint(e.court_m) && mappingAvailable(result, e.frame))
    .map(e => ({t_s: e.t_s, point: e.court_m!, label: 'Reviewed bounce', reviewed: true}));
  return {players, marks, bounces, mapping: mappingAvailable(result, frame), now};
}
function strokeLabel(s: ShotSignal) { return s.kind === 'unknown' ? 'Uncertain stroke' : s.kind.replaceAll('_', ' '); }
function plot([x, y]: Point): Point { return [35 + x / 8.23 * 140, 20 + y / 23.77 * 300]; }
const f = (n: number) => n.toFixed(2);

export function courtActivityView(result: Analysis, frame: number, now: number, layer: CourtLayer, filter: PlayerFilter, source: 'automatic' | 'reviewed') {
  const state = courtActivity(result, frame, now, source);
  const players = state.players.filter(p => filter === 'both' || filter === p.id);
  let heat = '', paths = '', heads = '', markers = '';
  for (const player of players) {
    const color = colors[player.id];
    if (layer === 'heatmap') for (const [cell, seconds] of player.cells) {
      const [x, y] = cell.split(',').map(Number), [cx, cy] = plot([x + .5, y + .5]);
      heat += `<ellipse cx="${f(cx)}" cy="${f(cy)}" rx="13" ry="11" fill="${color}" opacity="${f(Math.min(.75, seconds / 2))}"><title>${player.id}: ${seconds.toFixed(2)} s supported in this 1 m cell</title></ellipse>`;
    }
    if (layer === 'movement') {
      const lines = player.links.map(l => {const a = plot(l.a), b = plot(l.b); return `M${f(a[0])},${f(a[1])}L${f(b[0])},${f(b[1])}`;}).join('');
      paths += `<path d="${lines}" fill="none" stroke="${color}" stroke-width="2.2" stroke-linecap="round" opacity=".75"/>`;
    }
    if (player.point && state.mapping) {
      const [x, y] = plot(player.point);
      heads += `<circle cx="${f(x)}" cy="${f(y)}" r="7" fill="${color}" stroke="#153b32" stroke-width="2"/><text x="${f(x + 10)}" y="${f(y + 3)}" fill="${color}" font-size="7">${player.id.toUpperCase()}</text>`;
    }
  }
  // Only reviewed ground contacts enter the map; airborne ball pixels never do.
  const marks: CourtMark[] = layer === 'strokes' ? [...state.marks.filter(m => filter === 'both' || m.player === filter), ...state.bounces] : state.bounces;
  for (const m of marks) {
    const [x, y] = plot(m.point), color = m.player ? colors[m.player] : '#5ee7ff';
    const label = `${m.t_s.toFixed(2)} s · ${m.label}${m.player ? ' · ' + m.player + ' player footpoint' : ''}${m.reviewed ? ' · reviewed' : ' · candidate'}`;
    markers += `<g class="court-seek" role="button" tabindex="0" data-court-seek="${m.t_s}" aria-label="Seek to ${safe(label)}"><title>${safe(label)}</title><circle cx="${f(x)}" cy="${f(y)}" r="9" fill="transparent"/><path d="M${f(x-4)},${f(y)}h8M${f(x)},${f(y-4)}v8" stroke="${color}" stroke-width="2" ${m.reviewed ? '' : 'stroke-dasharray="2 1"'}/></g>`;
  }
  const svg = `<svg viewBox="0 -48 210 430" role="group" aria-label="${layer === 'heatmap' ? 'Time-weighted player occupancy' : layer === 'strokes' ? 'Stroke footpoints and reviewed bounces' : 'Recent player movement'} court map"><defs><filter id="court-heat-soft"><feGaussianBlur stdDeviation="3"/></filter><clipPath id="court-map-clip"><rect width="210" y="-48" height="430"/></clipPath></defs><rect width="210" y="-48" height="430" rx="12" fill="#173e34"/><g clip-path="url(#court-map-clip)"><g filter="url(#court-heat-soft)">${heat}</g><g fill="none" stroke="#b9cdbd" stroke-width="1"><rect x="11.67" y="20" width="186.66" height="300"/><rect x="35" y="20" width="140" height="300"/><path d="M35 89.23H175M35 250.77H175M105 89.23V250.77"/></g><path d="M7 170H203" stroke="#eff4e8" stroke-dasharray="2 2" stroke-width="2"/>${paths}${markers}${heads}</g><text x="105" y="10" text-anchor="middle" fill="#b4c9b6" font-size="7" letter-spacing="1.4">FAR BASELINE</text><text x="105" y="337" text-anchor="middle" fill="#b4c9b6" font-size="7" letter-spacing="1.4">NEAR BASELINE</text></svg>`;
  const summaries = players.map(p => `<article class="court-player-summary ${p.id}"><div><b>${p.id.toUpperCase()}</b><span>${p.shots} ${source === 'automatic' ? 'impact candidates' : 'reviewed strokes'}${p.boundarySwings ? ' · ' + p.boundarySwings + ' boundary swing' + (p.boundarySwings > 1 ? 's' : '') : ''}</span></div><div><strong>${p.distance == null ? '—' : '~' + p.distance.toFixed(1) + ' m'}</strong><small>supported path</small></div><div class="court-support"><i style="width:${f(p.supportFraction * 100)}%"></i></div><small>${p.supportedSeconds.toFixed(1)} s mapped · ${Math.round(p.supportFraction * 100)}% of elapsed time</small></article>`).join('');
  const note = !state.mapping ? 'Court mapping unavailable at this frame. Historical support remains visible; current positions are hidden.' : layer === 'heatmap' ? 'Brighter = more supported time. One-metre cells weighted by real timestamps; missing intervals excluded.' : layer === 'strokes' ? 'Crosses show player footpoints at strokes and reviewed bounces. Select a cross to seek. Dashed = automatic candidate.' : 'Last 3 seconds of observed movement. Gaps and jumps above 12 m/s are excluded.';
  return {svg, summaries, note, state};
}
