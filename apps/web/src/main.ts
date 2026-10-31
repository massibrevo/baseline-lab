import './style.css';
import {openCamera} from './camera';
import type {Analysis, Event as Bounce, Observation} from './contracts';

const $ = <T extends HTMLElement = HTMLElement>(s:string) => document.querySelector<T>(s)!;
const esc = (s:unknown) => String(s ?? '').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]!));
const fmt = (n:number) => `${Math.floor(n/60).toString().padStart(2,'0')}:${(n%60).toFixed(3).padStart(6,'0')}`;
const pct = (n:number) => `${(n*100).toFixed(0)}%`;
let result:Analysis|null = null;
let local = false, demo = true, sid = '', jid = '', title = 'The open court / Example 01';
let selected:Bounce|null = null, tab = 'overview', frame = 0, calibrationMode = false;
let corners:number[][] = [], picked:[number,number]|null = null;
let trailStyle = 'neutral';
let sessionMeta:any = null;
let pollTimer:number|undefined;
let sessions:any[] = [];
const overlays:Record<string,boolean> = {Ball:true, Players:true, Court:true, Events:true};

async function api(path:string, init:RequestInit = {}) {
  const headers = new Headers(init.headers);
  headers.set('X-Baseline-Request','1');
  if(init.body && !(init.body instanceof FormData)) headers.set('Content-Type','application/json');
  const r = await fetch('/api/v1'+path, {...init,headers,credentials:'same-origin'});
  if(!r.ok) { const e = await r.json().catch(()=>({detail:r.statusText})); throw Error(typeof e.detail==='string'?e.detail:JSON.stringify(e.detail)); }
  return r.status===204 ? null : r.json();
}
function notice(message:string, error=false) {
  $('#notice').innerHTML = `<span>${esc(message)}</span><button aria-label="Dismiss message">×</button>`;
  $('#notice').className = `notice ${error?'error':''}`;
  $('#notice button').onclick = ()=> $('#notice').className='notice hidden';
}
function shell() {
  $('#app').innerHTML = `
  <aside class="sidebar"><a class="brand" href="/" aria-label="Baseline Lab home"><svg viewBox="0 0 30 34"><path d="M3 3h24v28H3zm0 14h24M15 3v28"/></svg><span>BASELINE<span class="brand-lab">LAB</span></span></a>
  <div class="sidebar-caption">THE ANALYSIS DESK</div><button class="nav selected" id="desk"><span>▦</span> Workspace <small>01</small></button><button class="nav" id="session-nav"><span>▤</span> Your sessions</button><button class="nav" id="method-nav"><span>⌁</span> Field notes</button>
  <div class="sidebar-session"><span class="eyebrow">CURRENT SESSION</span><b id="side-title">Example 01</b><span id="side-status">Public reference clip</span><div class="tiny-court"><i></i><i></i></div></div>
  <div class="sidebar-bottom"><span class="live-dot"></span><span id="connection">Checking local service…</span><p>Observe. Question. Improve.</p><a href="https://github.com/massibrevo/baseline-lab" target="_blank" rel="noreferrer">Source & documentation ↗</a><span class="version">RESEARCH RELEASE / v0.1</span></div></aside>
  <main><header><div class="breadcrumb">Workspace <span>/</span> <b id="header-title">Example 01</b></div><div class="header-actions"><span class="status" id="mode">DEMO / PRECOMPUTED</span><button id="camera" class="button">◉ Live camera</button><button id="upload" class="button primary">＋ Upload clip</button></div></header>
  <div id="notice" class="notice hidden" role="status"></div>
  <section class="workspace"><div class="heading"><div><div class="eyebrow">RALLY STUDY <span class="tiny-number">/ 001</span></div><h1 id="session-title">The open court.</h1><p id="subtitle">One rally. A different perspective.</p></div><div class="heading-actions"><button class="button" id="export">Export ↗</button><button class="icon-button" id="delete" aria-label="Delete session">⌫</button></div></div>
  <div class="demo-note" id="demo-note"><b>PUBLIC DEMONSTRATION</b><span>Real footage. Pipeline-generated tracks. Detection failures are part of this study.</span></div>
  <div class="tabs" role="tablist">${['overview','events','tracks','methodology'].map(t=>`<button role="tab" aria-selected="${t==='overview'}" class="tab ${t==='overview'?'active':''}" data-tab="${t}">${t[0].toUpperCase()+t.slice(1)}</button>`).join('')}<span class="tab-note">SINGLES / ONE CAMERA</span></div>
  <div class="analysis-grid"><section class="video-column"><div class="video-top"><span>01 <b>VIDEO REVIEW</b></span><span id="video-meta">Loading reference…</span></div><div class="video-wrap" id="video-wrap"><video id="video" playsinline preload="metadata" aria-label="Tennis match review video"></video><canvas id="overlay" aria-label="Synchronized tracking overlays"></canvas><div class="video-label"><span class="live-dot"></span><span id="video-label">REFERENCE FOOTAGE</span></div><button class="big-play" id="big-play" aria-label="Play video">▶</button></div>
  <div class="transport"><div><button class="transport-button" id="prev-frame" aria-label="Previous frame" title="Previous frame (←)">|◀</button><button class="transport-button play" id="play" aria-label="Play video" title="Play / pause (Space)">▶</button><button class="transport-button" id="next-frame" aria-label="Next frame" title="Next frame (→)">▶|</button><span class="time" id="time">00:00.000</span><span class="duration" id="duration">/ —</span></div><div><label class="sr-only" for="speed">Playback speed</label><select id="speed"><option value="0.25">0.25×</option><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option></select><button class="transport-button" id="fullscreen" aria-label="Fullscreen video">⛶</button></div></div>
  <div class="overlay-bar"><span>OVERLAYS</span>${Object.keys(overlays).map(k=>`<label><input type="checkbox" checked data-overlay="${k}"><span>${k}</span></label>`).join('')}<select id="trail" aria-label="Ball trail style"><option value="neutral">Neutral trail</option><option value="chartreuse">Chartreuse trail</option><option value="points">Points only</option></select></div>
  <section class="timeline"><div class="section-label"><span>RALLY TIMELINE</span><small>Click to seek · ← → step · Space play</small></div><div id="timeline" class="timeline-track" role="slider" tabindex="0" aria-label="Video timeline" aria-valuemin="0" aria-valuemax="100" aria-valuenow="0"><canvas id="coverage"></canvas><div id="event-ticks"></div><i id="playhead"></i></div><div class="timeline-scale"><span>00:00</span><span id="half-time">—</span><span id="end-time">—</span></div><div class="legend"><span><i class="solid"></i>Observed ball</span><span><i class="dashed"></i>Interpolated</span><span><i class="event-dot"></i>Review event</span><span class="legend-right">Gaps = missing observations</span></div></section>
  <section class="live-desk"><div class="section-label"><span>LIVE MEASUREMENTS</span><small id="live-clock">SYNCHRONIZED TO VIDEO</small></div><div class="live-grid" id="live-metrics"></div><p class="small muted" id="live-evidence"></p></section><div class="metrics" id="metrics"></div><section class="detail-section" id="detail"></section></section>
  <aside class="inspector"><section class="court-panel"><div class="section-label"><span>02 <b>COURT VIEW</b></span><small>METERS</small></div><div id="minimap"></div><div class="court-key"><span><i class="near-dot"></i>Near</span><span><i class="far-dot"></i>Far</span></div><p class="muted small" id="court-note">Grounded player footpoints only. Airborne ball stays in image space.</p></section>
  <section class="review-panel"><div class="section-label"><span>EVENT REVIEW</span><span id="event-count">—</span></div><div id="event-list"></div><button class="button full" id="add-event">＋ Mark ground contact</button><div id="event-editor"></div></section>
  <section class="quality-panel"><div class="section-label"><span>SIGNAL QUALITY</span><span class="estimate">ESTIMATES</span></div><div id="quality"></div><p class="small muted">Scores describe model output, not calibrated probabilities. This is a review tool.</p></section></aside></div>
  <footer><span>BASELINE LAB <i>—</i> A study in seeing the game.</span><span>Space to play · Arrows to step</span></footer></section></main>
  <dialog id="dialog"><button id="close-dialog" class="dialog-close" aria-label="Close dialog">×</button><div id="dialog-content"></div></dialog>`;
  $('#desk').onclick=()=>selectTab('overview');
  $('#method-nav').onclick=()=>selectTab('methodology');
  $('#session-nav').onclick=showSessions;
  $('#upload').onclick=showUpload;
  $('#camera').onclick=()=>openCamera({local,api,modal,notice});
  $('#export').onclick=showExports;
  $('#delete').onclick=deleteSession;
  $('#close-dialog').onclick=()=>{calibrationMode=false;$<HTMLDialogElement>('#dialog').close();};
  document.querySelectorAll<HTMLButtonElement>('[data-tab]').forEach(b=>b.onclick=()=>selectTab(b.dataset.tab!));
  $('#play').onclick=togglePlay; $('#big-play').onclick=togglePlay;
  $('#prev-frame').onclick=()=>step(-1);$('#next-frame').onclick=()=>step(1);
  $<HTMLSelectElement>('#speed').onchange=e=>{$<HTMLVideoElement>('#video').playbackRate=Number((e.target as HTMLSelectElement).value);};
  $('#fullscreen').onclick=()=>$('#video-wrap').requestFullscreen();
  document.querySelectorAll<HTMLInputElement>('[data-overlay]').forEach(b=>b.onchange=()=>{overlays[b.dataset.overlay!]=b.checked;draw();});
  $<HTMLSelectElement>('#trail').onchange=e=>{trailStyle=(e.target as HTMLSelectElement).value;draw();};
  $('#timeline').onclick=e=>{const r=$('#timeline').getBoundingClientRect();seek((e.clientX-r.left)/r.width*(result?.metadata.duration_s||0));};
  $('#timeline').onkeydown=e=>{if(e.key==='ArrowRight')step(1);if(e.key==='ArrowLeft')step(-1);};
  $('#add-event').onclick=()=>{if(!result)return;selected=null;picked=null;renderEditor();notice('Pause and click the ball’s ground-contact location in the video, then save the event.');};
  $('#overlay').onclick=e=>{if(!result)return;const r=$('#overlay').getBoundingClientRect();picked=[(e.clientX-r.left)/r.width*result.metadata.width,(e.clientY-r.top)/r.height*result.metadata.height];renderEditor();draw();};
  const video=$<HTMLVideoElement>('#video');
  video.onplay=()=>{$('#play').textContent='Ⅱ';$('#play').setAttribute('aria-label','Pause video');$('#big-play').classList.add('hidden');};
  video.onpause=()=>{$('#play').textContent='▶';$('#play').setAttribute('aria-label','Play video');$('#big-play').classList.remove('hidden');};
  video.onseeked=draw; video.ontimeupdate=draw; video.onloadedmetadata=draw;
  if('requestVideoFrameCallback' in video){const tick=()=>{draw();video.requestVideoFrameCallback(tick);};video.requestVideoFrameCallback(tick);}
  new ResizeObserver(()=>{draw();renderCoverage();}).observe($('#video-wrap'));
  document.addEventListener('keydown',e=>{if((e.target as HTMLElement).matches('input,textarea,select')||$<HTMLDialogElement>('#dialog').open)return;if(e.code==='Space'){e.preventDefault();togglePlay();}if(e.key==='ArrowLeft'){e.preventDefault();step(-1);}if(e.key==='ArrowRight'){e.preventDefault();step(1);}});
}
function togglePlay(){const v=$<HTMLVideoElement>('#video');if(v.paused)v.play().catch(e=>notice(e.message,true));else v.pause();}
function seek(t:number){const v=$<HTMLVideoElement>('#video');v.currentTime=Math.max(0,Math.min(t,result?.metadata.duration_s??v.duration));draw();}
function currentFrame(t:number):number {if(!result)return 0;const ts=result.metadata.timestamps_s;let lo=0,hi=ts.length-1;while(lo<hi){const m=Math.ceil((lo+hi)/2);if(ts[m]<=t+0.001)lo=m;else hi=m-1;}return lo;}
function step(direction:number){const v=$<HTMLVideoElement>('#video');v.pause();if(result){const f=Math.max(0,Math.min(result.metadata.frames-1,currentFrame(v.currentTime)+direction));seek(result.metadata.timestamps_s[f]);const ti=document.querySelector<HTMLInputElement>('#event-time');if(ti)ti.value=result.metadata.timestamps_s[f].toFixed(3);}else seek(v.currentTime+direction/30);}
function selectTab(t:string){tab=t;document.querySelectorAll<HTMLElement>('[data-tab]').forEach(b=>{b.classList.toggle('active',b.dataset.tab===t);b.setAttribute('aria-selected',String(b.dataset.tab===t));});renderDetail();}
function artifact(name:string){return `/api/v1/sessions/${sid}/jobs/${jid}/artifacts/${name}`;}
function installResult(data:Analysis, videoUrl:string){
  result=data;selected=null;picked=null;
  $('#video-wrap').style.aspectRatio=String(data.metadata.width/data.metadata.height);
  const v=$<HTMLVideoElement>('#video');v.src=videoUrl;v.load();
  $('#session-title').textContent=demo?'The open court.':title.replace(/\.[^.]+$/,'');
  $('#header-title').textContent=demo?'Example 01':title;
  $('#side-title').textContent=demo?'Example 01':title;
  $('#side-status').textContent=demo?'Public reference clip':'Private local session';
  $('#subtitle').textContent=demo?'A real-world study of detection, geometry and the gaps between.':'Your footage, processed on this machine.';
  $('#mode').textContent=demo?'DEMO / PRECOMPUTED':'LOCAL / PRIVATE';
  $('#demo-note').classList.toggle('hidden',!demo);
  $('#video-label').textContent=demo?'REFERENCE FOOTAGE':'PRIVATE FOOTAGE';
  $('#video-meta').textContent=`${data.metadata.width} × ${data.metadata.height} · ${data.metadata.codec.toUpperCase()} · ${data.metadata.variable_frame_rate?'VFR':data.metadata.nominal_fps+' FPS'}`;
  $('#duration').textContent='/ '+fmt(data.metadata.duration_s);$('#half-time').textContent=fmt(data.metadata.duration_s/2).slice(0,5);$('#end-time').textContent=fmt(data.metadata.duration_s).slice(0,5);
  $('#event-count').textContent=String(data.events.filter(e=>e.status!=='rejected').length);
  $('#metrics').innerHTML=[['Clip duration',data.metadata.duration_s.toFixed(2),'s'],['Ball observations',pct(data.tracks[0].coverage),'coverage'],['Review events',String(data.events.filter(e=>e.status!=='rejected').length),'candidates + manual'],['Ball speed · 3D','—','unavailable · single view']].map(([l,v,u])=>`<div><span>${l}</span><strong>${v}</strong><small>${u}</small></div>`).join('');
  $('#quality').innerHTML=data.tracks.map(t=>`<div class="quality-row"><span>${esc(t.id==='ball'?'Ball observations':t.id+' player')}</span><b>${pct(t.coverage)}</b><div><i style="width:${t.coverage*100}%"></i></div></div>`).join('');
  if(demo && data.provenance.provisional_visual_check){const check=data.provenance.provisional_visual_check.ball;$('#quality').insertAdjacentHTML('beforeend',`<p class="validation-note">Provisional point check: ${check.hits_within_tolerance}/${check.n_labeled} ball locations within ${check.tolerance_px} px. Small visual-label sample; not an accuracy benchmark.</p>`);}
  renderEvents();renderDetail();renderCoverage();draw();
}
function draw(){
  if(!result)return;
  const v=$<HTMLVideoElement>('#video');frame=currentFrame(v.currentTime);
  $('#time').textContent=fmt(v.currentTime);$('#playhead').style.left=`${v.currentTime/result.metadata.duration_s*100}%`;
  $('#timeline').setAttribute('aria-valuenow',String(Math.round(v.currentTime/result.metadata.duration_s*100)));
  const canvas=$<HTMLCanvasElement>('#overlay');canvas.width=result.metadata.width;canvas.height=result.metadata.height;
  const box=$('#video-wrap').getBoundingClientRect();const fit=Math.min(box.width/canvas.width,box.height/canvas.height);canvas.style.width=`${canvas.width*fit}px`;canvas.style.height=`${canvas.height*fit}px`;canvas.style.left=`${(box.width-canvas.width*fit)/2}px`;canvas.style.top=`${(box.height-canvas.height*fit)/2}px`;canvas.style.right='auto';canvas.style.bottom='auto';
  const c=canvas.getContext('2d')!;const scale=result.metadata.width/1000;
  c.lineWidth=2*scale;c.font=`${13*scale}px monospace`;
  if(overlays.Court && result.calibration){const points=result.calibration.corners_px;c.strokeStyle='rgba(243,240,230,.8)';c.beginPath();points.forEach((p:number[],i:number)=>i?c.lineTo(p[0],p[1]):c.moveTo(p[0],p[1]));c.closePath();c.stroke();}
  for(const track of result.tracks){
    const p=track.observations[frame];if(!p)continue;
    if(track.id==='ball' && overlays.Ball){
      const tail=track.observations.slice(Math.max(0,frame-15),frame+1);
      c.strokeStyle=trailStyle==='chartreuse'?'#D8EF45':'#F3F0E6';
      if(trailStyle!=='points')for(let i=1;i<tail.length;i++){const a=tail[i-1],b=tail[i];if(!a.xy_px||!b.xy_px)continue;c.setLineDash(a.state==='interpolated'||b.state==='interpolated'?[5*scale,5*scale]:[]);c.beginPath();c.moveTo(...a.xy_px);c.lineTo(...b.xy_px);c.stroke();}
      c.setLineDash([]);if(p.xy_px){c.strokeStyle='#D8EF45';c.beginPath();c.arc(...p.xy_px,7*scale,0,Math.PI*2);c.stroke();if(p.state==='interpolated'){c.fillStyle='#F3F0E6';c.fillRect(p.xy_px[0]-2,p.xy_px[1]-2,4,4);}}
    } else if(track.id!=='ball'&&overlays.Players&&p.bbox_px){const [x,y,x2,y2]=p.bbox_px;c.strokeStyle=track.id==='near'?'#D8EF45':'#D68D6D';c.strokeRect(x,y,x2-x,y2-y);c.fillStyle=c.strokeStyle;c.fillText(track.id.toUpperCase(),x,y-7*scale);}
  }
  if(overlays.Events)for(const e of result.events){if(e.status==='rejected'||Math.abs(e.t_s-v.currentTime)>.15)continue;c.strokeStyle='#D8EF45';c.beginPath();c.arc(...e.xy_px,14*scale,0,Math.PI*2);c.stroke();c.fillStyle='#F3F0E6';c.fillText(e.status==='candidate'?'CONTACT?':'REVIEWED',e.xy_px[0]+18,e.xy_px[1]);}
  if(picked){c.strokeStyle='#D8EF45';c.strokeRect(picked[0]-10,picked[1]-10,20,20);}
  renderMinimap();renderLiveMetrics();
}
function renderLiveMetrics(){
  if(!result)return;
  const sample=result.live_metrics?.[frame];
  const value=(v:number|null|undefined,unit:string)=>v==null?'—':`${unit==='px/s'?Math.round(v):v.toFixed(1)} <small>${unit}</small>`;
  const t=$<HTMLVideoElement>('#video').currentTime;
  const contact=[...result.events].filter(e=>e.t_s<=t && e.court_m && ['manual','confirmed'].includes(e.status)).sort((a,b)=>b.t_s-a.t_s)[0];
  const depth=contact?.court_m?Math.min(contact.court_m[1],23.77-contact.court_m[1]):null;
  $('#live-clock').textContent=fmt(t)+' / PLAYBACK';
  $('#live-metrics').innerHTML=[
    ['Ball image motion',value(sample?.ball_image_speed_px_s,'px/s'),'2D observed track · not km/h'],
    ['Shot depth',value(depth,'m'),'Approx. from receiving baseline'],
    ['Near player speed',value(sample?.near_speed_m_s==null?null:sample.near_speed_m_s*3.6,'km/h'),'Approximate ground-plane speed'],
    ['Far player speed',value(sample?.far_speed_m_s==null?null:sample.far_speed_m_s*3.6,'km/h'),'Approximate ground-plane speed'],
    ['Near observed path',value(sample?.near_observed_distance_m,'m'),'Contiguous observations so far'],
    ['Far observed path',value(sample?.far_observed_distance_m,'m'),'Contiguous observations so far']
  ].map(([label,v,note])=>`<div><span>${label}</span><strong>${v}</strong><small>${note}</small></div>`).join('');
  $('#live-evidence').textContent=(sample?.ball_image_speed_px_s==null?'Ball motion unavailable at this frame: missing or discontinuous observations. ':'Ball motion uses real PTS over a trailing 50–150 ms observed window. ')+(contact?`Depth uses reviewed contact at ${fmt(contact.t_s)}; negative = beyond baseline. `:'Mark a reviewed ground contact to calculate court depth. ')+'Airborne 3D depth and physical ball speed are unvalidated.';
}
function renderMinimap(){
  if(!result)return;
  const spot=(p:[number,number],color:string,label:string)=>`<circle cx="${35+p[0]/8.23*140}" cy="${20+p[1]/23.77*300}" r="5" fill="${color}"/><text x="${45+p[0]/8.23*140}" y="${24+p[1]/23.77*300}" fill="#F3F0E6" font-size="8">${label}</text>`;
  let spots='';for(const t of result.tracks.filter(t=>t.id!=='ball')){const p=t.observations[frame];if(p?.court_m)spots+=spot(p.court_m,t.id==='near'?'#D8EF45':'#C86C45',t.id.toUpperCase());}
  for(const e of result.events)if(e.court_m&&e.status!=='rejected'&&e.status!=='candidate')spots+=spot(e.court_m,'#F3F0E6','');
  $('#minimap').innerHTML=`<svg viewBox="0 0 210 345" aria-label="Singles court minimap"><rect width="210" height="345" fill="#21493D"/><g fill="none" stroke="#C2C9B8" stroke-width="1"><rect x="15" y="20" width="180" height="300"/><rect x="35" y="20" width="140" height="300"/><path d="M35 87H175M35 253H175M105 87V253M10 170H200"/></g><path d="M10 170H200" stroke="#F3F0E6" stroke-dasharray="2 2" stroke-width="2"/><text x="105" y="12" text-anchor="middle" fill="#AEBEAE" font-size="7" letter-spacing="2">FAR BASELINE</text><text x="105" y="337" text-anchor="middle" fill="#AEBEAE" font-size="7" letter-spacing="2">NEAR BASELINE</text>${spots}${!result.calibration?'<text x="105" y="160" text-anchor="middle" fill="#F3F0E6" font-size="10">Calibration unavailable</text><text x="105" y="185" text-anchor="middle" fill="#C2C9B8" font-size="8">No metric positions plotted</text>':''}</svg>`;
  $('#court-note').textContent=result.calibration?'Approximate footpoints. Unvalidated manual calibration; ball contact appears only after human review.':'No calibration. Player meters and ground-contact locations are unavailable.';
}
function renderCoverage(){
  if(!result)return;const cv=$<HTMLCanvasElement>('#coverage');cv.width=Math.max(1,cv.clientWidth*devicePixelRatio);cv.height=56*devicePixelRatio;const c=cv.getContext('2d')!;
  const points=result.tracks[0].observations;
  points.forEach(p=>{if(p.state==='missing'||p.state==='rejected')return;const x=p.t_s/result!.metadata.duration_s*cv.width;c.fillStyle=p.state==='detected'?'#6A8B73':'#C1C69A';c.fillRect(x,14*devicePixelRatio,Math.max(1,cv.width/points.length),28*devicePixelRatio);});
  $('#event-ticks').innerHTML=result.events.filter(e=>e.status!=='rejected').map(e=>`<button title="${esc(e.status)} at ${fmt(e.t_s)}" aria-label="Review event at ${fmt(e.t_s)}" data-event="${e.id}" style="left:${e.t_s/result!.metadata.duration_s*100}%"></button>`).join('');
  $('#event-ticks').querySelectorAll<HTMLButtonElement>('button').forEach(b=>b.onclick=e=>{e.stopPropagation();review(result!.events.find(x=>x.id===b.dataset.event)!);});
}
function renderEvents(){
  if(!result)return;
  const activeCount=String(result.events.filter(e=>e.status!=='rejected').length);$('#event-count').textContent=activeCount;const eventMetric=document.querySelector('#metrics>div:nth-child(3) strong');if(eventMetric)eventMetric.textContent=activeCount;
  $('#event-list').innerHTML=result.events.length?result.events.map((e,i)=>`<button class="event-item ${selected?.id===e.id?'selected':''}" data-id="${e.id}"><span class="event-index">${String(i+1).padStart(2,'0')}</span><span><b>${e.status==='candidate'?'Possible contact':e.status==='rejected'?'Rejected candidate':'Reviewed contact'}</b><small>${fmt(e.t_s)} · ${e.classification.replace('_',' ')}</small></span><span>↗</span></button>`).join(''):'<div class="empty-events"><span>⌁</span><b>No supported bounce candidates</b><p>Review the footage and mark a ground contact manually. Missing events are kept missing.</p></div>';
  document.querySelectorAll<HTMLButtonElement>('[data-id]').forEach(b=>b.onclick=()=>review(result!.events.find(e=>e.id===b.dataset.id)!));
}
function review(e:Bounce){selected=e;picked=e.xy_px;$<HTMLVideoElement>('#video').pause();$<HTMLVideoElement>('#video').playbackRate=.25;$<HTMLSelectElement>('#speed').value='0.25';seek(e.t_s);renderEvents();renderEditor();}
function renderEditor(){
  if(!result)return;
  $('#event-editor').innerHTML=`<div class="editor"><h3>${selected?'Review contact':'Add ground contact'}</h3><p>Click the contact location in the paused video. Frame-step to adjust timing.</p><label>Time (seconds)<input id="event-time" type="number" step="0.001" min="0" max="${result.metadata.duration_s}" value="${$<HTMLVideoElement>('#video').currentTime.toFixed(3)}"></label><div class="coordinate">${picked?`${Math.round(picked[0])}, ${Math.round(picked[1])} px`:'Choose a point in the video'}</div><label>Review notes<textarea id="event-notes" maxlength="1000">${esc(selected?.notes||'')}</textarea></label><div class="editor-actions"><button class="button primary" id="save-event">${selected?'Confirm':'Save contact'}</button>${selected?'<button class="button" id="reject-event">Reject</button>':''}<button class="button" id="close-editor">Close</button></div><p class="small muted">Line classification stays uncertain until independent error bounds are validated.</p></div>`;
  $('#close-editor').onclick=()=>{selected=null;picked=null;$('#event-editor').innerHTML='';draw();};
  $('#save-event').onclick=()=>saveEvent(selected?'confirmed':'manual');
  if(selected)$('#reject-event').onclick=()=>saveEvent('rejected');
}
async function saveEvent(status:'confirmed'|'manual'|'rejected'){
  if(!picked||!result){notice('Choose a contact point in the video first.',true);return;}
  const t=Number($<HTMLInputElement>('#event-time').value);if(!Number.isFinite(t)||t<0||t>result.metadata.duration_s){notice('Time must be inside the clip.',true);return;}
  const body={status,t_s:t,xy_px:picked,notes:$<HTMLTextAreaElement>('#event-notes').value};
  try{if(demo){const e=selected??{id:'manual-'+crypto.randomUUID(),t_s:t,frame:currentFrame(t),xy_px:picked,evidence_score:null,score_semantics:'manual annotation',status:'manual',classification:'uncertain',court_m:null,original_prediction:null,notes:''} as Bounce;if(selected&&!e.original_prediction)e.original_prediction={...e};Object.assign(e,body,{frame:currentFrame(t),court_m:null});if(status!=='rejected'&&result.calibration){const h=result.calibration.homography;const [x,y]=picked;const w=h[2][0]*x+h[2][1]*y+h[2][2];if(Math.abs(w)>1e-10){e.court_m=[(h[0][0]*x+h[0][1]*y+h[0][2])/w,(h[1][0]*x+h[1][1]*y+h[1][2])/w];e.shot_depth_m=Math.min(e.court_m[1],23.77-e.court_m[1]);}}if(!selected)result.events.push(e);localStorage.setItem('baseline-demo-events',JSON.stringify(result.events));notice('Demo correction saved in this browser. It does not validate a line call.');}else{result=await api(`/sessions/${sid}/jobs/${jid}/events${selected?'/'+selected.id:''}`,{method:selected?'PATCH':'POST',body:JSON.stringify(body)});notice('Correction saved with original prediction preserved.');}selected=null;picked=null;$('#event-editor').innerHTML='';renderEvents();renderCoverage();renderDetail();draw();}catch(e){notice(String(e),true);}
}
function renderDetail(){
  if(!result){$('#detail').innerHTML='<p class="muted">Loading the reproducible analysis…</p>';return;}
  if(tab==='overview')$('#detail').innerHTML=`<div class="section-label"><span>READING THE RALLY</span><small>METHOD, NOT MAGIC</small></div><div class="insight-grid"><div><h3>Follow the evidence.</h3><p>Solid trail segments are observations. Dashed segments fill gaps of at most 100 ms. Empty intervals remain empty.</p></div><div><h3>Keep the perspective.</h3><p>A court homography locates grounded points. It cannot reveal the true court position or speed of an airborne ball.</p></div></div>${demo?`<p class="attribution">Example footage: <a href="https://commons.wikimedia.org/wiki/File:Carlos_Alcaraz_vs._Wu_Yibing_(2026_US_Open)_-_01.webm" target="_blank" rel="noreferrer">Oleg Yunakov / Wikimedia Commons</a> · CC BY-SA 4.0 · stabilized, transcoded to 30 fps; overlays by Baseline Lab.</p>`:`<p class="attribution">Source: your private local clip. No footage is published by this application.</p>`}`;
  if(tab==='events')$('#detail').innerHTML=`<h2>Contact ledger</h2><p>Candidate events require human confirmation. Scores are heuristic, not probabilities.</p><div class="table-wrap"><table><thead><tr><th>Time</th><th>Status</th><th>Line estimate</th><th>Evidence score</th></tr></thead><tbody>${result.events.map(e=>`<tr><td>${fmt(e.t_s)}</td><td>${esc(e.status)}</td><td>${e.classification}</td><td>${e.evidence_score?.toFixed(2)??'manual'}</td></tr>`).join('')||'<tr><td colspan="4">No events. Add a reviewed ground contact from the video.</td></tr>'}</tbody></table></div>`;
  if(tab==='tracks')$('#detail').innerHTML=`<h2>Track ledger</h2><div class="table-wrap"><table><thead><tr><th>Track</th><th>Observed</th><th>Missing</th><th>Contiguous path</th></tr></thead><tbody>${result.tracks.map(t=>`<tr><td>${esc(t.id)}</td><td>${pct(t.coverage)}</td><td>${t.observations.filter(p=>p.state==='missing').length} frames</td><td>${t.id==='ball'?'—':t.distance_m===null?'— unavailable':t.distance_m.toFixed(1)+' m (approx.)'}</td></tr>`).join('')}</tbody></table></div><p class="small muted">Distance excludes gaps and links above 12 m/s. It covers observed contiguous segments, not the whole rally.</p>`;
  if(tab==='methodology')$('#detail').innerHTML=`<h2>Field notes / How this works</h2><div class="insight-grid"><div><h3>01 — Pixels to court meters</h3><p>Four known singles corners define a plane homography: p′ ∝ Hp. A perfect four-point fit does not measure calibration accuracy.</p><h3>02 — Detection to tracking</h3><p>SSDLite finds people. A color-and-motion baseline proposes tiny moving ball-like blobs; a short motion gate associates them. False positives and misses remain possible.</p></div><div><h3>03 — Measurement to inference</h3><p>Bounces are temporal candidates, never proof. Manual contact review is preserved alongside the original prediction. Unvalidated geometry leaves all line calls uncertain.</p><h3>04 — Local by default</h3><p>Private uploads go only to the local Python service. The hosted demo has no upload service. No paid inference or training on uploads.</p></div></div><details><summary>Provenance & limitations</summary><pre>${esc(JSON.stringify(result.provenance,null,2))}</pre><ul>${result.warnings.map(w=>`<li>${esc(w)}</li>`).join('')}</ul></details><a href="https://github.com/massibrevo/baseline-lab/tree/main/docs" target="_blank" rel="noreferrer">Read the engineering notebook ↗</a>`;
}
function modal(html:string){$('#dialog-content').innerHTML=html;$<HTMLDialogElement>('#dialog').showModal();}
function showUpload(){
  if(!local){modal('<div class="eyebrow">PRIVATE PROCESSING</div><h2>Bring your own rally.</h2><p>This hosted demonstration has no Python inference service. Private uploads are available in the local app.</p><div class="code">uv sync --frozen<br>uv run baseline-lab setup-models<br>npm ci --prefix apps/web<br>npm run build --prefix apps/web<br>uv run baseline-lab serve<br><br># In another terminal:<br>uv run baseline-lab worker</div><p>Open <a href="http://127.0.0.1:8000">127.0.0.1:8000</a>. See the GitHub README for full prerequisites.</p>');return;}
  modal(`<div class="eyebrow">NEW STUDY</div><h2>Bring your own rally.</h2><p>Singles · fixed elevated baseline camera · visible court lines · 5–30 seconds.</p><label class="dropzone">＋<b>Choose an MP4 or MOV</b><span>Up to 100 MB · up to 1920 × 1080</span><input id="file" type="file" accept=".mp4,.mov,video/mp4,video/quicktime"></label><p class="small muted">Your clip remains on this machine. No training or external inference requests.</p><div id="upload-state" role="status"></div>`);
  $<HTMLInputElement>('#file').onchange=async e=>{const file=(e.target as HTMLInputElement).files?.[0];if(!file)return;$('#upload-state').textContent='Uploading and validating presentation timestamps…';try{const body=new FormData();body.append('file',file);const s=await api('/sessions',{method:'POST',body});sid=s.id;title=s.title;sessionMeta=s.metadata;demo=false;calibrationDialog();}catch(err){$('#upload-state').textContent=String(err);}};
}
function calibrationDialog(){
  corners=[];calibrationMode=true;
  modal(`<div class="eyebrow">RECORDING CHECK</div><h2>Give the court its coordinates.</h2><p>${sessionMeta.width} × ${sessionMeta.height} · ${sessionMeta.codec} · ${sessionMeta.duration_s.toFixed(2)} s · ${sessionMeta.variable_frame_rate?'variable frame rate':sessionMeta.nominal_fps+' fps'}</p><p id="corner-instruction">Optional: select far-left singles outside corner (1 of 4).</p><div class="calibration-view"><img id="cal-image" src="/api/v1/sessions/${sid}/thumbnail" alt="First frame for manual court calibration"><canvas id="cal-canvas"></canvas></div><div class="cal-legend">1 Far left → 2 Far right → 3 Near right → 4 Near left</div><button class="button" id="reset-cal">Reset corners</button><label class="suitability"><input id="suitable" type="checkbox"> I checked: one continuous singles rally, fixed camera, visible court, adequate lighting.</label><label class="preset-label">Analysis preset <select aria-label="Analysis preset"><option>CPU baseline · SSDlite + color/motion</option></select></label><button class="button primary full" id="analyze">Analyze clip</button><p class="small muted">Skip corners for image-space tracking only. Ground measurements remain unavailable. Automatic calibration is not validated in this release.</p><div id="cal-error" role="status"></div>`);
  const image=$<HTMLImageElement>('#cal-image'),canvas=$<HTMLCanvasElement>('#cal-canvas');
  const drawCal=()=>{canvas.width=sessionMeta.width;canvas.height=sessionMeta.height;const c=canvas.getContext('2d')!;c.strokeStyle='#D8EF45';c.lineWidth=3;c.fillStyle='#D8EF45';c.font='24px monospace';c.beginPath();corners.forEach((p,i)=>{if(i)c.lineTo(p[0],p[1]);else c.moveTo(p[0],p[1]);c.fillText(String(i+1),p[0]+8,p[1]-8);});if(corners.length===4)c.closePath();c.stroke();for(const p of corners){c.beginPath();c.arc(p[0],p[1],5,0,2*Math.PI);c.fill();}};
  image.onload=drawCal;canvas.onclick=e=>{if(corners.length>=4)return;const r=canvas.getBoundingClientRect();corners.push([(e.clientX-r.left)/r.width*sessionMeta.width,(e.clientY-r.top)/r.height*sessionMeta.height]);drawCal();$('#corner-instruction').textContent=corners.length<4?`Select ${['far-left','far-right','near-right','near-left'][corners.length]} singles outside corner (${corners.length+1} of 4).`:'Court outline selected. Check the correspondence order before analysis.';};
  $('#reset-cal').onclick=()=>{corners=[];drawCal();$('#corner-instruction').textContent='Select far-left singles outside corner (1 of 4).';};
  $('#analyze').onclick=async()=>{if(!$<HTMLInputElement>('#suitable').checked){$('#cal-error').textContent='Confirm recording suitability before starting.';return;}if(corners.length!==0&&corners.length!==4){$('#cal-error').textContent='Select all four corners, or reset to skip calibration.';return;}try{const j=await api(`/sessions/${sid}/jobs`,{method:'POST',body:JSON.stringify({preset:'cpu-baseline',calibration:corners.length?{corners_px:corners}:null,idempotency_key:crypto.randomUUID()})});jid=j.id;calibrationMode=false;$<HTMLDialogElement>('#dialog').close();processing();}catch(e){$('#cal-error').textContent=String(e);}};
}
function processing(){
  modal('<div class="eyebrow">LOCAL WORKER</div><h2>Reading the rally.</h2><p id="stage">Queued for analysis…</p><progress id="job-progress"></progress><p id="job-detail" class="small muted"></p><button id="cancel-job" class="button">Cancel analysis</button>');
  $('#cancel-job').onclick=async()=>{await api(`/sessions/${sid}/jobs/${jid}/cancel`,{method:'POST'});notice('Cancellation requested; worker will clean temporary artifacts.');};
  const poll=async()=>{try{const j=await api(`/sessions/${sid}/jobs/${jid}`);$('#stage').textContent=j.stage.replaceAll('_',' ');const p=$<HTMLProgressElement>('#job-progress');if(j.total){p.max=j.total;p.value=j.done||0;}else p.removeAttribute('value');$('#job-detail').textContent=j.total?`${j.done} / ${j.total} · ${j.worker_alive?'Worker active':'Worker heartbeat missing'}`:'Waiting for stage work count…';if(j.state.startsWith('completed')){const r=await api(`/sessions/${sid}/jobs/${jid}/artifacts/result.json`);$<HTMLDialogElement>('#dialog').close();installResult(r,artifact('review.mp4'));notice('Analysis complete with warnings. Review coverage and candidate events.');return;}if(j.state==='failed'||j.state==='cancelled'){$('#stage').textContent=j.state==='failed'?'Analysis could not complete':'Analysis cancelled';$('#job-detail').textContent=j.error||'No result artifacts were published.';$('#cancel-job').classList.add('hidden');return;}pollTimer=window.setTimeout(poll,1000);}catch(e){notice(String(e),true);}};void poll();
}
async function showSessions(){
  if(!local){modal('<h2>Your sessions</h2><p>Private session storage is available in the local app. This public demo keeps only your demo event edits in this browser.</p><button class="button" id="reset-demo">Reset demo corrections</button>');$('#reset-demo').onclick=()=>{localStorage.removeItem('baseline-demo-events');location.reload();};return;}
  try{sessions=await api('/sessions');modal(`<h2>Your local sessions</h2><p>Private to this browser identity. Clearing cookies loses access; source files remain until removed.</p><div id="session-list">${sessions.map(s=>`<button class="session-row" data-session="${s.id}"><b>${esc(s.title)}</b><span>${s.metadata.duration_s.toFixed(2)} s ↗</span></button>`).join('')||'<p>No sessions yet. Upload your first rally.</p>'}</div>`);document.querySelectorAll<HTMLButtonElement>('[data-session]').forEach(b=>b.onclick=async()=>{const s=await api('/sessions/'+b.dataset.session);sid=s.id;title=s.title;sessionMeta=s.metadata;demo=false;const j=s.jobs.find((j:any)=>j.state.startsWith('completed'));if(j){jid=j.id;const r=await api(`/sessions/${sid}/jobs/${jid}/artifacts/result.json`);$<HTMLDialogElement>('#dialog').close();installResult(r,artifact('review.mp4'));}else if(s.jobs.some((j:any)=>['running','queued'].includes(j.state))){jid=s.jobs.find((j:any)=>['running','queued'].includes(j.state)).id;processing();}else calibrationDialog();});}catch(e){notice(String(e),true);}
}
function download(name:string,data:string,type:string){const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([data],{type}));a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),5000);}
function showExports(){
  if(!result)return;
  modal(`<div class="eyebrow">TAKE THE EVIDENCE WITH YOU</div><h2>Export this study.</h2><p>Versioned data includes missing observations, provenance and review corrections. Video overlays are the immutable analysis snapshot; later event edits are in JSON/CSV.</p><div class="export-list"><button class="button" id="json-export">↓ Results JSON · schema v1.1</button><button class="button" id="events-export">↓ Events CSV · current corrections</button><button class="button" id="tracks-export">↓ Tracks CSV · pixels & meters</button><button class="button" id="live-export">↓ Live metrics CSV · timestamped units</button><a class="button" href="${demo?'/demo/annotated.mp4':artifact('annotated.mp4')}" download="baseline-annotated.mp4">↓ Annotated video · H.264 MP4</a></div><p class="small muted">Video exports omit audio. Unavailable values remain null. Example video adaptations retain their source media license.</p>`);
  $('#json-export').onclick=()=>download('baseline-result.json',JSON.stringify(result,null,2),'application/json');
  $('#events-export').onclick=()=>{const rows=[['id','time_s','status','x_px','y_px','classification','notes'],...result!.events.map(e=>[e.id,e.t_s,e.status,...e.xy_px,e.classification,e.notes])];download('baseline-events.csv',csv(rows),'text/csv');};
  $('#live-export').onclick=()=>{const values=result!.live_metrics||[];const fields=values.length?Object.keys(values[0]):[];download('baseline-live-metrics.csv',csv([fields,...values.map(v=>Object.values(v))]),'text/csv');};
  $('#tracks-export').onclick=()=>{const rows:any[][]=[['track','frame','time_s','state','x_px','y_px','court_x_m','court_y_m','score']];for(const t of result!.tracks)for(const p of t.observations)rows.push([t.id,p.frame,p.t_s,p.state,...(p.xy_px??['','']),...(p.court_m??['','']),p.score??'']);download('baseline-tracks.csv',csv(rows),'text/csv');};
}
function csv(rows:unknown[][]){return rows.map(r=>r.map(v=>{let s=String(v??'');if(typeof v==='string'&&/^[=+@\-]/.test(s))s="'"+s;return '"'+s.replaceAll('"','""')+'"';}).join(',')).join('\n');}
function deleteSession(){
  if(demo){modal('<h2>Reset demo edits?</h2><p>This removes your browser-local event corrections. The bundled public footage remains available.</p><button id="confirm-delete" class="button danger">Reset demo corrections</button>');$('#confirm-delete').onclick=()=>{localStorage.removeItem('baseline-demo-events');location.reload();};return;}
  modal('<h2>Delete this private session?</h2><p>The clip and generated artifacts will be removed. Running work is cancelled; active files are removed when the worker releases them.</p><button id="confirm-delete" class="button danger">Delete session and artifacts</button>');$('#confirm-delete').onclick=async()=>{try{await api('/sessions/'+sid,{method:'DELETE'});if(pollTimer)clearTimeout(pollTimer);location.reload();}catch(e){notice(String(e),true);}};
}
async function start(){
  shell();
  try{const r=await fetch('/api/v1/identity');if(r.ok&&(r.headers.get('content-type')||'').includes('application/json')){const h=await r.json();local=h.deployment==='local';$('#connection').textContent=h.worker_alive?'Local worker ready':'Local API · worker offline';}}catch{/* Static demonstration has no API by design. */}
  if(!local)$('#connection').textContent='Hosted demonstration';
  try{const r=await fetch('/demo/result.json');if(!r.ok)throw Error('Example analysis not installed. Run scripts/prepare-demo.py.');const data=await r.json();const edits=localStorage.getItem('baseline-demo-events');if(edits){try{data.events=JSON.parse(edits);}catch{localStorage.removeItem('baseline-demo-events');}}installResult(data,'/demo/clip.mp4');}catch(e){notice(String(e),true);$('#video-meta').textContent='Example not available';}
}
void start();
