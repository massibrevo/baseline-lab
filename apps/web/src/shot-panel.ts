import {displayTrail,legacyTrail,trailSvg,trailHeadSvg,TRAIL_COLOR,type TrailMode} from './trail';
import {activeStroke,strokeTitle,strokeEvidence,type StrokeSource} from './strokes';
import type {Analysis, Observation} from './contracts';
const safe=(s:unknown)=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]!));

function courtDrawing(calibration:any){
 if(!calibration?.corners_px)return '';
 let drawing=`<polygon points="${calibration.corners_px.map((p:number[])=>p.join(',')).join(' ')}" fill="#254b41" stroke="#d9e2cf" stroke-width="5"/>`;
 const h=calibration.homography as number[][];if(!h)return drawing;
 const [[a,b,c],[d,e,f],[g,j,k]]=h;
 const inverse=[[e*k-f*j,c*j-b*k,b*f-c*e],[f*g-d*k,a*k-c*g,c*d-a*f],[d*j-e*g,b*g-a*j,a*e-b*d]];
 const point=(x:number,y:number)=>{const z=inverse[2][0]*x+inverse[2][1]*y+inverse[2][2];return [(inverse[0][0]*x+inverse[0][1]*y+inverse[0][2])/z,(inverse[1][0]*x+inverse[1][1]*y+inverse[1][2])/z];};
 for(const [x1,y1,x2,y2] of [[0,5.485,8.23,5.485],[0,18.285,8.23,18.285],[4.115,5.485,4.115,18.285],[0,11.885,8.23,11.885]]){const a=point(x1,y1),b=point(x2,y2);drawing+=`<line x1="${a[0]}" y1="${a[1]}" x2="${b[0]}" y2="${b[1]}" stroke="#d9e2cf" stroke-width="4"/>`;}
 return drawing;
}
export function shotPanel(result:Analysis,frame:number,mode:TrailMode='continuous',seconds=2,playhead?:number,source:StrokeSource='automatic'){
 const now=playhead??result.metadata.timestamps_s[frame];
 const shot=activeStroke(result,now,source);
 const ball=result.tracks.find(t=>t.id==='ball')!.observations;
 const sample=result.live_metrics[frame];
 const calibration=result.frame_calibrations?.length?result.frame_calibrations[frame]:result.calibration;
 const w=result.metadata.width,h=result.metadata.height;
 const visual=displayTrail(result);const path=trailSvg(visual,now,seconds,mode);
 const p=mode==='continuous'?visual[frame]:ball[frame],observed=p?.xy_px;
 const outline=courtDrawing(calibration);
 const markers=result.events.filter(e=>e.t_s<=now&&['manual','confirmed'].includes(e.status)).map(e=>`<circle cx="${e.xy_px[0]}" cy="${e.xy_px[1]}" r="15" fill="${TRAIL_COLOR}" opacity=".65"/>`).join('');
 const title=strokeTitle(shot);
 const speed=latestSpeed(result,frame);
 const measured=[...result.events].filter(e=>e.t_s<=now&&e.shot_evidence?.speed_kmh!=null&&['manual','confirmed'].includes(e.status)).sort((a,b)=>b.t_s-a.t_s)[0];
 const age=speed?Math.max(0,now-(speed.support_end_s??speed.t_s)):0;
 return `<div class="section-label"><span>LIVE SHOT VIEW</span><small>${now.toFixed(2)} S</small></div><svg viewBox="0 0 ${w} ${h}" role="img" aria-label="Live ball trail in video coordinates" class="shot-radar">${outline}${path}${markers}${observed?trailHeadSvg(observed):''}</svg><div class="shot-readout"><div><b>${safe(title)}</b><small>${shot?safe(strokeEvidence(shot,now,source)):'Waiting for motion evidence'}</small>${shot?`<button class="shot-seek" data-review-shot="${shot.id}">Review stroke type ↗</button>`:''}</div><div><strong>${measured?measured.shot_evidence!.speed_kmh!.toFixed(0):speed?'~'+Math.round(speed.estimated_kmh!):'—'}</strong><small>km/h · ${measured?'entered measurement':speed?'experimental estimate':'insufficient flight evidence'}</small>${speed&&!measured?`<small>${Math.round(speed.low_kmh!)}–${Math.round(speed.high_kmh!)} sensitivity${age>.15?' · last flight '+age.toFixed(1)+' s ago':''}</small>`:''}</div></div><div class="stroke-jumps">${(source==='automatic'?(result.automatic_shots||[]):(result.shots||[]).filter(s=>s.status==='reviewed')).map(s=>`<button data-seek-shot="${s.t_s}">${s.t_s.toFixed(2)} s · ${safe(s.kind.replaceAll('_',' '))}</button>`).join('')}</div><p class="shot-note">${speed?'3D flight fit with camera assumptions. Range is a sensitivity allowance; radar accuracy is unvalidated.':'Calibrate the court and confirm normal 1× action timing for experimental km/h.'} Smooth cyan trail includes visual gap fill. Automatic labels use video pixels and temporal context; uncertain labels need review.</p>`;
}
export function latestSpeed(result:Analysis,frame:number){
 const now=result.metadata.timestamps_s[frame];
 const samples=result.ball_speed_estimates||[];
 if(samples[frame]?.estimated_kmh!=null)return samples[frame];
 // Hold the last supported flight as a historical shot readout, never a fresh measurement.
 for(let i=Math.min(frame,samples.length-1);i>=0;i--)if(samples[i].estimated_kmh!=null&&samples[i].t_s<=now)return samples[i];
 return undefined;
}

export function cameraShotCard(sample:any,width:number,height:number){
 const ball=sample.tracks.find((t:any)=>t.id==='ball')?.observation as Observation|undefined;
 const shot=activeStroke(sample as Analysis,sample.metrics.t_s,'automatic');
 const visual=sample.display_ball_trail?.length?sample.display_ball_trail:legacyTrail(sample.ball_trail||[],width,height);
 const p=visual.at(-1),point=p?.xy_px??ball?.xy_px;
 const trace=trailSvg(visual,sample.metrics.t_s,2,'continuous');
 return `<div class="section-label"><span>LIVE SHOT VIEW</span><small>ANALYZED FRAME</small></div><svg class="shot-radar" viewBox="0 0 ${width} ${height}" role="img" aria-label="Live camera ball location in image coordinates">${courtDrawing(sample.calibration)}${trace}${point?trailHeadSvg(point):''}</svg><div class="shot-readout"><div><b>${safe(strokeTitle(shot))}</b><small>${shot?safe(strokeEvidence(shot,sample.metrics.t_s,'automatic')):safe(sample.stroke_analysis?.reason||'Collecting motion context')}</small></div><div><strong>${sample.ball_speed?.estimated_kmh==null?'—':'~'+Math.round(sample.ball_speed.estimated_kmh)}</strong><small>km/h · experimental fit</small></div></div>`;
}
