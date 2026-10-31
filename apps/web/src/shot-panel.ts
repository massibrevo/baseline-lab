import type {Analysis, Observation, ShotSignal} from './contracts';
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
export function shotPanel(result:Analysis,frame:number){
 const now=result.metadata.timestamps_s[frame];
 const shot=[...(result.shots||[])].reverse().find(s=>s.t_s<=now);
 const ball=result.tracks.find(t=>t.id==='ball')!.observations;
 const sample=result.live_metrics[frame];
 const calibration=result.frame_calibrations?.length?result.frame_calibrations[frame]:result.calibration;
 const w=result.metadata.width,h=result.metadata.height;
 let path='';const tail=ball.slice(Math.max(0,frame-120),frame+1);
 for(let i=1;i<tail.length;i++){
  const a=tail[i-1],b=tail[i];if(!a.xy_px||!b.xy_px||now-b.t_s>2||b.t_s-a.t_s>.1)continue;
  path+=`<line x1="${a.xy_px[0]}" y1="${a.xy_px[1]}" x2="${b.xy_px[0]}" y2="${b.xy_px[1]}" ${a.state==='interpolated'||b.state==='interpolated'?'stroke-dasharray="12 10"':''}/>`;
 }
 const p=ball[frame],observed=p?.xy_px;
 const outline=courtDrawing(calibration);
 const markers=result.events.filter(e=>e.t_s<=now&&['manual','confirmed'].includes(e.status)).map(e=>`<circle cx="${e.xy_px[0]}" cy="${e.xy_px[1]}" r="15" fill="#e1a075"/>`).join('');
 const title=shot?.status==='reviewed'?shot.kind.replaceAll('_',' '):'Needs review';
 const speed=latestSpeed(result,frame);
 const measured=[...result.events].filter(e=>e.t_s<=now&&e.shot_evidence?.speed_kmh!=null&&['manual','confirmed'].includes(e.status)).sort((a,b)=>b.t_s-a.t_s)[0];
 const age=speed?Math.max(0,now-(speed.support_end_s??speed.t_s)):0;
 return `<div class="section-label"><span>LIVE SHOT VIEW</span><small>${now.toFixed(2)} S</small></div><svg viewBox="0 0 ${w} ${h}" role="img" aria-label="Live ball trail in video coordinates" class="shot-radar">${outline}<g stroke="#5ef3eb" stroke-width="9" stroke-linecap="round">${path}</g>${markers}${observed?`<circle cx="${observed[0]}" cy="${observed[1]}" r="20" fill="#d8ef45" stroke="#f3f0e6" stroke-width="6"/>`:''}</svg><div class="shot-readout"><div><b>${safe(title)}</b><small>${shot?(shot.status==='reviewed'?'Reviewed · ':'Candidate · ')+shot.player+' player · '+shot.t_s.toFixed(2)+' s':'Waiting for impact evidence'}</small>${shot?`<button class="shot-seek" data-review-shot="${shot.id}">Review stroke type ↗</button>`:''}</div><div><strong>${measured?measured.shot_evidence!.speed_kmh!.toFixed(0):speed?'~'+Math.round(speed.estimated_kmh!):'—'}</strong><small>km/h · ${measured?'entered measurement':speed?'experimental estimate':'insufficient flight evidence'}</small>${speed&&!measured?`<small>${Math.round(speed.low_kmh!)}–${Math.round(speed.high_kmh!)} sensitivity${age>.15?' · last flight '+age.toFixed(1)+' s ago':''}</small>`:''}</div></div><div class="stroke-jumps">${(result.shots||[]).filter(s=>s.status==='reviewed').map(s=>`<button data-seek-shot="${s.t_s}">${s.t_s.toFixed(2)} s · ${safe(s.kind.replaceAll('_',' '))}</button>`).join('')}</div><p class="shot-note">${speed?'3D flight fit with camera assumptions. Range is a sensitivity allowance; radar accuracy is unvalidated.':'Calibrate the court and confirm normal 1× action timing for experimental km/h.'} Cyan trail follows the detected ball. Stroke types require review.</p>`;
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
 const shot=sample.shots?.at(-1) as ShotSignal|undefined;
 const point=ball?.xy_px;
 return `<div class="section-label"><span>LIVE SHOT VIEW</span><small>ANALYZED FRAME</small></div><svg class="shot-radar" viewBox="0 0 ${width} ${height}" role="img" aria-label="Live camera ball location in image coordinates">${courtDrawing(sample.calibration)}${point?`<circle cx="${point[0]}" cy="${point[1]}" r="18" fill="#d8ef45"/>`:''}</svg><div class="shot-readout"><div><b>${safe(shot?.status==='reviewed'?shot.kind.replaceAll('_',' '):'Needs review')}</b><small>Candidate impact · review required</small></div><div><strong>${sample.ball_speed?.estimated_kmh==null?'—':'~'+Math.round(sample.ball_speed.estimated_kmh)}</strong><small>km/h · experimental fit</small></div></div>`;
}
