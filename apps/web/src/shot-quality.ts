import type {Event as Contact, ShotEvidence} from './contracts';
const esc=(s:unknown)=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]!));
export function evidenceForm(e?:ShotEvidence|null){return `<details class="shot-inputs"><summary>Optional measured speed, spin & bounce angle</summary><p class="small muted">Use radar, high-speed footage or another measured source. Leave unknown values empty. Video image motion does not supply these values.</p><label>Ball speed (km/h)<input id="shot-speed" type="number" min="0" max="300" step="0.1" value="${e?.speed_kmh??''}"></label><label>Spin magnitude (rpm)<input id="shot-spin" type="number" min="0" max="10000" value="${e?.spin_rpm??''}"></label><label>Outgoing bounce angle above court (°)<input id="shot-angle" type="number" min="0" max="90" step="0.1" value="${e?.bounce_angle_deg??''}"></label><label>Measurement source<input id="shot-source" maxlength="200" placeholder="e.g. radar + high-speed camera" value="${esc(e?.source??'')}"></label></details>`;}
export function readEvidence():ShotEvidence|null{
 const n=(id:string,max:number)=>{const s=(document.querySelector(id) as HTMLInputElement).value.trim();if(!s)return null;const v=Number(s);if(!Number.isFinite(v)||v<0||v>max)throw Error(`Measurement must be between 0 and ${max}.`);return v;};
 const speed_kmh=n('#shot-speed',300),spin_rpm=n('#shot-spin',10000),bounce_angle_deg=n('#shot-angle',90);
 if([speed_kmh,spin_rpm,bounce_angle_deg].every(v=>v==null))return null;
 const source=(document.querySelector('#shot-source') as HTMLInputElement).value.trim();if(source.length<3)throw Error('Name the measurement source for speed, spin or bounce angle.');
 return {speed_kmh,spin_rpm,bounce_angle_deg,source};
}
export function qualityView(event?:Contact|null){
 const q=event?.shot_quality;
 if(!q||q.score==null)return '<div class="quality-empty"><div><h3>Shot Quality</h3><p>Start with a reviewed bounce to score depth and width. Speed, spin and bounce angle need independent measurements.</p></div><button class="button" data-measurement-action="depth">Review bounce ↗</button></div>';
 return `<h3>Shot Quality <strong>${q.score.toFixed(1)} <small>/ 10</small></strong></h3><p class="small muted">${q.status==='complete'?'Complete experimental index':'Partial index · missing factors excluded'} · ${q.available_factors}/5 factors · ${(q.evidence_coverage*100).toFixed(0)}% weighted evidence</p><div class="shot-factors">${Object.entries(q.factors).map(([name,f])=>`<div><span>${esc(name.replace('_',' '))}</span><b>${f.value==null?'unavailable':f.value.toFixed(1)+' '+esc(f.unit)}</b><small>${f.score==null?'—':f.score.toFixed(1)+'/10'} · weight ${(f.weight*100).toFixed(0)}% · ${esc(f.source)}</small></div>`).join('')}</div><p class="small muted">Experimental deep/wide groundstroke pressure profile. It does not decide whether an error was forced. Receiver position, reach and execution still need review.</p>`;
}
// Same versioned formula as analytics/quality.py, only for browser-local demo edits.
export function demoQuality(e:Contact){
 const c=e.court_m,grounded=c&&['manual','confirmed'].includes(e.status),inside=grounded&&c[0]>=0&&c[0]<=8.23&&c[1]>=0&&c[1]<=23.77;
 const depth=grounded?Math.min(c[1],23.77-c[1]):null,width=grounded?Math.abs(c[0]-4.115):null;
 e.shot_depth_m=depth;e.shot_width_m=width;const ev=e.shot_evidence;
 const clamp=(n:number)=>Math.max(0,Math.min(10,n));
 const vals:[string,number|null,string,number,(v:number)=>number][]=[['speed',ev?.speed_kmh??null,'km/h',.3,v=>clamp((v-40)/14)],['spin',ev?.spin_rpm??null,'rpm',.15,v=>clamp(v/400)],['depth',depth,'m inside baseline',.25,v=>inside?clamp(10*(1-v/11.885)):0],['width',width,'m from center',.2,v=>inside?clamp(10*v/4.115):0],['bounce_angle',ev?.bounce_angle_deg??null,'degrees above court',.1,v=>clamp(v/6)]];
 const factors=Object.fromEntries(vals.map(([name,value,unit,weight,normalize])=>[name,{value,unit,weight,score:value!=null&&grounded?normalize(value):null,source:['depth','width'].includes(name)?'reviewed ground contact + manual calibration':ev?.source??'unavailable: independent measurement required'}]));
 const available=Object.values(factors).filter(f=>f.score!=null),coverage=available.reduce((a,f)=>a+f.weight,0);e.shot_quality={score:available.length>=2?available.reduce((a,f)=>a+f.score!*f.weight,0)/coverage:null,available_factors:available.length,total_factors:5,evidence_coverage:coverage,status:available.length===5?'complete':available.length>=2?'partial':'unavailable',factors,method:'groundstroke-pressure-v1; experimental weighted heuristic, not validated skill rating',error_attribution:'undetermined'};
}
