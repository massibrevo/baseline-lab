import type {Analysis,ShotSignal} from './contracts';
export type StrokeSource='automatic'|'reviewed';
export function activeStroke(result:Analysis,now:number,source:StrokeSource):ShotSignal|undefined{
 const shots=source==='automatic'?(result.automatic_shots||[]):(result.shots||[]).filter(s=>s.status==='reviewed');
 if(source==='automatic'){
  const active=shots.filter(s=>s.context_start_s!=null&&s.context_end_s!=null&&s.context_start_s<=now&&s.context_end_s>=now).sort((a,b)=>Math.abs(a.t_s-now)-Math.abs(b.t_s-now));
  if(active.length)return active[0];
 }
 return shots.filter(s=>s.t_s<=now).sort((a,b)=>b.t_s-a.t_s)[0];
}
export function strokeTitle(shot?:ShotSignal){return !shot?'Waiting for swing':shot.kind==='unknown'?'Uncertain stroke':shot.kind.replaceAll('_',' ');}
export function strokeEvidence(shot:ShotSignal,now:number,source:StrokeSource){
 if(source==='reviewed')return `Human reviewed · ${shot.player} player · ${shot.t_s.toFixed(2)} s`;
 const age=now-(shot.context_end_s??shot.t_s);
 return `${shot.recognition_basis?.startsWith('clip_boundary')?'Boundary swing · contact unverified':'Automatic · candidate impact'} · ${shot.player} · ${shot.t_s.toFixed(2)} s${age>0?' · last swing '+age.toFixed(1)+' s ago':''}`;
}
