import type {Analysis, DisplayTrailPoint, Observation} from './contracts';
export type TrailMode='continuous'|'observed'|'points';
// Legacy results receive a display-only bridge. Never write into observations.
export function legacyTrail(points:Observation[],width:number,height:number):DisplayTrailPoint[]{
 const anchors=points.filter(p=>p.xy_px&&['detected','interpolated'].includes(p.state));let right=0;
 return points.map(p=>{
  if(p.xy_px&&['detected','interpolated'].includes(p.state))return {t_s:p.t_s,frame:p.frame,xy_px:p.xy_px,state:p.state==='detected'?'observed':'interpolated',reason:null};
  while(right<anchors.length&&anchors[right].t_s<p.t_s)right++;
  let xy:[number,number]|null=null;
  if(right>0&&right<anchors.length){const a=anchors[right-1],b=anchors[right],w=(p.t_s-a.t_s)/(b.t_s-a.t_s);xy=[a.xy_px![0]+(b.xy_px![0]-a.xy_px![0])*w,a.xy_px![1]+(b.xy_px![1]-a.xy_px![1])*w];}
  else if(anchors.length){const a=right?anchors.at(-1)!:anchors[0],b=right?anchors.at(-2):anchors[1];const dt=b?a.t_s-b.t_s:0;const elapsed=p.t_s-a.t_s,damping=Math.sign(elapsed)*.35*(1-Math.exp(-Math.abs(elapsed)/.35));xy=[0,1].map(d=>a.xy_px![d]+(b&&dt&&Math.abs(dt)<=.15?(a.xy_px![d]-b.xy_px![d])/dt:0)*damping) as [number,number];}
  if(xy)xy=[Math.max(0,Math.min(width-1,xy[0])),Math.max(0,Math.min(height-1,xy[1]))];
  return {t_s:p.t_s,frame:p.frame,xy_px:xy,state:xy?'inferred':'unavailable',reason:xy?'legacy_display_fill':'no_ball_anchor'};
 });
}
const cache=new WeakMap<Analysis,DisplayTrailPoint[]>();
export function displayTrail(result:Analysis){if(result.display_ball_trail?.length)return result.display_ball_trail;let points=cache.get(result);if(!points){points=legacyTrail(result.tracks.find(t=>t.id==='ball')!.observations,result.metadata.width,result.metadata.height);cache.set(result,points);}return points;}
export function trailSegments(points:DisplayTrailPoint[],now:number,seconds:number,mode:TrailMode){
 const segments:{a:DisplayTrailPoint;b:DisplayTrailPoint;inferred:boolean;interpolated:boolean}[]=[];
 for(let i=1;i<points.length;i++){const a=points[i-1],b=points[i];if(b.t_s>now+.001)break;if(now-b.t_s>seconds||!a.xy_px||!b.xy_px)continue;const inferred=a.state==='inferred'||b.state==='inferred'||b.t_s-a.t_s>.1;if(mode!=='continuous'&&inferred)continue;segments.push({a,b,inferred,interpolated:a.state==='interpolated'||b.state==='interpolated'});}
 return segments;
}
export function drawTrail(context:CanvasRenderingContext2D,points:DisplayTrailPoint[],now:number,seconds:number,mode:TrailMode,scale=1){
 if(mode==='points'){for(const p of points){if(p.t_s>now||now-p.t_s>seconds||p.state!=='observed'||!p.xy_px)continue;context.fillStyle='#5ef3eb';context.beginPath();context.arc(...p.xy_px,2*scale,0,Math.PI*2);context.fill();}return;}
 context.setLineDash([]);context.lineWidth=3*scale;context.lineCap='round';context.lineJoin='round';
 for(const {a,b,inferred,interpolated} of trailSegments(points,now,seconds,mode)){context.strokeStyle=inferred?'#f5ba70':interpolated?'#a2d7df':'#5ef3eb';context.beginPath();context.moveTo(...a.xy_px!);context.lineTo(...b.xy_px!);context.stroke();}
}
export function trailSvg(points:DisplayTrailPoint[],now:number,seconds:number,mode:TrailMode){
 if(mode==='points')return points.filter(p=>p.t_s<=now&&now-p.t_s<=seconds&&p.state==='observed'&&p.xy_px).map(p=>`<circle cx="${p.xy_px![0]}" cy="${p.xy_px![1]}" r="6" fill="#5ef3eb"/>`).join('');
 const paths={observed:'',interpolated:'',inferred:''};
 for(const s of trailSegments(points,now,seconds,mode)){const key=s.inferred?'inferred':s.interpolated?'interpolated':'observed';paths[key]+=`M${s.a.xy_px!.join(' ')}L${s.b.xy_px!.join(' ')}`;}
 return Object.entries(paths).map(([kind,d])=>`<path d="${d}" fill="none" stroke="${kind==='inferred'?'#f5ba70':kind==='interpolated'?'#a2d7df':'#5ef3eb'}" stroke-width="9" stroke-linecap="round"/>`).join('');
}
