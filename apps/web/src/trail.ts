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
export const TRAIL_COLOR='#5ee7ff';
type XY=[number,number];
export type TrailCurve={start:XY;control:XY;end:XY;t_s:number};
const midpoint=(a:XY,b:XY):XY=>[(a[0]+b[0])/2,(a[1]+b[1])/2];
// Quadratic B-spline spans stay within the control polygon and join tangentially.
export function smoothCurves(run:DisplayTrailPoint[]):TrailCurve[]{
 if(run.length<2)return [];
 let start=run[0].xy_px!;const curves:TrailCurve[]=[];
 for(let i=0;i<run.length;i++){
  const p=run[i],next=run[i+1];const end=next?midpoint(p.xy_px!,next.xy_px!):p.xy_px!;
  curves.push({start,control:p.xy_px!,end,t_s:next?(p.t_s+next.t_s)/2:p.t_s});start=end;
 }
 return curves;
}
function visibleRuns(points:DisplayTrailPoint[],now:number,seconds:number,mode:TrailMode){
 const runs:DisplayTrailPoint[][]=[];let previous:DisplayTrailPoint|undefined;
 for(const {a,b} of trailSegments(points,now,seconds,mode)){
  if(previous!==a)runs.push([a]);runs.at(-1)!.push(b);previous=b;
 }
 return runs;
}
export function drawTrail(context:CanvasRenderingContext2D,points:DisplayTrailPoint[],now:number,seconds:number,mode:TrailMode,scale=1){
 context.save();context.setLineDash([]);context.strokeStyle=TRAIL_COLOR;context.fillStyle=TRAIL_COLOR;
 if(mode==='points'){for(const p of points){if(p.t_s>now||now-p.t_s>seconds||p.state!=='observed'||!p.xy_px)continue;context.beginPath();context.arc(...p.xy_px,2*scale,0,Math.PI*2);context.fill();}context.restore();return;}
 const curves=visibleRuns(points,now,seconds,mode).map(smoothCurves);
 context.lineCap='round';context.lineJoin='round';context.globalAlpha=.2;context.lineWidth=6*scale;context.shadowColor=TRAIL_COLOR;context.shadowBlur=8*scale;context.beginPath();
 for(const run of curves){if(!run.length)continue;context.moveTo(...run[0].start);for(const c of run)context.quadraticCurveTo(...c.control,...c.end);}context.stroke();context.shadowBlur=0;
 for(const run of curves)for(const c of run){const age=Math.max(0,Math.min(1,1-(now-c.t_s)/seconds));context.globalAlpha=.18+.82*age;context.lineWidth=(.9+1.9*Math.sqrt(age))*scale;context.beginPath();context.moveTo(...c.start);context.quadraticCurveTo(...c.control,...c.end);context.stroke();}
 context.restore();
}
export function drawBallHead(context:CanvasRenderingContext2D,xy:XY,scale=1){
 context.save();context.fillStyle=TRAIL_COLOR;context.strokeStyle=TRAIL_COLOR;context.shadowColor=TRAIL_COLOR;context.shadowBlur=12*scale;context.globalAlpha=.2;context.beginPath();context.arc(...xy,9*scale,0,Math.PI*2);context.fill();context.globalAlpha=1;context.shadowBlur=0;context.lineWidth=1.3*scale;context.beginPath();context.arc(...xy,5*scale,0,Math.PI*2);context.stroke();context.beginPath();context.arc(...xy,2.5*scale,0,Math.PI*2);context.fill();context.restore();
}
export function trailHeadSvg(xy:XY){return `<g fill="${TRAIL_COLOR}"><circle cx="${xy[0]}" cy="${xy[1]}" r="28" opacity=".17"/><circle cx="${xy[0]}" cy="${xy[1]}" r="12"/></g>`;}
export function trailSvg(points:DisplayTrailPoint[],now:number,seconds:number,mode:TrailMode){
 if(mode==='points')return points.filter(p=>p.t_s<=now&&now-p.t_s<=seconds&&p.state==='observed'&&p.xy_px).map(p=>`<circle cx="${p.xy_px![0]}" cy="${p.xy_px![1]}" r="6" fill="${TRAIL_COLOR}"/>`).join('');
 const runs=visibleRuns(points,now,seconds,mode).map(smoothCurves);let halo='',core='';
 for(const run of runs){if(!run.length)continue;halo+=`M${run[0].start.join(' ')}`;for(const c of run){const q=`Q${c.control.join(' ')} ${c.end.join(' ')}`;halo+=q;const age=Math.max(0,Math.min(1,1-(now-c.t_s)/seconds));core+=`<path d="M${c.start.join(' ')}${q}" opacity="${(.18+.82*age).toFixed(2)}" stroke-width="${3+6*Math.sqrt(age)}"/>`;}}
 return `<g fill="none" stroke="${TRAIL_COLOR}" stroke-linecap="round" stroke-linejoin="round"><path d="${halo}" opacity=".14" stroke-width="22"/><path d="${halo}" opacity=".18" stroke-width="13"/>${core}</g>`;
}
