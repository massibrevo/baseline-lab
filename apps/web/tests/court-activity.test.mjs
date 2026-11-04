import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import ts from 'typescript';
const code=ts.transpileModule(readFileSync(new URL('../src/court-activity.ts',import.meta.url),'utf8'),{compilerOptions:{module:ts.ModuleKind.ESNext,target:ts.ScriptTarget.ES2022}}).outputText;
const {courtActivity,courtActivityView}=await import('data:text/javascript;base64,'+Buffer.from(code).toString('base64'));
const observation=(t,frame,point=[4,20],state='detected')=>({t_s:t,frame,court_m:point,state});
function fixture(times){
 return {metadata:{timestamps_s:times},calibration:{},frame_calibrations:[],tracks:['near','far'].map(id=>({id,observations:times.map((t,i)=>observation(t,i,id==='near'?[4,20]:[4,3]))})),live_metrics:[],automatic_shots:[],shots:[],events:[]};
}
test('occupancy measures real time, independent of variable sampling rate',()=>{
 const a=fixture([0,.02,.09,.1,.2]),b=fixture([0,.1,.2]);
 for(const r of [a,b]){const p=courtActivity(r,r.metadata.timestamps_s.length-1,.2,'automatic').players[0];assert.ok(Math.abs(p.supportedSeconds-.2)<1e-9);assert.ok(Math.abs(p.cells.get('4,20')-.2)<1e-9);}
});
test('missing, interpolated, stale geometry and impossible jumps never contribute',()=>{
 const r=fixture([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.2]);
 const points=r.tracks[0].observations;points[2].state='missing';points[3].state='interpolated';points[7].court_m=[8,20];
 r.frame_calibrations=r.metadata.timestamps_s.map(()=>({}));r.frame_calibrations[5]=null;
 const p=courtActivity(r,10,1.2,'automatic').players[0];
 assert.ok(Math.abs(p.supportedSeconds-.2)<1e-9); // only 0→.1 and .8→.9
 assert.equal(p.links.length,2);assert.ok(p.supportFraction<.17);
 const unavailable=courtActivity(r,5,.5,'automatic');assert.equal(unavailable.mapping,false);assert.equal(unavailable.players[0].point,null);
});
test('seeking backwards removes future occupancy, strokes and bounces without mutating observations',()=>{
 const r=fixture([0,.1,.2]);r.events=[{t_s:.2,frame:2,status:'confirmed',court_m:[4,2]},{t_s:.1,frame:1,status:'candidate',court_m:[4,2]}];
 r.automatic_shots=[{t_s:.2,frame:2,player:'near',kind:'forehand'}];const before=structuredClone(r);
 assert.equal(courtActivity(r,2,.2,'automatic').marks.length,1);
 const state=courtActivity(r,1,.1,'automatic');assert.equal(state.marks.length,0);assert.equal(state.bounces.length,0);assert.ok(Math.abs(state.players[0].supportedSeconds-.1)<1e-9);assert.deepEqual(r,before);
});
test('stroke counts keep automatic candidates, human reviews and boundary swings separate',()=>{
 const r=fixture([0,.1,.2]);r.automatic_shots=[{t_s:.1,frame:1,player:'near',kind:'forehand'},{t_s:.2,frame:2,player:'near',kind:'backhand',recognition_basis:'clip_boundary_swing_not_verified_contact'}];
 r.shots=[{t_s:.1,frame:1,player:'far',kind:'serve',status:'reviewed'}];
 const auto=courtActivity(r,2,.2,'automatic'),review=courtActivity(r,2,.2,'reviewed');assert.equal(auto.players[0].shots,1);assert.equal(auto.players[0].boundarySwings,1);assert.equal(auto.marks.length,1);assert.equal(review.players[1].shots,1);assert.equal(review.players[0].shots,0);
});
test('court view uses footpoints instead of airborne ball and provides seekable markers',()=>{
 const r=fixture([0,.1,.2]);r.tracks.push({id:'ball',observations:[{court_m:[999,999]}]});r.automatic_shots=[{t_s:.1,frame:1,player:'near',kind:'forehand'}];
 const v=courtActivityView(r,2,.2,'strokes','near','automatic');assert.match(v.svg,/data-court-seek="0.1"/);assert.match(v.svg,/role="button" tabindex="0"/);assert.doesNotMatch(v.svg,/999/);assert.match(v.summaries,/1 impact candidates/);assert.doesNotMatch(v.summaries,/FAR/);
});
