import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import ts from 'typescript';
const code=ts.transpileModule(readFileSync(new URL('../src/trail.ts',import.meta.url),'utf8'),{compilerOptions:{module:ts.ModuleKind.ESNext,target:ts.ScriptTarget.ES2022}}).outputText;
const {legacyTrail,trailSegments,trailSvg,smoothCurves,TRAIL_COLOR}=await import('data:text/javascript;base64,'+Buffer.from(code).toString('base64'));
const observations=[{t_s:0,frame:0,state:'detected',xy_px:[10,10]},{t_s:.5,frame:1,state:'missing',xy_px:null},{t_s:1,frame:2,state:'missing',xy_px:null},{t_s:2,frame:3,state:'detected',xy_px:[100,80]}];
test('continuous fill spans every gap without altering measurements',()=>{
 const before=structuredClone(observations);const visual=legacyTrail(observations,640,480);
 assert.ok(visual.every(p=>p.xy_px));assert.equal(trailSegments(visual,2,5,'continuous').length,3);
 assert.deepEqual(observations,before);const svg=trailSvg(visual,2,5,'continuous');assert.ok(svg.includes(TRAIL_COLOR));assert.ok(svg.includes('Q'));assert.ok(!svg.includes('#f5ba70'));
});
test('measured-only mode never bridges inferred gaps or skips future context',()=>{
 const visual=legacyTrail(observations,640,480);
 assert.equal(trailSegments(visual,2,5,'observed').length,0);
 assert.equal(trailSegments(visual,.5,5,'continuous').length,1);
 assert.equal(trailSegments(visual,2,.25,'continuous').length,1);
});
test('an empty clip stays unavailable and damped tails remain in frame',()=>{
 assert.equal(legacyTrail([{t_s:0,frame:0,state:'missing',xy_px:null}],640,480)[0].xy_px,null);
 const visual=legacyTrail([{t_s:0,frame:0,state:'detected',xy_px:[610,20]},{t_s:.03,frame:1,state:'detected',xy_px:[630,30]},{t_s:10,frame:2,state:'missing',xy_px:null}],640,480);
 assert.equal(visual.at(-1).state,'inferred');assert.equal(visual.at(-1).xy_px[0],639);
});

test('point mode contains observations only and no connecting path',()=>{const visual=legacyTrail(observations,640,480);const svg=trailSvg(visual,2,5,'points');assert.equal((svg.match(/<circle/g)||[]).length,2);assert.ok(!svg.includes('<path'));});

test('smooth curve keeps endpoints and joins tangentially without touching source points',()=>{
 const run=legacyTrail(observations,640,480);const before=structuredClone(run);const curves=smoothCurves(run);
 assert.deepEqual(curves[0].start,run[0].xy_px);assert.deepEqual(curves.at(-1).end,run.at(-1).xy_px);
 for(let i=1;i<curves.length;i++){const a=curves[i-1],b=curves[i];assert.deepEqual(a.end,b.start);for(let d=0;d<2;d++)assert.ok(Math.abs((a.end[d]-a.control[d])-(b.control[d]-b.start[d]))<1e-9);}
 assert.deepEqual(run,before);
});
