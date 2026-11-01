import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import ts from 'typescript';
const code=ts.transpileModule(readFileSync(new URL('../src/strokes.ts',import.meta.url),'utf8'),{compilerOptions:{module:ts.ModuleKind.ESNext,target:ts.ScriptTarget.ES2022}}).outputText;
const {activeStroke,strokeTitle}=await import('data:text/javascript;base64,'+Buffer.from(code).toString('base64'));
test('automatic readout cannot inherit a timestamped human correction',()=>{
 const automatic={id:'auto',t_s:3.2,kind:'forehand',context_start_s:2.5,context_end_s:3.65,status:'estimated'};
 const result={automatic_shots:[automatic],shots:[{id:'human',t_s:2.1,kind:'backhand',status:'reviewed'}]};
 assert.equal(activeStroke(result,2.7,'automatic').kind,'forehand');
 assert.equal(activeStroke(result,2.7,'reviewed').kind,'backhand');
 result.shots[0].kind='serve';result.shots[0].t_s=2.6;
 assert.strictEqual(activeStroke(result,2.7,'automatic'),automatic);
 assert.equal(activeStroke(result,2,'automatic'),undefined);
});
test('uncertain model predictions remain uncertain despite reviewed labels',()=>{
 const result={automatic_shots:[{id:'ambiguous',t_s:6,kind:'unknown',status:'estimated'}],shots:[{id:'human',t_s:6,kind:'forehand',status:'reviewed'}]};
 assert.equal(strokeTitle(activeStroke(result,6,'automatic')),'Uncertain stroke');
 assert.equal(strokeTitle(activeStroke(result,6,'reviewed')),'forehand');
});
