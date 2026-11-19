const assert = require('node:assert/strict');
const s = require('./sampling.js');

assert.deepEqual(s.plan(16,5,null),[0,5,10,15]);
const range={startFrame:7,endFrame:18,sampleCount:5};
const plan=s.plan(26,5,range);
assert.deepEqual(plan,[0,5,7,10,13,15,18,20,25]);
assert.equal(s.neighbor(plan,5,1),7,'Coarse navigation lands on an off-grid swing start');
assert.equal(s.neighbor(plan,18,1),20,'Outside sampling resumes after the swing');
assert.equal(s.neighbor(plan,7,-1),5);
assert.equal(s.neighbor(plan,12,1),13,'Manual off-grid frame returns to the plan');
assert.equal(s.nextPending(plan,10,i=>[13,15].includes(i)),18,'Reviewed/skipped frames are bypassed');
assert.equal(s.nextPending(plan,25,i=>i!==7),undefined,'Earlier optional samples never prevent leaving a clip');
assert.equal(s.nextPending(plan,25,()=>true),undefined);
assert.deepEqual(s.swingFrames({...range,sampleCount:1}),[13]);
assert.deepEqual(s.swingFrames({startFrame:0,endFrame:0,sampleCount:1}),[0]);
assert.deepEqual(s.plan(5,5,{startFrame:0,endFrame:4,sampleCount:5}),[0,1,2,3,4]);
for(let length=1;length<=121;length++){
 for(let count=1;count<=length;count++){
  const selected=s.swingFrames({startFrame:3,endFrame:length+2,sampleCount:count});
  assert.equal(new Set(selected).size,count);
  assert.ok(selected.every(i=>Number.isInteger(i)&&i>=3&&i<=length+2));
  if(count>1){assert.equal(selected[0],3);assert.equal(selected.at(-1),length+2);}
 }
}
console.log('PASS: exact sample counts, boundary navigation, reviewed skips, optional earlier frames, and all-frame ranges');
