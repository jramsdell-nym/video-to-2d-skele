/* Shared by the labeler and its navigation checks. All frame indices are zero-based. */
(function(root){
 'use strict';
 function swingFrames(range){
  if(!range)return [];
  const {startFrame:start,endFrame:end,sampleCount:count}=range;
  if(count===1)return [Math.round((start+end)/2)];
  return Array.from({length:count},(_,i)=>Math.round(start+(end-start)*i/(count-1)));
 }
 function plan(length,step,range){
  const frames=[];
  for(let i=0;i<length;i+=step)if(!range||i<range.startFrame||i>range.endFrame)frames.push(i);
  return frames.concat(swingFrames(range)).sort((a,b)=>a-b);
 }
 function neighbor(frames,index,direction){
  return direction>0?frames.find(i=>i>index):frames.findLast(i=>i<index);
 }
 function nextPending(frames,index,complete){
  return frames.find(i=>i>index&&!complete(i));
 }
 function interval(frames,index,fallback){
  return (neighbor(frames,index,1)??(index+fallback))-index;
 }
 const api={swingFrames,plan,neighbor,nextPending,interval};
 if(typeof module==='object'&&module.exports)module.exports=api;
 else root.BatSampling=api;
})(globalThis);
