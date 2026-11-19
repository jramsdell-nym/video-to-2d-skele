import {loadPreparedModel,predictPose,predictBat} from '/Users/jramsdell/Documents/bitbucket/react_apps/src/bat-data-labeler/services/pose-model.js';
import {saveModel,getModel} from '/Users/jramsdell/Documents/bitbucket/react_apps/src/bat-data-labeler/services/pose-store.js';
import {gpuSession} from '/Users/jramsdell/Documents/bitbucket/react_apps/src/bat-data-labeler/services/pose-gpu.js';
import {decodePose,decodeBat,BONES,BODY_INDICES} from '/Users/jramsdell/Documents/bitbucket/react_apps/src/bat-data-labeler/utils/pose-data.js';
let ort;
import wasmUrl from '/Users/jramsdell/Documents/bitbucket/react_apps/src/bat-data-labeler/node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm?url';
import mjsUrl from '/Users/jramsdell/Documents/bitbucket/react_apps/src/bat-data-labeler/node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.mjs?url';

const status=document.querySelector('#status'),button=document.querySelector('#run');
const log=s=>status.textContent+='\n'+s;
const assert=(ok,msg)=>{if(!ok)throw Error(msg);};
button.onclick=async()=>{button.disabled=true;status.textContent='Running…';
try{
 ort=await import('onnxruntime-web/wasm');
 ort.env.wasm.numThreads=1;ort.env.wasm.proxy=true;ort.env.wasm.wasmPaths={wasm:new URL(wasmUrl,location.href).href,mjs:new URL(mjsUrl,location.href).href};
 const image=new Image();image.src='/sample.png';await image.decode();
 const canvas=document.querySelector('#canvas');canvas.width=image.width;canvas.height=image.height;const ctx=canvas.getContext('2d');ctx.drawImage(image,0,0);
 for(const kind of ['body','bat']){
  // This test runs on an isolated localhost origin, never the user's annotation origin.
  await saveModel({hash:'previous-release',name:'previous-release',blob:new Blob(),importedAt:new Date().toISOString()},kind);
  const model=await loadPreparedModel(kind);assert((await getModel(kind)).hash===model.hash,'Cache was not replaced');
  const meta=await (await fetch('/'+kind+'-golden.json')).json();
  const input=new Float32Array(await(await fetch('/'+kind+'-input.f32')).arrayBuffer());
  const expected=new Float32Array(await(await fetch('/'+kind+'-output.f32')).arrayBuffer());
  const bytes=new Uint8Array(await model.blob.arrayBuffer());
  const wasm=await ort.InferenceSession.create(bytes,{executionProviders:['wasm']});
  const gpu=await gpuSession({...model,hash:model.hash+'-parity'},kind,()=>{throw Error('WebGPU test unexpectedly fell back');});
  for(const [provider,session] of [['wasm',wasm],['webgpu',gpu]]){
   const tensor=new ort.Tensor('float32',new Float32Array(input),meta.inputShape);const output=await session.run({[session.inputNames[0]]:tensor});const result=output[session.outputNames[0]];
   const decoder=kind==='body'?decodePose:decodeBat;
   const original=decoder(expected,meta.outputShape,meta.transform),actual=decoder(result.data,result.dims,meta.transform);
   const a=kind==='body'?BODY_INDICES.map(i=>original[0].points[i]):[original.tip,original.knob];
   const b=kind==='body'?BODY_INDICES.map(i=>actual[0].points[i]):[actual.tip,actual.knob];
   let error=0;for(let i=0;i<a.length;i++){assert(!!a[i]===!!b[i],'Detection changed');if(a[i]?.x!=null&&b[i]?.x!=null)error=Math.max(error,Math.abs(a[i].x-b[i].x)*meta.transform.width,Math.abs(a[i].y-b[i].y)*meta.transform.height);}
   assert(error<.5,provider+' coordinate mismatch '+error);log(kind+' '+provider+' parity PASS: '+error.toFixed(5)+' px');
   tensor.dispose();Object.values(output).forEach(o=>o.dispose?.());await session.release();
  }
  const prediction=await(kind==='body'?predictPose:predictBat)(image,'release-source-image',model);
  assert(prediction.model.sha256===model.hash,'Incorrect model identity');
  if(kind==='body'){
   assert(prediction.people.length>0,'No body prediction');const p=prediction.people[0].points;
   ctx.strokeStyle='#5de1ff';ctx.lineWidth=4;for(const [a,b]of BONES){if(p[a].x==null||p[b].x==null)continue;ctx.beginPath();ctx.moveTo(p[a].x*canvas.width,p[a].y*canvas.height);ctx.lineTo(p[b].x*canvas.width,p[b].y*canvas.height);ctx.stroke();}
  }else{
   assert(prediction.tip&&prediction.knob,'No bat endpoints');ctx.strokeStyle='#ff3344';ctx.lineWidth=6;ctx.beginPath();ctx.moveTo(prediction.tip.x*canvas.width,prediction.tip.y*canvas.height);ctx.lineTo(prediction.knob.x*canvas.width,prediction.knob.y*canvas.height);ctx.stroke();
  }
  log(kind+' app image preprocessing/inference PASS ('+prediction.executionProvider+') '+model.hash);
 }
 log('PASS — both pinned models, cache upgrades, browser WASM/WebGPU and source-image predictions verified.');
}catch(e){log('FAIL '+e.stack);}finally{button.disabled=false;}};
