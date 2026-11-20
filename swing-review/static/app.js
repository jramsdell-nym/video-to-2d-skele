const $ = id => document.getElementById(id);
const JOINTS = ['left_shoulder','right_shoulder','left_elbow','right_elbow','left_wrist','right_wrist','left_hip','right_hip','left_knee','right_knee','left_ankle','right_ankle','batTip','batKnob'];
const NAMES = ['Left shoulder','Right shoulder','Left elbow','Right elbow','Left wrist','Right wrist','Left hip','Right hip','Left knee','Right knee','Left ankle','Right ankle','Bat tip · barrel end','Bat shaft · top of hands'];
const BONES = [[0,1],[0,2],[2,4],[1,3],[3,5],[0,6],[1,7],[6,7],[6,8],[8,10],[7,9],[9,11],[12,13]];
const actor = {type:'human', id:'local-reviewer'};
let trial=null, index=0, joint='left_elbow', pass='placement', busy=false, playing=false, recordReplay=false, drawEpoch=0, currentImage=null, pointer=null, bootstrap=null;
let pinnedReference=null;
const images = new Map();
const sleep = ms => new Promise(resolve => setTimeout(resolve,ms));
const point = (r,j) => j.startsWith('bat') ? r[j] : r.points[JOINTS.indexOf(j)+5];
const located = p => p && Number.isFinite(p.x) && Number.isFinite(p.y) && p.visibility>0;
const color = j => j.startsWith('bat') ? '#ffcc71' : j.startsWith('left') ? '#74d3ff' : '#ff91cd';
const name = j => NAMES[JOINTS.indexOf(j)] || j;
const frame = () => trial?.frames[index];
const label = row => (row.trial||row.id).replace('markers_','').replaceAll('_',' · ');
const textNode = (tag, value, cls) => {const n=document.createElement(tag);n.textContent=value;if(cls)n.className=cls;return n;};
const proposals = (idx=null) => (trial?.agentReports||[]).flatMap(r=>(r.changes||[]).filter(c=>c.decision==='pending'&&(idx===null||c.frameIndex===idx)).map(c=>({...c,runId:r.runId})));
function previewRecord(f){
  if(!$('preview-proposals').checked)return f.record;
  const r=structuredClone(f.record);
  for(const c of proposals(f.index)){if(c.joint.startsWith('bat'))r[c.joint]=c.point;else r.points[JOINTS.indexOf(c.joint)+5]=c.point;}
  return r;
}
function renderAgentReview(){
  const reports=trial.agentReports||[],pending=proposals(),here=proposals(index),selected=here.find(c=>c.joint===joint);
  $('agent-summary').hidden=!reports.length;
  $('agent-heading').textContent=`Agent review ready · ${pending.length} point suggestions to decide · human review below`;
  $('agent-evidence').replaceChildren();
  for(const r of reports){
    const section=textNode('section','');section.append(textNode('p',r.summary||'Agent review completed.'));
    section.append(textNode('small',(r.models||[]).map(m=>`${(m.path||m.name||'Model').split('/').pop()} · ${(m.sha256||'').slice(0,12)}`).join(' | ')));
    for(const p of r.passes||[])section.append(textNode('p',`${p.name}: ${new Set(p.frames).size} / ${trial.frames.length} frames. ${p.notes||''}`));
    if(r.uncertainties?.length){section.append(textNode('strong',`${r.uncertainties.length} items for your attention`));for(const u of r.uncertainties){const b=textNode('button',`Frame ${u.frameIndex+1} · ${name(u.joint)} · ${u.reason}`,'uncertainty-link');b.onclick=()=>run(async()=>{joint=JOINTS.includes(u.joint)?u.joint:joint;await go(u.frameIndex);});section.append(b);}}
    $('agent-evidence').append(section);
  }
  $('agent-proposals').hidden=!pending.length;$('proposal-count').textContent=here.length;
  $('proposal-list').replaceChildren(...here.map(c=>{const b=textNode('button',name(c.joint)+(c.uncertain?' · check':''),c.joint===joint?'active':'');b.onclick=()=>run(async()=>{joint=c.joint;await render();});return b;}));
  $('proposal-reason').textContent=selected?.reason||(here.length?'Select a suggested point to inspect it.':'No pending suggestions on this frame.');
  $('accept-proposal').disabled=$('reject-proposal').disabled=!selected;
  $('accept-frame').disabled=$('reject-frame').disabled=!here.length;
  $('check').disabled=busy||!!here.length;$('all-joints').disabled=!!here.length;
  if(here.length)$('check').textContent='Decide suggestions first';
  const candidate=selected?.point;
  if(candidate)$('coords').textContent+=` | Proposed ${located(candidate)?`x ${(candidate.x*frame().record.width).toFixed(1)} · y ${(candidate.y*frame().record.height).toFixed(1)} px`: 'unlabelled'} · ${candidate.visibility===1?'hidden estimate':candidate.visibility===2?'visible':'not labelled'}`;
  if(here.length&&$('preview-proposals').checked)$('canvas-note').textContent='Agent preview · orange rings are suggestions · accept or keep current';
}
async function decideProposal(accept,allFrame=false){
  const selected=proposals(index).find(c=>allFrame||c.joint===joint);if(!selected)return;
  await mutate(accept?'accept-proposal':'reject-proposal',{runId:selected.runId,frameIndex:index,joint,allFrame});
}

async function api(path, body) {
  const res=await fetch(path,body===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json','X-Swing-Review':'1'},body:JSON.stringify(body)});
  const data=await res.json();if(!res.ok)throw new Error(data.error||`Request failed (${res.status})`);return data;
}
let toastTimer;
function toast(message,error=false){$('toast').textContent=message;$('toast').className=error?'error':'';$('toast').hidden=false;clearTimeout(toastTimer);toastTimer=setTimeout(()=>$('toast').hidden=true,error?9000:4000);}
async function run(fn){try{await fn();}catch(e){playing=false;$('play').textContent='▶ Play';toast(e.message,true);}}
function remember(){if(trial)localStorage.setItem('swing-review-view',JSON.stringify({id:trial.id,index,joint,pass}));}

async function loadBootstrap(){
  bootstrap=await api('/api/bootstrap');renderQueue();
  const hasSnapshot=Boolean(bootstrap.backend.retrievedAt);
  $('connection').textContent=hasSnapshot?'GCS · read only':'Local videos';
  $('snapshot').textContent=hasSnapshot?`Catalog snapshot ${new Date(bootstrap.backend.retrievedAt).toLocaleString()}`:'No GCS catalog snapshot';
  for(const option of $('catalog-mode').options) option.disabled=option.value!=='local'&&!hasSnapshot;
  if(!hasSnapshot) $('catalog-mode').value='local';
}
function renderQueue(){
  $('queue').replaceChildren();$('trial-count').textContent=`${bootstrap.trials.length} trials`;
  for(const t of bootstrap.trials){
    const b=textNode('button','',`queue-item${t.id===trial?.id?' active':''}`);
    b.append(textNode('span',t.state,'queue-state'),textNode('strong',label(t.row)),textNode('small',`${t.row.camera||'Camera'} · ${t.summary.frames} frames · ${t.summary.changed} changed`));
    if(t.lease?.owner&&t.lease.until>Date.now()/1000)b.append(textNode('small',`Claimed by ${t.lease.owner}`));
    if(t.agentReports?.runs)b.append(textNode('small',`${t.agentReports.pending} agent suggestions · ready to review`));
    b.onclick=()=>run(()=>loadTrial(t.id));$('queue').append(b);
  }
}
async function loadTrial(id,restore=false){
  playing=false;trial=await api(`/api/trials/${id}`);images.clear();pinnedReference=null;index=restore?Math.min(index,trial.frames.length-1):0;
  $('empty').hidden=true;$('workspace').hidden=false;renderQueue();await render();remember();
}
async function mutate(kind,body={},quiet=false){
  if(busy)throw new Error('Wait for the current local save.');
  busy=true;$('save-state').textContent='Saving…';
  const id=trial.id;
  try{
    await api(`/api/trials/${id}/${kind}`,{...body,actor,expectedRevision:trial.revision,requestId:crypto.randomUUID()});
    trial=await api(`/api/trials/${id}`);await render();
    if(!quiet)toast('Saved locally');
  }catch(e){
    trial=await api(`/api/trials/${id}`);await render();throw e;
  }finally{busy=false;$('check').disabled=!!proposals(index).length;$('save-state').textContent=`Saved locally · r${trial.revision}`;}
}
function hasPass(f,p=pass){return p==='joints'?!!f.meta.passes.joints?.[joint]:!!f.meta.passes[p];}
function unresolved(f){return f.issues.filter(x=>!f.meta.acknowledged.includes(x));}
function guidance(){
  if(joint==='batKnob')return '“Bat knob” in the saved format means the shaft leaving the TOP of the hands toward the barrel. Place it above the hands.';
  if(joint==='batTip')return 'Place the marker at the actual end of the barrel.';
  if(joint.includes('knee'))return 'Center the actual knee joint—not the thigh or inner bend edge. Right means the athlete’s right; it was the pink/back leg in prior examples.';
  if(joint.includes('ankle'))return 'Center the ankle joint—not the shoe tongue or foot.';
  if(joint.includes('elbow'))return 'Center the actual elbow bend—not the sleeve edge, biceps, glove or forearm.';
  if(joint.includes('shoulder'))return 'Place inside the shoulder joint center—not near the neck or high shoulder surface.';
  return 'Use the athlete’s anatomical side. Center the actual joint and inspect neighboring frames when uncertain.';
}
async function render(){
  if(!trial)return;
  const f=frame(),r=f.record,s=trial.summary;
  const queued=bootstrap?.trials.find(t=>t.id===trial.id);if(queued){Object.assign(queued,{state:trial.state,revision:trial.revision,summary:trial.summary,lease:trial.lease,agentReports:{runs:(trial.agentReports||[]).length,pending:proposals().length}});renderQueue();}
  $('session-name').textContent=(trial.asset.row.session||'Source trial').replaceAll('_',' ');
  $('trial-name').textContent=label(trial.asset.row);$('camera').textContent=`${trial.asset.row.camera||'Camera'} · ${r.width} × ${r.height} · ${trial.frames.length} frames`;
  $('state').textContent=trial.state;$('save-state').textContent=`Saved locally · r${trial.revision}`;
  for(const p of ['placement','replay','joints','final'])$('count-'+p).textContent=`${s.remaining[p]} ${p==='joints'?'joint checks':'frames'} remaining`;
  document.querySelectorAll('[data-pass]').forEach(b=>b.classList.toggle('active',b.dataset.pass===pass));
  $('frame-label').textContent=`Frame ${index+1} / ${trial.frames.length}`;
  $('phase-label').textContent=index<trial.settings.swingStart?'SETUP':index>trial.settings.swingEnd?'FOLLOW-THROUGH':'SWING · EVERY FRAME';
  $('jump').value=index+1;$('jump').max=trial.frames.length;
  $('lock').textContent=f.meta.locked?'● Locked reference':'○ Unlocked · edit';
  $('canvas-note').textContent=f.meta.locked?'Reference protected · unlock to edit':`Click or drag to place ${name(joint).toLowerCase()}`;
  $('joint').value=joint;$('joint-guidance').textContent=guidance();
  const p=point(r,joint),b=point(f.baseline,joint);
  $('coords').textContent=located(p)?`x ${(p.x*r.width).toFixed(1)} · y ${(p.y*r.height).toFixed(1)} px${located(b)?` | Δ ${Math.hypot((p.x-b.x)*r.width,(p.y-b.y)*r.height).toFixed(1)} px`:''}`:'No located point';
  $('visibility').value=p?.visibility??0;$('visibility').disabled=f.meta.locked;
  $('uncertain').checked=f.meta.uncertain.includes(joint);$('uncertain').disabled=false;
  $('note').value=f.meta.note;
  $('check').textContent=pass==='replay'?'▶ Replay & record checks':hasPass(f)?'Checked · next ↵':'Checked & next ↵';
  $('check').disabled=busy;
  $('joint-progress').replaceChildren(...JOINTS.map(j=>{const el=textNode('span','',`joint-dot${f.meta.passes.joints?.[j]?' done':''}`);el.title=name(j);return el;}));
  $('flag-count').textContent=unresolved(f).length;
  $('issues').replaceChildren(...f.issues.map(issue=>textNode('div',issue.replaceAll(':',' · ').replaceAll('_',' '),`flag${f.meta.acknowledged.includes(issue)?' ack':''}`)));
  if(!f.issues.length)$('issues').append(textNode('p','No automated flags','clear'));
  $('acknowledge').hidden=!f.issues.some(x=>x.startsWith('jump:')||x.startsWith('possible-swap:'));
  $('swing-start').value=trial.settings.swingStart+1;$('swing-end').value=trial.settings.swingEnd+1;
  $('export').hidden=trial.state!=='approved';
  $('all-joints').hidden=pass!=='joints';
  renderAgentReview();
  $('timeline').replaceChildren(...trial.frames.map(f=>{
    const interpolated=JOINTS.some(j=>point(f.record,j)?.source==='interpolated');
    const b=textNode('button','',`tick${f.meta.changed?' changed':''}${interpolated?' interpolated':''}${hasPass(f)?' checked':''}${unresolved(f).length?' issue':''}${f.index===index?' current':''}`);
    const proposed=proposals(f.index).length;b.classList.toggle('proposed',!!proposed);
    b.title=`Frame ${f.index+1}${f.meta.locked?' · reference':''}${hasPass(f)?' · checked':''}${proposed?' · agent suggestions':''}`;b.setAttribute('aria-label',b.title);b.onclick=()=>run(()=>go(f.index));return b;
  }));
  const from=Math.min(Math.max(0,index-4),Math.max(0,trial.frames.length-9));
  $('filmstrip').replaceChildren(...trial.frames.slice(from,from+9).map(f=>{
    const b=textNode('button','',`film${f.index===index?' active':''}`),img=document.createElement('img');img.src=frameUrl(f.index);img.alt=`Frame ${f.index+1}`;img.loading='lazy';b.append(img,textNode('span',`${f.index+1}${f.meta.locked?' · ●':''}`));b.onclick=()=>run(()=>go(f.index));return b;
  }));
  const auto=textNode('option','Auto nearest');auto.value='auto';
  $('reference-select').replaceChildren(auto,...trial.frames.filter(f=>f.meta.locked).map(f=>{const o=textNode('option',`Frame ${f.index+1}`);o.value=f.index;return o;}));
  if(pinnedReference!==null&&trial.frames[pinnedReference]?.meta.locked)$('reference-select').value=pinnedReference;
  else{pinnedReference=null;$('reference-select').value='auto';}
  $('play').textContent=playing?'Ⅱ Pause':'▶ Play';remember();await drawFrame();
}
const frameUrl=i=>`/api/trials/${trial.id}/frames/${i}`;
async function imageAt(i){
  const key=frameUrl(i);let entry=images.get(key);
  if(!entry){entry=new Promise((resolve,reject)=>{const img=new Image();img.onload=()=>resolve(img);img.onerror=()=>{images.delete(key);reject(new Error('Could not decode source frame'));};img.src=key;});images.set(key,entry);if(images.size>24)images.delete(images.keys().next().value);}
  return entry;
}
function draw(canvas,img,f,{baseline=false,zoom=false,preview=false}={}){
  const ctx=canvas.getContext('2d'),w=canvas.width,h=canvas.height;ctx.clearRect(0,0,w,h);
  if(!img||!f)return;
  const r=baseline?f.baseline:preview?previewRecord(f):f.record;
  let scale=w/img.naturalWidth,ox=0,oy=0;
  if(zoom){const p=point(r,joint);scale=Math.max(w/img.naturalWidth,h/img.naturalHeight)*3.6;ox=w/2-(located(p)?p.x:.5)*img.naturalWidth*scale;oy=h/2-(located(p)?p.y:.5)*img.naturalHeight*scale;}
  ctx.save();ctx.translate(ox,oy);ctx.scale(scale,scale);ctx.drawImage(img,0,0);
  if($('overlay').checked||zoom){
    ctx.lineWidth=(zoom?1.2:1.4)/scale;
    for(const [a,b] of BONES){const pa=point(r,JOINTS[a]),pb=point(r,JOINTS[b]);if(located(pa)&&located(pb)){ctx.strokeStyle=color(JOINTS[a])+'b0';ctx.beginPath();ctx.moveTo(pa.x*img.naturalWidth,pa.y*img.naturalHeight);ctx.lineTo(pb.x*img.naturalWidth,pb.y*img.naturalHeight);ctx.stroke();}}
    for(const j of JOINTS){const p=point(r,j);if(!located(p))continue;const x=p.x*img.naturalWidth,y=p.y*img.naturalHeight;
      if(!baseline&&$('baseline').checked){const b=point(f.baseline,j);if(located(b)){ctx.strokeStyle='#fff9';ctx.setLineDash([3/scale,3/scale]);ctx.beginPath();ctx.moveTo(b.x*img.naturalWidth,b.y*img.naturalHeight);ctx.lineTo(x,y);ctx.stroke();ctx.strokeRect(b.x*img.naturalWidth-3/scale,b.y*img.naturalHeight-3/scale,6/scale,6/scale);ctx.setLineDash([]);}}
      ctx.beginPath();ctx.arc(x,y,(j===joint?5:3.2)/scale,0,Math.PI*2);ctx.fillStyle=color(j);ctx.fill();ctx.lineWidth=1/scale;ctx.strokeStyle='#122026';ctx.stroke();
      if(j===joint){ctx.beginPath();ctx.arc(x,y,8/scale,0,Math.PI*2);ctx.strokeStyle='#fff';ctx.stroke();}
      if(preview&&$('preview-proposals').checked&&proposals(f.index).some(c=>c.joint===j)){ctx.beginPath();ctx.arc(x,y,10/scale,0,Math.PI*2);ctx.strokeStyle='#ffab5c';ctx.lineWidth=2/scale;ctx.stroke();const old=point(f.record,j);if(located(old)){ctx.strokeStyle='#fff9';ctx.lineWidth=1/scale;ctx.strokeRect(old.x*img.naturalWidth-3/scale,old.y*img.naturalHeight-3/scale,6/scale,6/scale);}}
      if(p.visibility===1){ctx.strokeStyle='#fff';ctx.beginPath();ctx.moveTo(x-3/scale,y);ctx.lineTo(x+3/scale,y);ctx.stroke();}
    }
  }
  ctx.restore();
}
async function drawFrame(){
  const epoch=++drawEpoch,idx=index,f=frame();$('image-loading').hidden=false;
  const img=await imageAt(idx);if(epoch!==drawEpoch)return;
  currentImage=img;$('canvas').width=img.naturalWidth;$('canvas').height=img.naturalHeight;
  draw($('canvas'),img,f,{preview:true});draw($('zoom'),img,f,{zoom:true,preview:true});$('image-loading').hidden=true;
  $('neighbor-label').textContent=idx?'PREVIOUS FRAME · '+idx:'NEXT FRAME · 2';
  const neighbor=Math.min(trial.frames.length-1,idx?idx-1:1);
  const refs=trial.frames.filter(f=>f.meta.locked&&f.index!==idx).sort((a,b)=>Math.abs(a.index-idx)-Math.abs(b.index-idx)||b.index-a.index);
  const ref=pinnedReference??refs[0]?.index;
  const results=await Promise.allSettled([imageAt(neighbor),ref!==undefined?imageAt(ref):Promise.resolve(null)]);
  if(epoch!==drawEpoch)return;
  if(results[0].status==='fulfilled')draw($('neighbor'),results[0].value,trial.frames[neighbor]);
  if(results[1].status==='fulfilled')draw($('reference'),results[1].value,trial.frames[ref]);
  for(let i=idx+1;i<=Math.min(idx+3,trial.frames.length-1);i++)imageAt(i).catch(()=>{});
}
async function go(i,fromPlayback=false){
  if(!trial)return;if(!fromPlayback)playing=false;
  index=Math.max(0,Math.min(trial.frames.length-1,i));await render();
}
async function playback(record=false){
  if(playing){playing=false;return;}
  playing=true;recordReplay=record;
  if(index===trial.frames.length-1)index=0;
  if(record){pass='replay';index=0;toast('Recording the replay pass as each displayed frame is inspected. Pause any time.');}
  try{
    while(playing){
      if(document.hidden){playing=false;break;}
      await go(index,true);
      const idx=index;
      const dt=(trial.frames[idx+1]?.record.frame.time??(frame().record.frame.time+1/30))-frame().record.frame.time;
      await sleep(Math.max(100,Math.min(1000,dt*1000/Number($('speed').value))));
      if(!playing||document.hidden)break;
      if(recordReplay&&!hasPass(frame(),'replay'))await mutate('review',{frameIndex:index,pass:'replay'},true);
      if(index>=trial.frames.length-1){playing=false;break;}
      index++;
    }
  }finally{playing=false;$('play').textContent='▶ Play';}
}
async function checked(){
  if(pass==='replay')return playback(true);
  playing=false;if(!hasPass(frame()))await mutate('review',{frameIndex:index,pass,joint},true);
  if(index<trial.frames.length-1)await go(index+1);else toast('End of this pass. Select another joint or review pass.');
}
async function edit(p,uncertain){
  if(frame().meta.locked)throw new Error('Unlock this reference frame to edit its points.');
  const change={frameIndex:index,joint,point:p};if(uncertain!==undefined)change.uncertain=uncertain;
  await mutate('batch',{changes:[change]},true);
}
function pointerPos(e){const r=$('canvas').getBoundingClientRect();return{x:Math.max(0,Math.min(1,(e.clientX-r.left)/r.width)),y:Math.max(0,Math.min(1,(e.clientY-r.top)/r.height))};}
$('canvas').onpointerdown=e=>{
  if(busy||playing||!trial||$('image-loading').hidden===false)return;
  if($('preview-proposals').checked&&proposals(index).length){toast('Accept a suggestion or turn off Preview suggestions to edit saved points.');return;}
  const pos=pointerPos(e),rect=$('canvas').getBoundingClientRect();let near=null,dist=15;
  for(const j of JOINTS){const p=point(frame().record,j);if(located(p)){const d=Math.hypot((p.x-pos.x)*rect.width,(p.y-pos.y)*rect.height);if(d<dist){near=j;dist=d;}}}
  if(near){joint=near;render();}
  if(frame().meta.locked){if(!near)toast('Reference protected. Unlock to place or drag a joint.');return;}
  pointer={pos,start:point(frame().record,joint)?structuredClone(point(frame().record,joint)):null};$('canvas').setPointerCapture(e.pointerId);
};
$('canvas').onpointermove=e=>{if(!pointer)return;pointer.pos=pointerPos(e);const preview=structuredClone(frame());const p={...(point(preview.record,joint)||{}),...pointer.pos,visibility:2};if(joint.startsWith('bat'))preview.record[joint]=p;else preview.record.points[JOINTS.indexOf(joint)+5]=p;draw($('canvas'),currentImage,preview);draw($('zoom'),currentImage,preview,{zoom:true});};
$('canvas').onpointerup=e=>{if(!pointer)return;const pos=pointerPos(e);pointer=null;run(()=>edit({...pos,visibility:2}));};
$('canvas').onpointercancel=()=>{pointer=null;drawFrame();};

let catalogRequest=0;
async function catalog(){
  const request=++catalogRequest,mode=$('catalog-mode').value,local=mode==='local';
  $('catalog-summary').textContent=local?'Looking for local videos…':'Reading source catalog…';
  $('catalog-rows').replaceChildren();
  const result=await api(`/api/catalog?q=${encodeURIComponent($('search').value)}&mode=${mode}`);
  if(request!==catalogRequest)return;
  const item=local?'video':'trial';
  const matched=`${result.total} matching ${item}${result.total===1?'':'s'}${result.total>100?' · showing first 100':''}. Import creates a local draft.`;
  $('catalog-summary').textContent=local&&!result.total&&!$('search').value
    ?'No videos found. Add MP4, MOV or M4V files to the configured local video folder.'
    :matched;
  $('catalog-rows').replaceChildren(...result.rows.map(row=>{
    const el=textNode('div','','catalog-row'),info=textNode('div','');
    const details=local?row.session:`${row.camera||'Camera'} · ${row.session||''} · shared: ${row.sharedStatus}${row.cached?' · cached source':''}`;
    info.append(textNode('strong',label(row)),textNode('small',details));
    const loaded=bootstrap.trials.some(t=>t.id===row.id),b=textNode('button',loaded?'Open draft':'Load local copy');b.disabled=row.sharedStatus==='active'||row.rejected||row.sharedStatus==='rejected';
    b.onclick=()=>run(async()=>{b.disabled=true;b.textContent='Loading…';try{if(!loaded)await api('/api/import',{id:row.id});await loadBootstrap();await loadTrial(row.id);$('catalog-dialog').close();}finally{b.disabled=false;b.textContent=loaded?'Open draft':'Load local copy';}});el.append(info,b);return el;
  }));
}
function openCatalog(mode){
  if(!$('catalog-mode').querySelector(`option[value="${mode}"]`).disabled) $('catalog-mode').value=mode;
  $('search').value='';
  $('catalog-dialog').showModal();return catalog();
}
$('browse').onclick=$('empty-browse').onclick=()=>run(()=>openCatalog('cached'));
$('browse-local').onclick=()=>run(()=>openCatalog('local'));
$('close-catalog').onclick=()=>$('catalog-dialog').close();
let searchTimer;$('search').oninput=()=>{clearTimeout(searchTimer);searchTimer=setTimeout(()=>run(catalog),180);};$('catalog-mode').onchange=()=>run(catalog);
$('refresh').onclick=()=>run(async()=>{$('refresh').disabled=true;try{await api('/api/refresh',{});await loadBootstrap();toast('Updated the local catalog snapshot from GCS.');}finally{$('refresh').disabled=false;}});
$('joint').replaceChildren(...JOINTS.map((j,i)=>{const o=textNode('option',NAMES[i]);o.value=j;return o;}));
$('joint').onchange=()=>run(async()=>{joint=$('joint').value;await render();});
document.querySelectorAll('[data-pass]').forEach(b=>b.onclick=()=>run(async()=>{playing=false;pass=b.dataset.pass;await render();}));
$('prev').onclick=()=>run(()=>go(index-1));$('next').onclick=()=>run(()=>go(index+1));$('prev5').onclick=()=>run(()=>go(index-5));$('next5').onclick=()=>run(()=>go(index+5));
$('jump').onchange=()=>run(()=>go(Number($('jump').value)-1));$('play').onclick=()=>run(()=>playback());$('check').onclick=()=>run(checked);
$('overlay').onchange=$('baseline').onchange=()=>run(drawFrame);
$('preview-proposals').onchange=()=>run(render);
$('accept-proposal').onclick=()=>run(()=>decideProposal(true));$('reject-proposal').onclick=()=>run(()=>decideProposal(false));
$('accept-frame').onclick=()=>run(()=>decideProposal(true,true));$('reject-frame').onclick=()=>run(()=>decideProposal(false,true));
$('next-proposal').onclick=()=>run(async()=>{const p=proposals().sort((a,b)=>a.frameIndex-b.frameIndex),next=p.find(c=>c.frameIndex>index)||p[0];if(next){joint=next.joint;await go(next.frameIndex);}});
$('reference-select').onchange=()=>run(async()=>{pinnedReference=$('reference-select').value==='auto'?null:Number($('reference-select').value);await drawFrame();});
$('lock').onclick=()=>run(()=>mutate('lock',{frameIndex:index,locked:!frame().meta.locked}));
$('visibility').onchange=()=>run(()=>edit({visibility:Number($('visibility').value)}));
$('uncertain').onchange=()=>run(()=>mutate('flag',{frameIndex:index,joint,uncertain:$('uncertain').checked},true));
$('save-note').onclick=()=>run(()=>mutate('note',{frameIndex:index,note:$('note').value}));
$('acknowledge').onclick=()=>run(()=>mutate('acknowledge',{frameIndex:index}));
$('save-range').onclick=()=>run(()=>mutate('settings',{swingStart:Number($('swing-start').value)-1,swingEnd:Number($('swing-end').value)-1}));
$('interpolate').onclick=()=>run(()=>mutate('interpolate',{start:Number($('interp-start').value)-1,end:Number($('interp-end').value)-1,joint}));
$('undo').onclick=()=>run(()=>mutate('undo'));
$('all-joints').onclick=()=>run(async()=>{await mutate('review',{frameIndex:index,pass:'joints',allJointsInspected:true},true);await go(index+1);});
$('next-pending').onclick=()=>run(async()=>{const pending=trial.frames.slice(index+1).concat(trial.frames.slice(0,index+1)).find(f=>!hasPass(f));if(pending)await go(pending.index);else toast('This pass is complete for the selected joint.');});
$('history').onclick=()=>run(async()=>{const history=await api(`/api/trials/${trial.id}/history`);$('history-rows').replaceChildren(...history.map(h=>{const el=textNode('div',`r${h.revision} · ${h.kind}`,'history-row');el.append(textNode('small',`${h.actor.id} · ${new Date(h.created).toLocaleString()}${h.detail.note?' · '+h.detail.note:''}`));return el;}));if(!history.length)$('history-rows').textContent='Immutable source imported. No local edits yet.';$('history-dialog').showModal();});
$('close-history').onclick=()=>$('history-dialog').close();
$('approve').onclick=()=>{playing=false;const s=trial.summary;$('approval-summary').textContent=`${proposals().length} agent suggestions · ${s.remaining.placement} placement frames · ${s.remaining.replay} replay frames · ${s.remaining.joints} joint checks · ${s.remaining.final} final frames · ${s.issues} unresolved flags`;$('confirm-approve').disabled=proposals().length>0||Object.values(s.remaining).some(n=>n>0)||s.issues>0;$('approval-dialog').showModal();};
$('cancel-approve').onclick=()=>$('approval-dialog').close();
$('confirm-approve').onclick=()=>run(async()=>{await mutate('approve',{confirmation:'Approve local draft'});$('approval-dialog').close();toast('Approved locally. Nothing was uploaded.');await loadBootstrap();});
$('export').onclick=()=>run(async()=>{const data=await api(`/api/trials/${trial.id}/export`),url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'})),a=document.createElement('a');a.href=url;a.download=`${trial.id}-approved-r${trial.revision}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),30000);});
document.addEventListener('keydown',e=>{
  if(!trial||document.querySelector('dialog[open]')||['INPUT','TEXTAREA','SELECT'].includes(document.activeElement?.tagName))return;
  if(e.code==='Space'){e.preventDefault();run(()=>playback());}
  if(e.key==='ArrowRight'||e.key==='ArrowLeft'){e.preventDefault();run(()=>go(index+(e.key==='ArrowRight'?1:-1)*(e.shiftKey?5:1)));}
  if(e.key==='Enter'){e.preventDefault();run(checked);}
  if(e.key.toLowerCase()==='j'){joint=JOINTS[(JOINTS.indexOf(joint)+1)%JOINTS.length];run(render);}
  if((e.metaKey||e.ctrlKey)&&e.key==='z'){e.preventDefault();run(()=>mutate('undo'));}
});
document.addEventListener('visibilitychange',()=>{if(document.hidden)playing=false;});
await run(async()=>{await loadBootstrap();let saved;try{saved=JSON.parse(localStorage.getItem('swing-review-view'));}catch{}if(saved&&bootstrap.trials.some(t=>t.id===saved.id)){index=saved.index||0;joint=JOINTS.includes(saved.joint)?saved.joint:joint;pass=['placement','replay','joints','final'].includes(saved.pass)?saved.pass:pass;await loadTrial(saved.id,true);}else if(bootstrap.trials.length)await loadTrial(bootstrap.trials[0].id);});
// Pick up completed agent drafts without interrupting editing, playback, or an open dialog.
setInterval(async()=>{
  if(document.hidden||busy||playing||pointer||document.querySelector('dialog[open]')||['INPUT','TEXTAREA','SELECT'].includes(document.activeElement?.tagName))return;
  try{
    await loadBootstrap();
    const latest=bootstrap.trials.find(t=>t.id===trial?.id);
    if(latest&&latest.revision!==trial.revision&&!busy&&!playing){trial=await api(`/api/trials/${trial.id}`);await render();toast('Loaded the latest local draft revision.');}
  }catch{$('connection').textContent='Local server unavailable';}
},5000);
