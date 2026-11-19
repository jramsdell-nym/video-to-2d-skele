'use strict';
const $=id=>document.getElementById(id),R=window.REVIEW,M=window.MEDIAPIPE,V=$('video');
for(const [id,models] of Object.entries(M.clips))Object.assign(R.clips[id],models);
let clip='occlusion',mode='body',index=0,token=0,ready=false,threshold=.3;
const bones=[['left_shoulder','right_shoulder'],['left_shoulder','left_hip'],['right_shoulder','right_hip'],['left_hip','right_hip'],['left_shoulder','left_elbow'],['left_elbow','left_wrist'],['right_shoulder','right_elbow'],['right_elbow','right_wrist'],['left_hip','left_knee'],['left_knee','left_ankle'],['right_hip','right_knee'],['right_knee','right_ankle']];
const major=['left_shoulder','right_shoulder','left_elbow','right_elbow','left_wrist','right_wrist','left_hip','right_hip','left_knee','right_knee','left_ankle','right_ankle'];
const good=p=>p&&p.confidence>=threshold&&p.x>=0&&p.x<=1&&p.y>=0&&p.y<=1;
const labels={elian:'Elian · left',trey:'Trey · right',occlusion:'Trey · people + occlusion',seed:'Nov 6 · seed'};
const titles={original:'Apple Vision · original',selected:'Apple Vision · selected hitter',pose:'YOLO11s-pose',full:'MediaPipe Full',heavy:'MediaPipe Heavy'};
const originalFindings={elian:'Watch wrists and elbows as the bat becomes foreshortened and accelerates. The original and selected Vision outputs agree because the original rule already retained this clip.',trey:'The bat rack creates background detections. The existing 50px wrist-distance gate rejects inspected rack boxes, but the dark swinging bat is still missed through much of the downswing.',occlusion:'Original Vision returned no pose in all 120 frames because of its multi-person rule. Hitter selection recovers observations; inspect wrist and elbow placement rather than treating coverage as accuracy.',seed:'Bright-bat detections are more frequent than in the dark-bat clips. Image-line candidates still need visual verification.'};
function info(){return R.manifest.clips.find(c=>c.id===clip)}
function jointCount(f){return major.filter(j=>good(f.points[j])).length;}
function present(k,i){const d=R.clips[clip];if(k==='source')return true;if(k==='bat')return !!d.bat[i].detection;if(k==='line')return d.line[i].status!=='missing';return jointCount(d[k][i])>=5;}
function setupPanels(){
 const config=mode==='body'?[['source','Original source','Same decoded frame · no overlay'],['original',titles.original,'Existing app rule: reject multiple people'],['selected',titles.selected,'Largest person + gated temporal association'],['pose',titles.pose,'COCO pretrained · MPS · same association'],['full',titles.full,'VIDEO · up to 4 poses · CPU · same association'],['heavy',titles.heavy,'VIDEO · up to 4 poses · CPU · same association']]:[['source','Original source','No estimated geometry'],['bat','YOLO11m · bounding box','Observed box only; no physical endpoints'],['line','OpenCV · candidate segment','Solid = image line; dashed = gated flow']];
 $('body-options').hidden=mode!=='body';
 $('panels').innerHTML=config.map(([key,title,sub])=>`<section class="panel"><h3>${title}</h3><p>${sub}</p><canvas width="960" height="540" data-key="${key}" aria-label="${title} overlay"></canvas><canvas class="timeline" width="720" height="15" data-track="${key}" aria-label="${title} coverage timeline"></canvas><div class="state" id="state-${key}"></div></section>`).join('');
 $('legend').innerHTML=mode==='body'?'<span class="dot" style="background:#37e1ff"></span>Anatomical left <span class="dot" style="background:#ff7d6e"></span>Anatomical right. Same 12 shared joints; default display threshold ≥0.30. Points below the display threshold or outside the image are not drawn. Coverage is model-specific score-threshold coverage: Vision confidence, YOLO keypoint scores and MediaPipe visibility are not calibrated equivalents. MediaPipe presence is saved separately and is not the coverage score.':'<span class="dot" style="background:#ffeb4b"></span>Observed bbox <span class="dot" style="background:#eb69eb"></span>Observed image segment <span class="dot" style="background:#ffa532"></span>Propagated segment · Gray = missing. No interpolation. Segment endpoints are not tip/knob.';
 $('finding').textContent=mode==='body'?(M.findings?.[clip]||'Compare identical source frames. Use Upper body for wrists/elbows and Feet for ankles/heels. These outputs are unvalidated estimates; full clip coverage does not establish correct placement or identity.') :originalFindings[clip];
 document.querySelectorAll('canvas[data-key]').forEach(c=>c.onclick=()=>c.parentElement.classList.toggle('expanded'));
 document.querySelectorAll('[data-track]').forEach(c=>c.onclick=e=>seek(Math.min(119,Math.floor((e.clientX-c.getBoundingClientRect().left)/c.clientWidth*120))));
 draw();
}
function jointOverlay(ctx,f,full=false){
 const p=f.points,edges=full?M.edges.map(([a,b])=>[M.names[a],M.names[b]]):bones;ctx.lineWidth=2;ctx.lineCap='round';
 const color=n=>n.startsWith('left')?'#37e1ff':n.startsWith('right')?'#ff7d6e':'#e9eff5';
 for(const[a,b]of edges){if(!good(p[a])||!good(p[b]))continue;ctx.strokeStyle=color(a)===color(b)?color(a):'#e9eff5';ctx.beginPath();ctx.moveTo(p[a].x*960,p[a].y*540);ctx.lineTo(p[b].x*960,p[b].y*540);ctx.stroke();}
 for(const[n,q]of Object.entries(p)){if(!good(q)||(!full&&!major.includes(n)))continue;ctx.fillStyle=color(n);ctx.beginPath();ctx.arc(q.x*960,q.y*540,3,0,7);ctx.fill();if(n.endsWith('wrist')){ctx.font='bold 14px system-ui';ctx.fillText(n[0].toUpperCase(),q.x*960+5,q.y*540-5);}}
}
function draw(){
 if(!ready||V.readyState<2)return;const d=R.clips[clip];$('scrub').value=index;$('clock').textContent=`${d.pose[index].time.toFixed(3)} s · ${index+1}/120`;$('status').textContent=V.paused?'Paused':'Playing';$('play').textContent=V.paused?'Play':'Pause';
 for(const can of document.querySelectorAll('canvas[data-key]')){
  const key=can.dataset.key,ctx=can.getContext('2d');ctx.clearRect(0,0,960,540);ctx.save();
  if(mode==='body'){const [x1,y1,x2,y2]=M.crops[clip][$('view').value];ctx.scale(1/(x2-x1),1/(y2-y1));ctx.translate(-x1*960,-y1*540);}
  ctx.drawImage(V,0,0,960,540);ctx.setLineDash([]);let text='';
  if(titles[key]){const f=d[key][index];jointOverlay(ctx,f,$('full33').checked&&['full','heavy'].includes(key));text=`${Object.keys(f.points).length?'H1':'No pose output'} · ${jointCount(f)}/12 shared joints ≥${threshold.toFixed(2)} · ${f.people} returned people · source frame ${index+1}`;}
  else if(key==='bat'){const f=d.bat[index],b=f.detection;ctx.strokeStyle='#ffeb4b';ctx.lineWidth=3;if(b){const[x1,y1,x2,y2]=b.bbox;ctx.strokeRect(x1*960,y1*540,(x2-x1)*960,(y2-y1)*540);text=`Observed bbox · confidence ${b.confidence.toFixed(2)} · no tip/knob`;}else text='MISSING · no accepted bat box';for(let j=Math.max(0,index-7);j<index;j++){const q=d.bat[j].detection;if(!q)continue;const b=q.bbox;ctx.fillStyle='#ffeb4b';ctx.beginPath();ctx.arc((b[0]+b[2])*480,(b[1]+b[3])*270,3,0,7);ctx.fill();}}
  else if(key==='line'){const f=d.line[index];if(f.segment){ctx.strokeStyle=f.status==='observed_line'?'#eb69eb':'#ffa532';ctx.lineWidth=4;ctx.setLineDash(f.status==='propagated_line'?[10,8]:[]);const[a,b]=f.segment;ctx.beginPath();ctx.moveTo(a[0]*960,a[1]*540);ctx.lineTo(b[0]*960,b[1]*540);ctx.stroke();ctx.setLineDash([]);}text=`${f.status} · ${f.reason}`;}
  else text=`Source frame ${index+1} · ${d.pose[index].time.toFixed(3)} s · original orientation`;
  ctx.restore();$('state-'+key).textContent=text;const tl=document.querySelector(`[data-track="${key}"]`),tc=tl.getContext('2d');for(let i=0;i<120;i++){tc.fillStyle=present(key,i)?key==='line'&&d.line[i].status==='propagated_line'?'#ffa532':'#53b9a5':'#35454f';tc.fillRect(i*6,0,6,15);}tc.fillStyle='#fff';tc.fillRect(index*6,0,3,15);
 }
}
function frameAt(t){const fs=R.clips[clip].pose;let i=0;while(i+1<fs.length&&fs[i+1].time<=t+.00001)i++;return i;}
function callback(now,meta){if(ready){index=frameAt(meta.mediaTime);draw();}V.requestVideoFrameCallback(callback);}
V.requestVideoFrameCallback(callback);
V.addEventListener('seeked',()=>{if(ready){index=frameAt(V.currentTime);draw();}});V.addEventListener('ended',draw);
function seek(i){if(!ready)return;V.pause();index=Math.max(0,Math.min(119,i));V.currentTime=R.clips[clip].pose[index].time+.00001;}
async function choose(id){
 const my=++token;V.pause();ready=false;clip=id;index=0;const c=info();$('title').textContent=`${labels[id]} · ${c.trial.replace('markers_','')}`;$('description').textContent=`${c.case} · ${c.session} · Camera 5 · Side: ${c.side} (${c.sideSource})`;$('status').textContent='Loading source…';$('source').src=c.source;
 document.querySelectorAll('[data-clip]').forEach(b=>b.classList.toggle('active',b.dataset.clip===id));
 $('downloads').innerHTML=[['mediapipe-comparison.mp4','All five models MP4'],['mediapipe-detail.mp4','Wrist / elbow comparison MP4'],['mediapipe-full-overlay.mp4','MediaPipe Full · 33-point MP4'],['mediapipe-heavy-overlay.mp4','MediaPipe Heavy · 33-point MP4'],['mediapipe-contact-sheet.jpg','Six-phase contact sheet'],['body-comparison.mp4','Original three-model MP4'],['bat-comparison.mp4','Bat comparison MP4'],['learned-overlay.mp4','YOLO + bat MP4'],['vision-lines-overlay.mp4','Vision + line MP4']].map(([file,title])=>`<a class="link" href="outputs/${id}/${file}">${title}</a>`).join('');
 setupPanels();const loaded=new Promise((resolve,reject)=>{V.addEventListener('loadeddata',resolve,{once:true});V.addEventListener('error',reject,{once:true});});V.src=c.source;V.playbackRate=Number($('speed').value);try{await loaded;if(my===token){ready=true;index=0;draw();}}catch{$('status').textContent='Source failed to load';}
}
$('clips').innerHTML=R.manifest.clips.map(c=>`<button data-clip="${c.id}">${labels[c.id]}</button>`).join('');document.querySelectorAll('[data-clip]').forEach(b=>b.onclick=()=>choose(b.dataset.clip));
document.querySelectorAll('[data-mode]').forEach(b=>b.onclick=()=>{mode=b.dataset.mode;document.querySelectorAll('[data-mode]').forEach(x=>x.classList.toggle('active',x===b));setupPanels();});
document.querySelectorAll('[data-jump]').forEach(b=>b.onclick=()=>seek(clip==='seed'&&b.dataset.jump==='80'?60:+b.dataset.jump));
$('scrub').oninput=()=>seek(+$('scrub').value);$('prev').onclick=()=>seek(index-1);$('next').onclick=()=>seek(index+1);$('speed').onchange=()=>V.playbackRate=+$('speed').value;
$('play').onclick=async()=>{if(!ready)return;if(V.paused){if(index===119){seek(0);await new Promise(r=>V.addEventListener('seeked',r,{once:true}));}await V.play();}else V.pause();draw();};
$('view').onchange=draw;$('full33').onchange=draw;$('threshold').onchange=()=>{threshold=+$('threshold').value;draw();};
$('inspect').onclick=()=>{$('view').value='upper';seek(M.inspectFrame[clip]);};
window.addEventListener('keydown',e=>{if(['INPUT','SELECT','TEXTAREA'].includes(e.target.tagName))return;if(e.key==='ArrowLeft'||e.key==='ArrowRight'){e.preventDefault();seek(index+(e.key==='ArrowLeft'?-1:1));}});
$('body-metrics').innerHTML=M.metrics.flatMap(m=>Object.entries(m.models).map(([k,v])=>`<tr><td>${labels[m.clip]}</td><td>${titles[k]}</td><td>${v.retained}/120</td><td>${(v.sharedJointCoverage['0.3']*100).toFixed(1)}%</td><td>${(v.sharedJointCoverage['0.7']*100).toFixed(1)}%</td><td>${(v.wristCoverage*100).toFixed(1)}%</td><td>${v.timing?v.timing.meanProcessingMs.toFixed(1)+' ms':'see original timing below'}</td></tr>`)).join('');
$('metrics').innerHTML=R.metrics.map(m=>`<tr><td>${labels[m.id]}</td><td>${m.bodyFrames.original}/120</td><td>${m.bodyFrames.selected}/120</td><td>${m.bodyFrames.pose}/120</td><td>${m.bboxFrames}/120</td><td>${m.lineFrames.observed_line} / ${m.lineFrames.propagated_line}</td></tr>`).join('');
const angles=['Obstructed sample','Open side; background people','Open side; background people','Open side; exact seed'];$('sessions').innerHTML=R.sessions.sessions.map((s,i)=>`<tr><td>${s.session.replace('session_','')}</td><td>${s.markerTrials}</td><td>${s.cameras['cam4.mp4']||0}</td><td>${s.cameras['cam5.mp4']||0}</td><td>${angles[i]}</td></tr>`).join('');
choose(clip);
