/* Uses an isolated SQLite copy and a fresh browser profile. Never touches the user's review DB. */
const fs=require('fs'),path=require('path'),os=require('os'),{spawn,spawnSync}=require('child_process'),assert=require('assert/strict');
const {chromium}=require(process.env.SWING_PLAYWRIGHT||'/Users/jramsdell/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const root=path.resolve(__dirname,'..'),python=path.resolve(root,'../experiments/2d-tracking/.venv/bin/python');
const data=fs.mkdtempSync(path.join(os.tmpdir(),'swing-review-qa-'));
const backup=spawnSync(python,['-c','import sqlite3,sys; a=sqlite3.connect("file:"+sys.argv[1]+"?mode=ro",uri=True); b=sqlite3.connect(sys.argv[2]); a.backup(b); b.close(); a.close()',path.join(root,'data/drafts.sqlite3'),path.join(data,'drafts.sqlite3')],{encoding:'utf8'});
if(backup.status!==0)throw new Error(backup.stderr);
// Reset only the disposable copy to imported baselines, so the test is independent
// of ongoing agent work or human decisions in the real database.
const reset=spawnSync(python,['-c',`import sqlite3,json,sys
db=sqlite3.connect(sys.argv[1])
for trial,idx,baseline,meta in db.execute('SELECT trial,idx,baseline,meta FROM frames').fetchall():
 m=json.loads(meta);r=json.loads(baseline)
 m.update(changed=False,passes={},acknowledged=[],humanEdited=False,uncertain=[],note='',locked=r.get('status')=='reviewed',version=0)
 db.execute('UPDATE frames SET record=?,meta=? WHERE trial=? AND idx=?',(baseline,json.dumps(m),trial,idx))
for table in ('events','requests','agent_reports'): db.execute('DELETE FROM '+table)
db.execute("UPDATE trials SET state='draft',revision=0,lease_owner=NULL,lease_token=NULL,lease_until=NULL")
db.commit();db.close()`,path.join(data,'drafts.sqlite3')],{encoding:'utf8'});
if(reset.status!==0)throw new Error(reset.stderr);
const server=spawn(python,['server.py','--port','0','--data',data],{cwd:root});
let serverError='';server.stderr.on('data',s=>serverError+=s);
(async()=>{
 let browser,page;
 try{
  const url=await new Promise((resolve,reject)=>{let output='';server.stdout.on('data',part=>{output+=part;const m=output.match(/http:\/\/127\.0\.0\.1:\d+/);if(m)resolve(m[0]);});server.on('exit',()=>reject(new Error(serverError)));setTimeout(()=>reject(new Error('Server did not start')),10000).unref();});
  browser=await chromium.launch({headless:true,executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'});
  page=await browser.newPage({viewport:{width:1500,height:1050},deviceScaleFactor:1});
  const errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.goto(url);await page.locator('#workspace').waitFor({state:'visible'});await page.locator('#image-loading').waitFor({state:'hidden',timeout:60000});
  assert.match(await page.locator('#frame-label').innerText(),/Frame 1 \/ 120/);
  const getTrial=()=>page.evaluate(async()=>{const b=await(await fetch('/api/bootstrap')).json();return await(await fetch('/api/trials/'+b.trials[0].id)).json();});
  const initial=await getTrial();const original=initial.frames[0].record.points[7];
  await page.locator('#lock').click();await page.waitForFunction(()=>document.querySelector('#lock').textContent.includes('Unlocked'));
  await page.locator('#joint').selectOption('left_elbow');
  await page.locator('#canvas').click({position:{x:300,y:200}});
  await page.waitForFunction(()=>document.querySelector('#save-state').textContent.includes('r2'));
  const edited=await getTrial();assert.notEqual(edited.frames[0].record.points[7].x,original.x);assert.deepEqual(edited.frames[0].baseline.points[7],original);
  await page.reload();await page.locator('#image-loading').waitFor({state:'hidden'});assert.equal((await getTrial()).revision,2);
  await page.locator('#undo').click();await page.waitForFunction(()=>document.querySelector('#save-state').textContent.includes('r3'));assert.deepEqual((await getTrial()).frames[0].record.points[7],original);
  await page.locator('#uncertain').check();await page.waitForFunction(()=>document.querySelector('#issues').textContent.includes('uncertain'));
  await page.locator('#uncertain').uncheck();await page.waitForFunction(()=>!document.querySelector('#issues').textContent.includes('uncertain'));
  await page.locator('#lock').click();await page.waitForFunction(()=>document.querySelector('#lock').textContent.includes('Locked'));
  await page.locator('#check').click();await page.waitForFunction(()=>document.querySelector('#frame-label').textContent.includes('Frame 2 /'));
  await page.locator('#jump').fill('71');await page.locator('#jump').press('Enter');await page.locator('#image-loading').waitFor({state:'hidden'});
  await page.locator('#baseline').check();await page.locator('#joint').selectOption('right_knee');
  await page.locator('#approve').click();assert.equal(await page.locator('#confirm-approve').isDisabled(),true);await page.locator('#cancel-approve').click();
  await page.locator('#play').click();await page.waitForFunction(()=>!document.querySelector('#frame-label').textContent.includes('Frame 71 /'));await page.locator('#play').click();
  await page.locator('#jump').fill('71');await page.locator('#jump').press('Enter');await page.locator('#image-loading').waitFor({state:'hidden'});
  fs.mkdirSync(path.join(root,'data/qa'),{recursive:true});await page.screenshot({path:path.join(root,'data/qa/desktop.png'),fullPage:true});
  await page.locator('#browse').click();await page.locator('.catalog-row').first().waitFor();assert.ok(await page.locator('.catalog-row').count()>0);await page.locator('#close-catalog').click();
  await page.setViewportSize({width:820,height:1180});await page.screenshot({path:path.join(root,'data/qa/tablet.png'),fullPage:true});
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
  const noUpload=await page.evaluate(async()=>{const r=await fetch('/api/trials/test/upload',{method:'POST',headers:{'Content-Type':'application/json','X-Swing-Review':'1'},body:'{}'});return r.status;});assert.equal(noUpload,409);
  const blocked=await page.evaluate(async()=>{const r=await fetch('/api/import',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});return r.status;});assert.equal(blocked,403);
  await page.evaluate(async()=>{
    const post=async(path,body)=>{const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json','X-Swing-Review':'1'},body:JSON.stringify(body)});const data=await r.json();if(!r.ok)throw Error(data.error);return data;};
    const b=await(await fetch('/api/bootstrap')).json(),id=b.trials[0].id,base='/api/trials/'+id;
    let t=await(await fetch(base)).json();
    await post(base+'/reopen',{actor:{type:'human',id:'test'},requestId:crypto.randomUUID(),expectedRevision:t.revision});
    const lease=await post(base+'/claim',{owner:'qa-agent'});t=await(await fetch(base)).json();
    await post(base+'/agent-report',{actor:{type:'agent',id:'qa-agent'},leaseToken:lease.leaseToken,expectedRevision:t.revision,requestId:crypto.randomUUID(),report:{trialId:id,sourceRevision:t.revision,summary:'Isolated UI test report',models:[{path:'test-only.pt',sha256:'a'.repeat(64)}],passes:[{name:'UI test',frames:[0],notes:'Synthetic suggestions in the disposable test database.'}],changes:[{frameIndex:0,joint:'left_elbow',point:{x:.45,y:.4,visibility:2},reason:'Test candidate'},{frameIndex:0,joint:'batTip',point:{x:.5,y:.3,visibility:2},reason:'Test candidate'}]}});
  });
  await page.reload();await page.locator('#jump').fill('1');await page.locator('#jump').press('Enter');await page.locator('#joint').selectOption('left_elbow');
  await page.locator('#agent-proposals').waitFor({state:'visible'});assert.match(await page.locator('#agent-heading').innerText(),/2 point suggestions/);
  assert.equal(await page.locator('#check').isDisabled(),true);
  await page.setViewportSize({width:1500,height:1050});await page.screenshot({path:path.join(root,'data/qa/agent-proposals.png'),fullPage:true});
  await page.locator('#accept-proposal').click();await page.waitForFunction(()=>document.querySelector('#agent-heading').textContent.includes('1 point suggestions'));
  let proposed=await getTrial();assert.equal(proposed.frames[0].record.points[7].x,.45);assert.equal(proposed.frames[0].meta.locked,true);assert.deepEqual(proposed.frames[0].baseline.points[7],original);
  await page.locator('#undo').click();await page.waitForFunction(()=>document.querySelector('#agent-heading').textContent.includes('2 point suggestions'));assert.deepEqual((await getTrial()).frames[0].record.points[7],original);
  await page.locator('#reject-frame').click();await page.locator('#agent-proposals').waitFor({state:'hidden'});assert.equal((await getTrial()).agentReports[0].changes.filter(c=>c.decision==='pending').length,0);
  assert.deepEqual(errors,[]);
  const result={passed:true,checks:['real frame decode','overlay edit','immutable baseline','persisted reload','undo','uncertainty flags','reference lock','review and advance','frame 71 navigation','approval gate','synchronized replay','catalog','tablet layout','upload disabled','write origin guard','agent proposal preview','accept locked proposal','undo accepted proposal','keep current frame','no browser exceptions'],screenshots:['data/qa/desktop.png','data/qa/tablet.png','data/qa/agent-proposals.png']};
  fs.writeFileSync(path.join(root,'data/qa/result.json'),JSON.stringify(result,null,2));console.log(JSON.stringify(result,null,2));
 }catch(e){if(page){fs.mkdirSync(path.join(root,'data/qa'),{recursive:true});await page.screenshot({path:path.join(root,'data/qa/failure.png'),fullPage:true});console.error(await page.locator('#toast').textContent());}throw e;
 }finally{if(browser)await browser.close();server.kill();if(server.exitCode===null)await new Promise(resolve=>server.once('exit',resolve));fs.rmSync(data,{recursive:true,force:true});}
})().catch(e=>{console.error(e);process.exitCode=1;});
