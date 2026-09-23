const $ = id => document.getElementById(id);
let policies = [], history = JSON.parse(localStorage.getItem("bipedalHistory") || "[]"), lastResult = null, torqueChart = null;

const views = {
  overview:["Overview","Inspect, evaluate, and compare your trained bipedal policies."],
  evaluate:["Evaluation Lab","Run a controlled episode and inspect the resulting video and metrics."],
  showdown:["Policy Showdown","Compare two checkpoints under the same environment seed."],
  stress:["Stress Testing","Probe robustness under gravity and friction changes."],
  analytics:["Torque Analytics","Inspect the four action channels returned by the environment."],
  curriculum:["Curriculum Explorer","Review the configured difficulty progression."],
  history:["Experiment History","Review evaluations saved locally in this browser."]
};

function go(view){
  document.querySelectorAll(".view").forEach(v=>v.classList.remove("active"));
  $("view-"+view).classList.add("active");
  document.querySelectorAll(".nav-item").forEach(n=>n.classList.toggle("active",n.dataset.view===view));
  $("pageTitle").textContent=views[view][0]; $("pageSubtitle").textContent=views[view][1];
}
document.addEventListener("click",e=>{
  const el=e.target.closest("[data-go]");
  if(el) go(el.dataset.go);
  const nav=e.target.closest("[data-view]");
  if(nav) go(nav.dataset.view);
});

async function api(path, options={}){
  const r=await fetch(path,{headers:{"Content-Type":"application/json"},...options});
  if(!r.ok){let detail="Request failed";try{detail=(await r.json()).detail||detail}catch{};throw new Error(detail)}
  return r.json();
}
function fillSelect(id){
  const s=$(id); s.innerHTML="";
  policies.forEach(p=>{const o=document.createElement("option");o.value=p;o.textContent=p.replaceAll("_"," ");s.appendChild(o)});
}
function fillAllSelects(){["evalPolicy","leftPolicy","rightPolicy","stressPolicy"].forEach(fillSelect);if(policies.length>1)$("rightPolicy").value=policies[1]}
function setRange(id,out){$(id).addEventListener("input",()=>$(out).textContent=Number($(id).value).toFixed(2))}
setRange("evalGravity","evalGravityOut");setRange("evalFriction","evalFrictionOut");setRange("stressGravity","stressGravityOut");setRange("stressFriction","stressFrictionOut");

async function health(){
  try{const h=await api("/health");$("healthDot").style.background="var(--good)";$("healthText").textContent="API healthy";$("mHealth").textContent="Online";$("mHealthSub").textContent=h.status||"healthy"}
  catch(e){$("healthDot").style.background="var(--bad)";$("healthText").textContent="API unavailable";$("mHealth").textContent="Offline";$("mHealthSub").textContent=e.message}
}
async function loadPolicies(){
  try{policies=await api("/api/checkpoints");$("mPolicies").textContent=policies.length;$("checkpointCount").textContent=policies.length+" models";fillAllSelects();
  $("checkpointList").classList.remove("empty");$("checkpointList").innerHTML=policies.map(p=>`<div class="checkpoint"><div><b>${escapeHtml(p.replaceAll("_"," "))}</b><small> · .pth checkpoint</small></div><button class="chip" data-policy="${escapeHtml(p)}">Use</button></div>`).join("");
  }catch(e){$("checkpointList").textContent=e.message}
}
$("checkpointList").addEventListener("click",e=>{const b=e.target.closest("[data-policy]");if(!b)return;$("evalPolicy").value=b.dataset.policy;go("evaluate")});
function escapeHtml(s){return String(s).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]))}

function metricsHtml(r){return [["Score",r.score],["Distance",r.distance],["Steps",r.steps],["Passed",r.passed?"Yes":"No"],["Fell",r.fell?"Yes":"No"],["Policy",r.name]].map(x=>`<div class="result-metric"><span>${x[0]}</span><strong>${escapeHtml(x[1])}</strong></div>`).join("")}
function saveRecord(type,request,result){
  const record={id:Date.now(),time:new Date().toLocaleString(),type,...request,result};
  history.unshift(record);history=history.slice(0,100);localStorage.setItem("bipedalHistory",JSON.stringify(history));renderHistory();updateOverview(result);
}
function updateOverview(r){lastResult=r;$("mScore").textContent=r.score;$("mScoreSub").textContent=r.passed?"Passed evaluation":"Evaluation completed";$("mDistance").textContent=r.distance;$("mDistanceSub").textContent=`${r.steps} steps`;$("mScore").style.color=r.passed?"var(--good)":"var(--text)";renderTorque(r.torques||[])}
function setMessage(id,msg,bad=false){$(id).textContent=msg;$(id).style.color=bad?"var(--bad)":"var(--muted)"}

$("runEval").onclick=async()=>{
  setMessage("evalMessage","Running episode...");
  $("runEval").disabled=true;
  try{
    const request={checkpoint:$("evalPolicy").value,gravity:+$("evalGravity").value,friction:+$("evalFriction").value,seed:+$("evalSeed").value};
    const r=await api("/api/stress",{method:"POST",body:JSON.stringify(request)});
    $("evalMetrics").innerHTML=metricsHtml(r);$("evalStatus").textContent=r.passed?"PASSED":"COMPLETED";$("evalStatus").className="badge "+(r.passed?"":"neutral");
    $("evalVideo").src=r.video_url;$("evalVideo").load();saveRecord("evaluation",request,r);setMessage("evalMessage","Evaluation completed successfully.");
  }catch(e){setMessage("evalMessage",e.message,true)}finally{$("runEval").disabled=false}
};
document.querySelectorAll("[data-preset]").forEach(b=>b.onclick=()=>{
  const p=b.dataset.preset;const vals={normal:[1,2.5],lowGravity:[.5,2.5],highGravity:[1.5,2.5],lowFriction:[1,.2]}[p];
  $("evalGravity").value=vals[0];$("evalFriction").value=vals[1];$("evalGravityOut").textContent=vals[0].toFixed(2);$("evalFrictionOut").textContent=vals[1].toFixed(2);
});

$("runStress").onclick=async()=>{
  setMessage("stressMessage","Running stress scenario...");$("runStress").disabled=true;
  try{
    const request={checkpoint:$("stressPolicy").value,gravity:+$("stressGravity").value,friction:+$("stressFriction").value,seed:+$("stressSeed").value};
    const r=await api("/api/stress",{method:"POST",body:JSON.stringify(request)});
    $("stressMetrics").innerHTML=metricsHtml(r);$("stressStatus").textContent=r.passed?"PASSED":"COMPLETED";$("stressVideo").src=r.video_url;$("stressVideo").load();saveRecord("stress",request,r);setMessage("stressMessage","Stress test completed.");
  }catch(e){setMessage("stressMessage",e.message,true)}finally{$("runStress").disabled=false}
};
document.querySelectorAll("[data-stress]").forEach(b=>b.onclick=()=>{
  const v={baseline:[1,2.5],heavy:[1.5,2.5],slippery:[1,.2],sticky:[1,3]}[b.dataset.stress];
  $("stressGravity").value=v[0];$("stressFriction").value=v[1];$("stressGravityOut").textContent=v[0].toFixed(2);$("stressFrictionOut").textContent=v[1].toFixed(2);
});

$("runShowdown").onclick=async()=>{
  setMessage("showMessage","Running both policies...");$("runShowdown").disabled=true;
  try{
    const request={left:$("leftPolicy").value,right:$("rightPolicy").value,seed:+$("showSeed").value,hardcore:$("hardcore").checked};
    const results=await api("/api/showdown",{method:"POST",body:JSON.stringify(request)});
    $("showResults").innerHTML=results.map((r,i)=>`<div class="show-card"><h3>${i===0?"Policy A":"Policy B"} · ${escapeHtml(r.name)}</h3><div class="result-metrics">${metricsHtml(r)}</div><video controls src="${r.video_url}"></video></div>`).join("");
    results.forEach(r=>saveRecord("showdown",request,r));setMessage("showMessage","Showdown completed.");
  }catch(e){setMessage("showMessage",e.message,true)}finally{$("runShowdown").disabled=false}
};

function renderTorque(torques){
  if(!torques.length){$("torqueStats").innerHTML="<div class='info-banner'>Run an evaluation to populate torque analytics.</div>";return}
  const labels=torques.map((_,i)=>i+1), datasets=[0,1,2,3].map(i=>({label:"Action "+(i+1),data:torques.map(t=>t[i]||0),borderWidth:1.5,pointRadius:0,tension:.2}));
  if(torqueChart)torqueChart.destroy();
  torqueChart=new Chart($("torqueChart"),{type:"line",data:{labels,datasets},options:{responsive:true,maintainAspectRatio:false,interaction:{mode:"index",intersect:false},plugins:{legend:{labels:{color:getComputedStyle(document.body).getPropertyValue("--text")}}},scales:{x:{ticks:{color:"#91a0bd"},grid:{color:"#273451"}},y:{ticks:{color:"#91a0bd"},grid:{color:"#273451"}}}}});
  $("torqueStats").innerHTML=[0,1,2,3].map(i=>{const a=torques.map(t=>Math.abs(t[i]||0));return `<div class="metric-card"><span>Action ${i+1} avg |action|</span><strong>${(a.reduce((x,y)=>x+y,0)/a.length).toFixed(3)}</strong><small>Peak ${Math.max(...a).toFixed(3)}</small></div>`}).join("");
}
$("downloadTorque").onclick=()=>{
  if(!lastResult?.torques?.length)return alert("Run an evaluation first.");
  const csv=["step,action_1,action_2,action_3,action_4",...lastResult.torques.map((t,i)=>[i+1,...t].join(","))].join("\n");
  const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([csv],{type:"text/csv"}));a.download="torque_analysis.csv";a.click();URL.revokeObjectURL(a.href);
};
function renderHistory(){
  const html=history.length?`<table><thead><tr><th>Time</th><th>Type</th><th>Policy</th><th>Score</th><th>Distance</th><th>Steps</th><th>Fall</th></tr></thead><tbody>${history.map(h=>`<tr><td>${escapeHtml(h.time)}</td><td>${h.type}</td><td>${escapeHtml(h.result.name)}</td><td>${h.result.score}</td><td>${h.result.distance}</td><td>${h.result.steps}</td><td>${h.result.fell?"Yes":"No"}</td></tr>`).join("")}</tbody></table>`:"<div class='empty'>No experiments saved yet.</div>";
  $("recentExperiments").innerHTML=html;$("fullHistory").innerHTML=html;
}
$("clearHistory").onclick=()=>{if(confirm("Clear all local experiment history?")){history=[];localStorage.removeItem("bipedalHistory");renderHistory()}};
$("exportHistory").onclick=()=>{
  const rows=["time,type,policy,score,distance,steps,fell",...history.map(h=>[h.time,h.type,h.result.name,h.result.score,h.result.distance,h.result.steps,h.result.fell].map(v=>`"${String(v).replaceAll('"','""')}"`).join(","))].join("\n");
  const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([rows],{type:"text/csv"}));a.download="experiment_history.csv";a.click();URL.revokeObjectURL(a.href);
};
$("refreshBtn").onclick=()=>{health();loadPolicies()};
$("themeToggle").onclick=()=>{document.body.classList.toggle("light");localStorage.setItem("bipedalTheme",document.body.classList.contains("light")?"light":"dark")};
if(localStorage.getItem("bipedalTheme")==="light")document.body.classList.add("light");
renderHistory();health();loadPolicies();