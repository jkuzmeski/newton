// SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
// SPDX-License-Identifier: Apache-2.0

// Inline into a saved replay by spring_viewer_report.py. Uses only SVG and DOM.
const springSvg=document.getElementById('springs'),svgNS='http://www.w3.org/2000/svg';let sd=d.springs;
const heatSvg=document.createElementNS(svgNS,'svg');
heatSvg.setAttribute('id','heatmap');heatSvg.setAttribute('viewBox','0 0 700 300');heatSvg.setAttribute('style','width:100%;max-width:700px;border:1px solid #ccd');
document.getElementById('heat').replaceWith(heatSvg);
const heatDots=d.anchors.map(p=>{const c=document.createElementNS(svgNS,'circle');c.setAttribute('cx',350+p[0]*2);c.setAttribute('cy',150-p[1]*2);c.setAttribute('r',3.4);heatSvg.appendChild(c);return c});
let yaw=-0.72,pitch=0.30,zoom=1,drag=null,overlayData=null,overlayRadius=0;
const springPaths=sd.rest_mm.map(()=>{const p=document.createElementNS(svgNS,'path');p.setAttribute('fill','none');springSvg.appendChild(p);return p});
let overlayPaths=[];
function radiusOf(payload){let r2=0;for(const endpoint of [payload.springs.bottom,payload.springs.top])for(const framePoints of endpoint)for(let k=0;k<framePoints.length;k+=3){const x=framePoints[k],y=framePoints[k+1],z=framePoints[k+2];r2=Math.max(r2,x*x+y*y+z*z)}return Math.sqrt(r2)/payload.springs.scale}
let baseRadius=radiusOf(d);
function drawHeat(){for(let j=0;j<heatDots.length;j++)heatDots[j].setAttribute('fill',heatColor(d.compression[frame][j]))}
function heatColor(mm){const v=Math.max(0,Math.min(1,mm/(d.heat_max||60)));let r=0,g=0,b=0;if(v<1/3){g=3*v;b=1}else if(v<2/3){r=3*v-1;g=1;b=2-3*v}else{r=1;g=3-3*v}return `rgb(${r*255},${g*255},${b*255})`}
function point(array,j){const k=3*j;return [array[k]/sd.scale,array[k+1]/sd.scale,array[k+2]/sd.scale]}
function project(p){const cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch);const x=cy*p[0]-sy*p[1],z=sy*p[0]+cy*p[1];return [x,cp*p[2]-sp*z,sp*p[2]+cp*z]}
function drawSprings(){
 const W=900,H=440,n=sd.rest_mm.length;
 // A rotation-invariant bound keeps framing fixed during playback and orbit.
 const extent=Math.max(baseRadius,overlayRadius);
 const sc=Math.min((W-50),(H-50))/(2*Math.max(extent,.001))*zoom,bottom=sd.bottom[frame],top=sd.top[frame];
 springSvg.setAttribute('viewBox',`0 0 ${W} ${H}`);
 for(let j=0;j<n;j++){
  const a=project(point(bottom,j)),b=project(point(top,j)),dx=(b[0]-a[0])*sc,dy=-(b[1]-a[1])*sc,len=Math.hypot(dx,dy);
  const x0=W/2+a[0]*sc,y0=H/2-a[1]*sc;
  if(len<.3){springPaths[j].setAttribute('d','');continue}
  const ux=dx/len,uy=dy/len,nx=-uy,ny=ux,amp=Math.max(.55,Math.min(2.0,sc*.0007));let path=`M${x0.toFixed(1)} ${y0.toFixed(1)}`;
  for(let q=1;q<=24;q++){const u=q/24,side=Math.sin(u*8*Math.PI)*amp;path+=` L${(x0+dx*u+nx*side).toFixed(1)} ${(y0+dy*u+ny*side).toFixed(1)}`}
  const el=springPaths[j];el.setAttribute('d',path);el.setAttribute('stroke',heatColor(d.compression[frame][j]));el.setAttribute('stroke-width',sd.driven[j]?'1.2':'.85');el.setAttribute('stroke-dasharray',sd.driven[j]?'':'2 2');
 }
 drawOverlaySprings(sc,W,H);
}
function drawOverlaySprings(sc,W,H){
 if(!overlayData?.springs){overlayPaths.forEach(p=>p.setAttribute('d',''));return}
 const od=overlayData.springs,oi=overlayFrame();if(oi<0){overlayPaths.forEach(p=>p.setAttribute('d',''));return}
 while(overlayPaths.length<od.rest_mm.length){const p=document.createElementNS(svgNS,'path');p.setAttribute('fill','none');p.setAttribute('stroke','#a23ac2');p.setAttribute('stroke-opacity','.4');p.setAttribute('stroke-dasharray','3 2');springSvg.appendChild(p);overlayPaths.push(p)}
 for(let j=0;j<overlayPaths.length;j++){
  const el=overlayPaths[j];if(j>=od.rest_mm.length){el.setAttribute('d','');continue}
  const a=project(point(od.bottom[oi],j)),b=project(point(od.top[oi],j)),dx=(b[0]-a[0])*sc,dy=-(b[1]-a[1])*sc,len=Math.hypot(dx,dy),x0=W/2+a[0]*sc,y0=H/2-a[1]*sc;
  if(len<.3){el.setAttribute('d','');continue}const ux=dx/len,uy=dy/len,nx=-uy,ny=ux,amp=Math.max(.55,Math.min(2,sc*.0007));let path=`M${x0.toFixed(1)} ${y0.toFixed(1)}`;
  for(let q=1;q<=24;q++){const u=q/24,side=Math.sin(u*8*Math.PI)*amp;path+=` L${(x0+dx*u+nx*side).toFixed(1)} ${(y0+dy*u+ny*side).toFixed(1)}`}el.setAttribute('d',path);el.setAttribute('stroke-width',od.driven[j]?'1.1':'.8');
 }
}
function overlayFrame(){if(!overlayData?.times?.length)return -1;const t=d.times[frame]-d.times[0]+overlayData.times[0];if(t<overlayData.times[0]||t>overlayData.times.at(-1))return -1;let lo=0,hi=overlayData.times.length-1;while(lo<hi){const mid=(lo+hi)>>1;if(overlayData.times[mid]<t)lo=mid+1;else hi=mid}if(lo&&Math.abs(overlayData.times[lo-1]-t)<Math.abs(overlayData.times[lo]-t))lo--;return lo}
function drawPoseOverlay(){const old=document.getElementById('replay-overlay-pose');if(old)old.remove();const oi=overlayFrame();if(oi<0||!overlayData?.scene)return;const g=document.createElementNS(svgNS,'g');g.id='replay-overlay-pose';g.setAttribute('opacity','.42');g.innerHTML=overlayData.scene[oi];g.querySelectorAll('path,line,polyline,polygon,circle,rect').forEach(el=>{el.setAttribute('stroke','#a23ac2');el.setAttribute('stroke-dasharray','4 3');if(el.tagName==='path'||el.tagName==='polygon')el.setAttribute('fill','none')});pose.appendChild(g)}
function setView(name){if(name==='side'){yaw=0;pitch=0}else if(name==='top'){yaw=0;pitch=Math.PI/2}else if(name==='end'){yaw=Math.PI/2;pitch=0}drawSprings()}
document.querySelectorAll('[data-view]').forEach(b=>b.addEventListener('click',()=>setView(b.dataset.view)));
document.getElementById('spring-zoom-in').onclick=()=>{zoom=Math.min(zoom*1.25,8);drawSprings()};
document.getElementById('spring-zoom-out').onclick=()=>{zoom=Math.max(zoom/1.25,.25);drawSprings()};
function resetView(){yaw=-.72;pitch=.30;zoom=1;drawSprings()}
document.getElementById('spring-reset').onclick=resetView;
springSvg.addEventListener('pointerdown',e=>{drag={x:e.clientX,y:e.clientY,id:e.pointerId};springSvg.setPointerCapture(e.pointerId);springSvg.focus()});
springSvg.addEventListener('pointermove',e=>{if(!drag)return;const dx=e.clientX-drag.x,dy=e.clientY-drag.y;drag={x:e.clientX,y:e.clientY,id:e.pointerId};yaw+=dx*.009;pitch=Math.max(-Math.PI/2,Math.min(Math.PI/2,pitch+dy*.009));drawSprings()});
springSvg.addEventListener('pointerup',()=>{drag=null});springSvg.addEventListener('pointercancel',()=>{drag=null});
springSvg.addEventListener('wheel',e=>{e.preventDefault();zoom=Math.max(.25,Math.min(8,zoom*(e.deltaY<0?1.12:1/1.12)));drawSprings()},{passive:false});
springSvg.addEventListener('keydown',e=>{const step=.10;if(e.key==='ArrowLeft')yaw-=step;else if(e.key==='ArrowRight')yaw+=step;else if(e.key==='ArrowUp')pitch=Math.min(Math.PI/2,pitch+step);else if(e.key==='ArrowDown')pitch=Math.max(-Math.PI/2,pitch-step);else if(e.key==='+'||e.key==='=')zoom=Math.min(8,zoom*1.2);else if(e.key==='-')zoom=Math.max(.25,zoom/1.2);else return;e.preventDefault();drawSprings()});
function setOverlay(payload){overlayData=payload;overlayRadius=payload?.springs?radiusOf(payload):0;const key=document.getElementById('comparison-key');key.hidden=!payload;key.textContent=payload?`Purple dashed: ${payload.label||'comparison shoe'} (nearest matching elapsed time; centered on shoe)`:'';drawSprings();drawPoseOverlay()}
window.replay=Object.assign(window.replay||{}, {ready:true,getData:()=>d,getFrame:()=>frame,getCamera:()=>({yaw,pitch,zoom}),setFrame:i=>{frame=Math.max(0,Math.min(d.times.length-1,Math.round(i)));clock=d.times[frame];draw()},setData:payload=>{if(!payload?.springs)throw new TypeError('Replay payload must include spring geometry');d=payload;sd=payload.springs;baseRadius=radiusOf(d);frame=0;draw()},setOverlay});
window.addEventListener('message',event=>{
 if(event.source!==window.parent)return;
 const message=event.data||{};
 if(message.type==='replay:request')window.parent.postMessage({type:'replay:data',payload:window.replay.getData()},'*');
 else if(message.type==='replay:overlay')window.replay.setOverlay(message.payload||null);
 else if(message.type==='replay:camera'&&message.payload){const c=message.payload;if(Number.isFinite(c.yaw))yaw=c.yaw;if(Number.isFinite(c.pitch))pitch=Math.max(-Math.PI/2,Math.min(Math.PI/2,c.pitch));if(Number.isFinite(c.zoom))zoom=Math.max(.25,Math.min(8,c.zoom));drawSprings()}
});
if(window.parent!==window)window.parent.postMessage({type:'replay:ready',height:Math.ceil(document.body.getBoundingClientRect().height+24)},'*');
