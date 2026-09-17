# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Render the single-leg experiment using recorded motion and force comparisons."""

from __future__ import annotations

import base64
import html
import json
from pathlib import Path

import numpy as np


def _plain(value):
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _spring_payload(springs: dict) -> dict:
    """Pack view snapshots as little-endian float32 blocks for offline playback."""
    if not springs.get("available"):
        return springs
    result = dict(springs)
    for key in ("bottom_m", "top_m", "compression_m"):
        array = np.asarray(result[key], dtype="<f4")
        result[key] = {"shape": array.shape, "f32le": base64.b64encode(array.tobytes()).decode("ascii")}
    return result


def _ground_height(summary: dict) -> float | None:
    """Return the saved per-condition ground height when it is available."""
    candidates = [summary.get("ground_height_m"), summary.get("shoe", {}).get("ground_height_m")]
    for candidate in candidates:
        try:
            value = float(candidate)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            return value
    return None


def write_report(directory, reference, trace, summary, *, profile=None, include_springs: bool = True):
    """Write a saved one-leg replay without inferred-output fitting panels.

    Args:
        directory: Report destination.
        reference: Saved recorded inputs.
        trace: Saved simulated states and contact outputs.
        summary: Saved numerical results and provenance.
        profile: Fixed body and controller settings.
        include_springs: Load or reconstruct audited spring history. False renders
            meshes only and never advances contact, including for GPU results.
    """
    from .rendering import load_geometry  # noqa: PLC0415
    from .springs import load_springs  # noqa: PLC0415

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    run = summary.get("run", summary)
    complete = run.get("status") == "completed" and not run.get("failure")
    title = "Single-leg rollout completed" if complete else "Single-leg rollout stopped"
    geometry = load_geometry(summary.get("shoe", {}))
    springs = (
        load_springs(directory, reference, trace, summary, profile or {})
        if include_springs
        else {"available": False, "reason": "Mesh-only report requested; no contact replay or leg simulation was run."}
    )
    payload = _plain(
        {
            "reference": reference,
            "trace": trace,
            "summary": summary,
            "profile": profile or {},
            "geometry": geometry,
            "springs": _spring_payload(springs),
            "ground_height_m": _ground_height(summary),
            "rendering_info": {
                "foot_representation": "actual rigid fullfoot_last triangles, no ankle-to-marker foot stick",
                "physics_changed": False,
                "midsole_representation": "selectable undeformed CAD mesh, solved spring endpoints, or both",
                "spring_representation": "verified contact-history replay of saved simulated states, not a new leg rollout",
            },
        }
    )
    encoded = json.dumps(payload, allow_nan=False).replace("<", "\\u003c").replace("&", "\\u0026")
    metrics = summary.get("metrics", {})
    cards = [f"{key}: {value}" for key, value in metrics.items()]
    note = "No trunk, second leg, hip torque, or upper-body weight. Hip Cartesian spring + knee/ankle springs; one shoe supplies GRF."
    document = _PAGE.replace("__DATA__", encoded).replace("__TITLE__", html.escape(title))
    document = document.replace("__NOTE__", html.escape(note)).replace("__METRICS__", html.escape("\n".join(cards)))
    document = document.replace("__SUMMARY__", html.escape(json.dumps(_plain(summary), indent=2, allow_nan=False)))
    shoe_map_source = Path(__file__).with_name("shoe_map.js").read_text(encoding="utf-8")
    document = document.replace("__SHOE_MAP__", shoe_map_source)
    path = directory / "report.html"
    path.write_text(document)
    return path


_PAGE = r"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Cartesian hip — single-leg stance</title><style>
body{font:15px system-ui,sans-serif;background:#f5f7f8;color:#223641;margin:0}main{max-width:1200px;margin:auto;padding:24px}h1{font-size:26px}h2{font-size:18px}.note,.card{background:white;border:1px solid #d7e0e4;border-radius:8px;padding:16px;margin:12px 0}.note{border-left:5px solid #bd8630}canvas{width:100%;height:430px;background:#fff;border:1px solid #d7e0e4}.replay-grid{display:grid;grid-template-columns:minmax(0,1.65fr) minmax(0,1fr);gap:16px}.foot-card{min-width:0;background:white;padding:12px;border:1px solid #d7e0e4}.foot-card h2{margin:0 0 8px}#foot_view{height:310px}@media(max-width:800px){.replay-grid{grid-template-columns:1fr}}input[type=range]{width:65%}#spring_slice{width:110px;vertical-align:middle}select{padding:6px;margin:4px 12px 4px 3px}#spring_controls label{white-space:nowrap;font-size:13px}.color-ramp{height:12px;max-width:360px;background:linear-gradient(to right,#0000ff 0%,#00ffff 33.333%,#ffff00 66.667%,#ff0000 100%);border:1px solid #a2b1b9;margin-top:6px}.ramp-ticks{display:flex;justify-content:space-between;max-width:360px;font-size:11px}#compression_title{font-size:12px}#spring_map{height:245px;border:0}button{padding:7px 15px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:12px}.card svg{width:100%;height:auto}svg text{font:11px system-ui;fill:#536774}[hidden]{display:none!important}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:12px ui-monospace,monospace}.legend{font-size:12px;color:#526975}details{margin-top:18px}</style>
<main><h1 id="report_title">Cartesian hip + knee/ankle impedance</h1><div class="note" id="report_note"><b>__TITLE__ — experimental, not a validated running fit.</b><p>__NOTE__</p></div>
<div class="card"><label>Shoe view <select id="shoe_view"><option value="mesh">Mesh</option><option value="springs">Springs + deformation heat map</option><option value="both" selected>Both</option></select></label>
<span id="spring_controls"> <label>Color <select id="spring_metric"><option value="mm">Compression [mm]</option><option value="strain">Compression / rest length [%]</option></select></label> <label>Side-view width slice <input id="spring_slice" type="range" min="0" max="1" value="0"> <span id="slice_label"></span></label> <label><input id="all_springs" type="checkbox"> All columns (side-view overlap)</label></span>
<p class="legend" id="spring_status"></p><p class="legend" id="condition_status"></p><div id="compression_legend"><span id="compression_title"></span><div class="color-ramp"></div><div class="ramp-ticks" id="compression_ticks"></div></div></div>
<div class="replay-grid"><div><canvas id="view"></canvas></div><div class="foot-card"><h2>Ankle-last attachment</h2><canvas id="foot_view"></canvas><p class="legend">The shank connects at the ankle hinge. The short connector shows the existing rigid offset to the last; it is not a new joint or force path.</p></div></div>
<p><button id="play">Play</button> <input id="slider" type="range" min="0" max="1000" value="0"> <span id="clock"></span></p>
<p><label><input id="show_reference" type="checkbox" checked> Recorded-reference leg</label> <label><input id="show_markers" type="checkbox"> Recorded foot markers</label></p>
<p class="legend" id="native_legend">Blue: shank/thigh · gold: rigid Instron last (foot geometry) · green: undeformed midsole · gray: reference · purple cross/arrow: hip equilibrium/applied force · orange: resultant GRF. No marker-endpoint stick is drawn as the physical foot. The last and its attachment stay unchanged. Springs use the existing carried-column endpoint rule and simulated compression. In Both mode, the translucent midsole is undeformed context, not a deformed mesh. No new last-surface contact is enabled.</p>
<div class="card" id="spring_map_card"><h2>Whole-shoe deformation map</h2><canvas id="spring_map"></canvas><p class="legend" id="spring_hover">Hover over a column for its compression and rest length.</p><p class="legend">All columns, in the fixed shoe-local layout. Squares: driven footprint. Circles: passive surround. The outlined band is the side-view slice; click the map to move it. Color is the solved foundation compression, not force or a fit target.</p></div>
<div class="grid" id="plots"></div><details id="metrics_details" open><summary>Recorded-output metrics</summary><pre>__METRICS__</pre></details><details id="summary_details"><summary>Full experiment settings and numerical diagnostics</summary><pre>__SUMMARY__</pre></details></main>
<script>
__SHOE_MAP__
const data=__DATA__;let r=data.reference,tr=data.trace;const s=data.summary,rt=r.time_s,tt=tr.time_s||[],duration=rt.at(-1),svgNS="http://www.w3.org/2000/svg";
function idx(t,arr){let lo=0,hi=arr.length-1;while(lo<hi){const mid=(lo+hi+1)>>1;if(arr[mid]<=t)lo=mid;else hi=mid-1;}return lo;}
function joints(q){const a=q[2],b=a+q[3],L=r.lengths_m,p=[[q[0],q[1]]];p.push([p[0][0]+L[0]*Math.cos(a),p[0][1]+L[0]*Math.sin(a)]);p.push([p[1][0]+L[1]*Math.cos(b),p[1][1]+L[1]*Math.sin(b)]);return p;}
const canvas=document.getElementById("view"),ctx=canvas.getContext("2d"),footCanvas=document.getElementById("foot_view"),footCtx=footCanvas.getContext("2d");let geometry=data.geometry||{},meshCache={};
function makeMeshCache(mesh,color){const vertices=mesh.vertices,pp=6000,pad=3;let xmin=Infinity,xmax=-Infinity,zmin=Infinity,zmax=-Infinity;for(const v of vertices){xmin=Math.min(xmin,v[0]);xmax=Math.max(xmax,v[0]);zmin=Math.min(zmin,v[2]);zmax=Math.max(zmax,v[2]);}const image=document.createElement("canvas");image.width=Math.ceil((xmax-xmin)*pp)+2*pad;image.height=Math.ceil((zmax-zmin)*pp)+2*pad;const c=image.getContext("2d");for(let f=0;f<mesh.triangles.length;f++){const face=mesh.triangles[f],shade=mesh.shade[f];c.fillStyle=`rgb(${color.map(v=>Math.round(v*shade)).join(",")})`;c.beginPath();for(let k=0;k<3;k++){const v=vertices[face[k]],x=pad+(v[0]-xmin)*pp,y=pad+(zmax-v[2])*pp;k?c.lineTo(x,y):c.moveTo(x,y);}c.closePath();c.fill();}return {image,xmin,xmax,zmin,zmax,pp,pad};}
if(geometry.available){meshCache.last=makeMeshCache(geometry.last,[231,178,112]);meshCache.midsole=makeMeshCache(geometry.midsole,[92,159,147]);}

let springs=data.springs||{};const viewSelect=document.getElementById("shoe_view"),metricSelect=document.getElementById("spring_metric"),sliceSelect=document.getElementById("spring_slice"),allSprings=document.getElementById("all_springs"),mapCanvas=document.getElementById("spring_map");
let springBottom,springTop,springCompression,springRows=[],sliceColumns=[],depthColumns=[],mapProjection=null,hoveredColumn=null;
let externalHeatmaps=false,compressionMetric="mm",compactMode=false;
function unpackFloat(block){return NativeShoeMap.unpack(block);}
if(springs.available){springBottom=unpackFloat(springs.bottom_m);springTop=unpackFloat(springs.top_m);springCompression=unpackFloat(springs.compression_m);springRows=[...new Set(springs.anchor_local_m.map(p=>p[1].toFixed(8)))].map(Number).sort((a,b)=>a-b);sliceSelect.max=springRows.length-1;sliceSelect.value=springRows.reduce((best,y,i)=>Math.abs(y)<Math.abs(springRows[best])?i:best,0);depthColumns=springs.anchor_local_m.map((_,i)=>i).sort((a,b)=>springs.anchor_local_m[a][1]-springs.anchor_local_m[b][1]);}else{viewSelect.value="mesh";for(const option of viewSelect.options)if(option.value!=="mesh")option.disabled=true;document.getElementById("spring_controls").hidden=true;}
function springColor(value){return NativeShoeMap.color(value);}
function colorValue(frame,column){const compression=springCompression[frame*springs.rest_length_m.length+column];return metricSelect.value==="strain"?compression/springs.rest_length_m[column]:compression*1000/(springs._activeColorMaxMm||springs.color_max_mm);}
function refreshSpringControls(){
  const active=springs.available&&viewSelect.value!=="mesh";
  document.getElementById("compression_legend").hidden=!active||externalHeatmaps;
  document.getElementById("spring_map_card").hidden=!active||externalHeatmaps;
  if(metricSelect.parentElement)metricSelect.parentElement.hidden=externalHeatmaps;
  if(!springs.available){document.getElementById("spring_status").textContent=springs.reason||"Per-column history is unavailable; mesh view only.";return;}
  const y=springRows.length?springRows[Math.min(Number(sliceSelect.value)||0,springRows.length-1)]:0;
  sliceColumns=depthColumns.filter(i=>Math.abs(springs.anchor_local_m[i][1]-y)<springs.spacing_m*.45);
  document.getElementById("slice_label").textContent=(y*1000).toFixed(1)+" mm";
  document.getElementById("compression_title").textContent=metricSelect.value==="strain"?"Foundation compression / original rest length [%] — fixed 0-100%":"Foundation compression [mm] — one fixed scale for the whole rollout";
  const max=metricSelect.value==="strain"?100:(springs._activeColorMaxMm||springs.color_max_mm);
  document.getElementById("compression_ticks").replaceChildren(...[0,1/3,2/3,1].map(v=>{const e=document.createElement("span");e.textContent=(max*v).toFixed(1);return e;}));
  document.getElementById("spring_status").textContent=`${springs.rest_length_m.length} columns; ${allSprings.checked?springs.rest_length_m.length:sliceColumns.length} drawn in the side view. Springs and pose use the same saved step. Contact replay is checked against saved forces and moment; no leg dynamics or fitting rerun.`;
}
function springFrame(t){return springs.available?idx(t,springs.time_s):0;}
function springPoints(frame){if(!springs.available||viewSelect.value==="mesh")return [];const result=[],count=springs.rest_length_m.length;for(const i of (allSprings.checked?depthColumns:sliceColumns)){const k=(frame*count+i)*3;result.push([springBottom[k],springBottom[k+2]],[springTop[k],springTop[k+2]]);}return result;}
function drawSprings(context,v,frame){if(!springs.available||viewSelect.value==="mesh")return;const count=springs.rest_length_m.length;context.save();context.lineCap="round";context.lineJoin="round";context.lineWidth=Math.max(.8,Math.min(1.6,v.scale*springs.spacing_m*.15));for(const i of (allSprings.checked?depthColumns:sliceColumns)){const k=(frame*count+i)*3,a=v.xy([springBottom[k],springBottom[k+2]]),b=v.xy([springTop[k],springTop[k+2]]),dx=b[0]-a[0],dy=b[1]-a[1],length=Math.hypot(dx,dy),amp=Math.min(springs.spacing_m*v.scale*.26,length*.13,3);context.strokeStyle=springColor(colorValue(frame,i));context.beginPath();context.moveTo(...a);if(length>2){context.lineTo(a[0]+dx*.13,a[1]+dy*.13);for(let j=0;j<6;j++){const u=.2+j*.12,offset=(j%2?1:-1)*amp;context.lineTo(a[0]+dx*u-dy/length*offset,a[1]+dy*u+dx/length*offset);}context.lineTo(a[0]+dx*.87,a[1]+dy*.87);}context.lineTo(...b);context.stroke();}context.restore();}
function drawSpringMap(frame){
  if(!springs.available||viewSelect.value==="mesh"||externalHeatmaps)return;
  const max=metricSelect.value==="strain"?1:(springs._activeColorMaxMm||springs.color_max_mm);
  mapProjection=NativeShoeMap.draw(mapCanvas,{
    anchor_local_m:springs.anchor_local_m,rest_length_m:springs.rest_length_m,
    driven:springs.driven,spacing_m:springs.spacing_m,mount_m:geometry.mount_m,
    compression:springCompression,time_s:springs.time_s,label:"Saved spring frame: "+springs.time_s[frame].toFixed(3)+" s"
  },{frame,metric:metricSelect.value,mm_max:max,height:245,
    slice_y_m:springRows.length?springRows[Number(sliceSelect.value)]+geometry.mount_m[1]:null});
}
function updateColumnLabel(frame){if(hoveredColumn===null)return;const i=hoveredColumn,compression=springCompression[frame*springs.rest_length_m.length+i],rest=springs.rest_length_m[i];document.getElementById("spring_hover").textContent=`Column ${i} (${springs.driven[i]?"driven":"passive"}): compression ${(compression*1000).toFixed(2)} mm / ${(compression/rest*100).toFixed(1)}%; original rest length ${(rest*1000).toFixed(2)} mm (saved frame ${springs.time_s[frame].toFixed(3)} s).`;}
mapCanvas.onmouseleave=()=>{hoveredColumn=null;document.getElementById("spring_hover").textContent="Hover over a column for its compression and rest length.";};
mapCanvas.onmousemove=e=>{if(!mapProjection)return;const rect=mapCanvas.getBoundingClientRect(),x=(e.clientX-rect.left-mapProjection.ox)/mapProjection.scale-mapProjection.offsetX,y=(mapProjection.oy-e.clientY+rect.top)/mapProjection.scale-mapProjection.offsetY;let best=0,distance=Infinity;for(let i=0;i<springs.anchor_local_m.length;i++){const a=springs.anchor_local_m[i],d=Math.hypot(a[0]-x,a[1]-y);if(d<distance){distance=d;best=i;}}const label=document.getElementById("spring_hover");if(distance>springs.spacing_m*.75){hoveredColumn=null;label.textContent="Hover over a column for its compression and rest length.";return;}hoveredColumn=best;updateColumnLabel(mapProjection.frame);};
mapCanvas.onclick=e=>{if(!mapProjection)return;const y=(mapProjection.oy-(e.clientY-mapCanvas.getBoundingClientRect().top))/mapProjection.scale-mapProjection.offsetY;sliceSelect.value=springRows.reduce((best,v,i)=>Math.abs(v-y)<Math.abs(springRows[best]-y)?i:best,0);refreshSpringControls();draw(time);};for(const control of [viewSelect,metricSelect,sliceSelect,allSprings])control.oninput=()=>{refreshSpringControls();draw(time);};refreshSpringControls();

function place(v,ankle,a){const c=Math.cos(a),sn=Math.sin(a);return [ankle[0]+c*v[0]-sn*v[2],ankle[1]+sn*v[0]+c*v[2]];}
function angle(q){return q[2]+q[3]+Math.PI/2+q[4]-(geometry.static_pitch_rad||0);}
function boundsPoints(p,q){if(!geometry.available)return [];const result=[],a=angle(q);for(const key of ["midsole","last"]){const g=meshCache[key];for(const x of [g.xmin,g.xmax])for(const z of [g.zmin,g.zmax])result.push(place([x,0,z],p[2],a));}return result;}
function setup(c,height){const ratio=devicePixelRatio||1,w=c.clientWidth;c.width=w*ratio;c.height=height*ratio;const context=c.getContext("2d");context.setTransform(ratio,0,0,ratio,0,0);context.clearRect(0,0,w,height);return [context,w,height];}
function projection(points,w,h,pad){const xmin=Math.min(...points.map(p=>p[0]))-pad,xmax=Math.max(...points.map(p=>p[0]))+pad,ymin=Math.min(0,...points.map(p=>p[1]))-pad,ymax=Math.max(...points.map(p=>p[1]))+pad,scale=Math.min((w-40)/(xmax-xmin),(h-45)/(ymax-ymin)),ox=w/2-(xmin+xmax)*scale/2,oy=h-20+ymin*scale;return {scale,oy,xy:p=>[ox+scale*p[0],oy-scale*p[1]]};}
function ground(context,w,v,height=0){const y=v.xy([0,height])[1];context.strokeStyle="#7b9d86";context.lineWidth=1;context.beginPath();context.moveTo(0,y);context.lineTo(w,y);context.stroke();}
function chain(context,p,v,color,dashed=false){context.strokeStyle=color;context.fillStyle=color;context.lineWidth=dashed?2:4;context.setLineDash(dashed?[6,5]:[]);context.beginPath();p.forEach((point,i)=>{const z=v.xy(point);i?context.lineTo(...z):context.moveTo(...z);});context.stroke();context.setLineDash([]);for(const point of p.slice(0,2)){context.beginPath();context.arc(...v.xy(point),4,0,Math.PI*2);context.fill();}}
function meshes(context,p,q,v,alpha=1,mode=viewSelect.value){if(!geometry.available)return;const a=angle(q),ankle=v.xy(p[2]);context.save();context.translate(...ankle);context.rotate(-a);for(const key of ["last","midsole"]){if(key==="midsole"&&mode!=="mesh"&&mode!=="both")continue;context.globalAlpha=alpha*(key==="midsole"&&mode==="both"?.18:1);const g=meshCache[key],unit=v.scale/g.pp;context.drawImage(g.image,g.xmin*v.scale-g.pad*unit,-g.zmax*v.scale-g.pad*unit,g.image.width*unit,g.image.height*unit);}context.restore();}
function mounting(context,p,q,v,label=false){if(!geometry.available)return;const ankle=v.xy(p[2]),mount=v.xy(place(geometry.mount_connector_local_m,p[2],angle(q)));context.strokeStyle="#506373";context.lineWidth=5;context.beginPath();context.moveTo(...ankle);context.lineTo(...mount);context.stroke();context.strokeStyle="#233b4a";context.fillStyle="white";context.lineWidth=2;context.beginPath();context.arc(...ankle,label?7:5,0,Math.PI*2);context.fill();context.stroke();context.fillStyle="#176b89";context.beginPath();context.arc(...ankle,2,0,Math.PI*2);context.fill();if(label){context.fillStyle="#233b4a";context.font="12px system-ui";context.fillText("Ankle hinge",Math.max(8,ankle[0]-45),ankle[1]-16);}}
function markers(context,ri,v){if(!document.getElementById("show_markers").checked||!r.foot_marker_target_m)return;context.fillStyle="#b3387d";for(const p of r.foot_marker_target_m[ri]){const z=v.xy(p);context.fillRect(z[0]-2,z[1]-2,4,4);}}
function arrow(context,p,f,v,color){if(!f)return;const a=v.xy(p),b=v.xy([p[0]+f[0]*.0001,p[1]+f[1]*.0001]),ang=Math.atan2(b[1]-a[1],b[0]-a[0]);context.strokeStyle=color;context.fillStyle=color;context.lineWidth=2;context.beginPath();context.moveTo(...a);context.lineTo(...b);context.stroke();context.beginPath();context.moveTo(...b);context.lineTo(b[0]-8*Math.cos(ang-.5),b[1]-8*Math.sin(ang-.5));context.lineTo(b[0]-8*Math.cos(ang+.5),b[1]-8*Math.sin(ang+.5));context.closePath();context.fill();}
function draw(t){if(hashExperiment&&!tt.length){setup(canvas,430);setup(footCanvas,310);setup(mapCanvas,245);return;}const [context,w,h]=setup(canvas,430),sf=springFrame(Math.min(t,tt.at(-1)||0)),ti=springs.available?springs.trace_index[sf]:(tt.length?idx(Math.min(t,tt.at(-1)),tt):0),ri=idx(springs.available?springs.time_s[sf]:t,rt),q=tt.length?tr.state[ti]:null,ref=joints(r.state[ri]),actual=q?joints(q):null,eq=tr.equilibrium&&tr.equilibrium[ti],showRef=document.getElementById("show_reference").checked;let points=(actual||ref).concat(boundsPoints(actual||ref,q||r.state[ri]),springPoints(sf));if(showRef)points=points.concat(ref,boundsPoints(ref,r.state[ri]));if(eq)points.push(eq.slice(0,2));const v=projection(points,w,h,.1);ground(context,w,v,Number(s.ground_height_m)||0);if(showRef){meshes(context,ref,r.state[ri],v,.18,"last");chain(context,ref,v,"#a1adb3",true);}if(actual){meshes(context,actual,q,v);drawSprings(context,v,sf);chain(context,actual,v,"#176b89");mounting(context,actual,q,v);}else if(!showRef&&!hashExperiment){meshes(context,ref,r.state[ri],v);chain(context,ref,v,"#a1adb3",true);}markers(context,ri,v);
if(eq&&actual){const e=v.xy(eq.slice(0,2)),p=v.xy(actual[0]);context.strokeStyle="#874daf";context.lineWidth=2;context.beginPath();context.moveTo(e[0]-6,e[1]);context.lineTo(e[0]+6,e[1]);context.moveTo(e[0],e[1]-6);context.lineTo(e[0],e[1]+6);context.stroke();context.setLineDash([3,3]);context.beginPath();context.moveTo(...p);context.lineTo(...e);context.stroke();context.setLineDash([]);}
if(actual){arrow(context,actual[0],tr.hip_force_n&&tr.hip_force_n[ti],v,"#874daf");const f=tr.grf_n&&tr.grf_n[ti],m=tr.ankle_contact_moment_nm&&tr.ankle_contact_moment_nm[ti];let cop=actual[2][0];if(f&&f[1]>5&&Number.isFinite(m))cop+=(m-actual[2][1]*f[0])/f[1];arrow(context,[cop,0],f,v,"#c36b2d");}context.fillStyle="#536774";context.font="12px system-ui";context.fillText("y: up | x: forward",15,22);if(tt.length&&t>tt.at(-1)+.001)context.fillText("Simulation held at last saved state",15,40);if(!geometry.available)context.fillText(geometry.reason||"Last mesh unavailable",15,60);
const [detail,dw,dh]=setup(footCanvas,310),dp=actual||ref,dq=q||r.state[ri],ankle=dp[2],knee=dp[1],length=Math.hypot(knee[0]-ankle[0],knee[1]-ankle[1]),shortShank=[ankle[0]+.09*(knee[0]-ankle[0])/length,ankle[1]+.09*(knee[1]-ankle[1])/length],detailPoints=[ankle,shortShank,...boundsPoints(dp,dq),...springPoints(sf)],dv=projection(detailPoints,dw,dh,.025);ground(detail,dw,dv,Number(s.ground_height_m)||0);meshes(detail,dp,dq,dv);drawSprings(detail,dv,sf);detail.strokeStyle="#176b89";detail.lineWidth=6;detail.beginPath();detail.moveTo(...dv.xy(shortShank));detail.lineTo(...dv.xy(ankle));detail.stroke();mounting(detail,dp,dq,dv,true);markers(detail,ri,dv);detail.fillStyle="#95642b";detail.font="12px system-ui";detail.fillText("Gold: fullfoot last mesh",10,dh-22);detail.fillStyle="#467d73";detail.fillText(viewSelect.value==="mesh"?"Green: undeformed midsole mesh":"Spring colors: solved foundation compression",10,dh-7);drawSpringMap(sf);updateColumnLabel(sf);document.getElementById("clock").textContent=t.toFixed(3)+" / "+duration.toFixed(3)+" s"+(springs.available?" | saved frame "+springs.time_s[sf].toFixed(3)+" s":"");}
document.getElementById("show_reference").onchange=()=>draw(time);document.getElementById("show_markers").onchange=()=>draw(time);
function el(name,attrs,text){const e=document.createElementNS(svgNS,name);for(const [k,v]of Object.entries(attrs||{}))e.setAttribute(k,v);if(text!==undefined)e.textContent=text;return e;}
function plot(title,unit,curves){const card=document.createElement("div");card.className="card";const h=document.createElement("h2");h.textContent=title;card.append(h);const svg=el("svg",{viewBox:"0 0 560 245"}),values=curves.flatMap(c=>c.v).filter(Number.isFinite);let lo=values.length?Math.min(...values):0,hi=values.length?Math.max(...values):1,pad=Math.max((hi-lo)*.1,.001);lo-=pad;hi+=pad;const X=t=>60+480*t/duration,Y=v=>205-175*(v-lo)/(hi-lo);for(let i=0;i<5;i++){const v=lo+(hi-lo)*i/4,t=duration*i/4;svg.append(el("path",{d:`M60 ${Y(v)}H540`,stroke:"#e0e6e8"}),el("text",{x:54,y:Y(v)+4,"text-anchor":"end"},v.toPrecision(3)),el("text",{x:X(t),y:225,"text-anchor":"middle"},t.toFixed(2)));}svg.append(el("text",{x:6,y:15},unit));for(const c of curves){let d="";for(let i=0;i<c.v.length;i++)if(Number.isFinite(c.v[i]))d+=(d?"L":"M")+X(c.t[i])+" "+Y(c.v[i])+" ";svg.append(el("path",{d,fill:"none",stroke:c.color,"stroke-width":2,"stroke-dasharray":c.dash?"6 4":""}));}card.append(svg);const legend=document.createElement("div");legend.className="legend";legend.textContent=curves.map(c=>c.name).join(" · ");card.append(legend);document.getElementById("plots").append(card);}
const curve=(name,t,v,color,dash=false)=>({name,t,v,color,dash}),state=tr.state||[],eq=tr.equilibrium||[];
for(let j=0;j<2;j++)plot("Hip "+(j?"up":"forward")+" position","m",[curve("Recorded",rt,r.hip_target_m.map(v=>v[j]),"#909ea5",true),curve("Simulated",tt,state.map(v=>v[j]),"#176b89"),curve("Equilibrium",tt,eq.map(v=>v[j]),"#874daf")]);
for(let j=0;j<2;j++)plot((j?"Ankle":"Knee")+" angle","rad",[curve("Recorded",rt,r.joint_target_rad.map(v=>v[j]),"#909ea5",true),curve("Simulated",tt,state.map(v=>v[j+3]),"#176b89"),curve("Equilibrium",tt,eq.map(v=>v[j+2]),"#874daf")]);
for(let j=0;j<2;j++)plot((j?"Vertical":"Fore-aft")+" ground force","N",[curve("Measured",r.grf_time_s,r.grf_target_n.map(v=>v[j]),"#9b744f",true),curve("Simulated",tt,(tr.grf_n||[]).map(v=>v[j]),"#c36b2d")]);
let playing=false,last=null,time=0;document.getElementById("slider").oninput=e=>{time=duration*Number(e.target.value)/1000;draw(time);};document.getElementById("play").onclick=()=>{playing=!playing;last=null;document.getElementById("play").textContent=playing?"Pause":"Play";};function tick(now){if(playing){if(last!==null)time=(time+(now-last)/1000*.25)%duration;last=now;document.getElementById("slider").value=time/duration*1000;draw(time);}requestAnimationFrame(tick);}addEventListener("resize",()=>{if(!location.hash.includes("#experiment"))draw(time);});
const hashExperiment=location.hash.replace(/^#/,"").split("&")[0]==="experiment";
if(hashExperiment){
  document.getElementById("plots").hidden=true;
  document.getElementById("show_reference").checked=false;document.getElementById("show_reference").parentElement.hidden=true;
  document.getElementById("show_markers").checked=false;document.getElementById("show_markers").parentElement.hidden=true;
  document.getElementById("play").parentElement.hidden=true;
  document.getElementById("spring_status").textContent="Experiment mode: waiting for selected condition data; no recorded motion overlay is used.";
}
draw(0);requestAnimationFrame(tick);


// Native embedding protocol.  The page remains a complete standalone report when no
// parent configures it; experiment mode only adds the selected-condition comparison.
const nativePayload=data;
let visualPrimary=null, visualComparison=null, experimentMode=hashExperiment, comparisonOpacity=.25;
let primaryLabel="Primary", comparisonLabel="Comparison", showComparison=true;
function visualDataset(payload,label){
  const d={payload,r:payload.reference||{},tr:payload.trace||{},summary:payload.summary||{},geometry:payload.geometry||{},springs:payload.springs||{},label:label||"Condition",meshCache:{}};
  if(d.geometry.available){d.meshCache.last=makeMeshCache(d.geometry.last,[231,178,112]);d.meshCache.midsole=makeMeshCache(d.geometry.midsole,[92,159,147]);}
  if(d.springs.available){
    d.springs=Object.assign({},d.springs);
    d.springs._baseColorMaxMm=Number(d.springs.color_max_mm)||0;
    for(const key of ["bottom_m","top_m","compression_m"])d[key]=unpackFloat(d.springs[key]);
    d.springs.bottom_m=d.bottom_m;d.springs.top_m=d.top_m;d.springs.compression_m=d.compression_m;
  }
  return d;
}
function activateDataset(d){
  r=d.r;tr=d.tr;geometry=d.geometry;springs=d.springs;meshCache=d.meshCache;
  springBottom=springs.bottom_m||null;springTop=springs.top_m||null;springCompression=springs.compression_m||null;
  if(springs.available&&springs.anchor_local_m){springRows=[...new Set(springs.anchor_local_m.map(p=>p[1].toFixed(8)))].map(Number).sort((a,b)=>a-b);depthColumns=springs.anchor_local_m.map((_,i)=>i).sort((a,b)=>springs.anchor_local_m[a][1]-springs.anchor_local_m[b][1]);}
}
function groundHeight(d){
  const candidates=[d.payload.ground_height_m,d.summary.ground_height_m,d.summary.shoe&&d.summary.shoe.ground_height_m,d.r.ground_height_m,d.tr.ground_height_m];
  for(const value of candidates)if(Number.isFinite(Number(value)))return Number(value);
  return 0;
}
function sideState(d,t){
  const times=d.tr.time_s||[], refs=d.r.time_s||[];
  if(!times.length||!Array.isArray(d.tr.state)||!d.tr.state.length)return {q:null,ti:-1,sf:0,ref:refs.length?d.r.state[idx(Math.min(t,refs.at(-1)),refs)]:null,ri:refs.length?idx(Math.min(t,refs.at(-1)),refs):0};
  const ti=idx(Math.min(Math.max(t,0),times.at(-1)),times),sf=d.springs.available&&d.springs.time_s.length?idx(Math.min(Math.max(t,0),d.springs.time_s.at(-1)),d.springs.time_s):0;
  const ri=refs.length?idx(Math.min(Math.max(t,0),refs.at(-1)),refs):0;
  const traceIndex=d.springs.available&&Array.isArray(d.springs.trace_index)&&d.springs.trace_index.length?d.springs.trace_index[sf]:ti;
  return {q:d.tr.state[Math.min(traceIndex,d.tr.state.length-1)],ti:Math.min(traceIndex,d.tr.state.length-1),sf,ref:refs.length?d.r.state[ri]:null,ri};
}
function datasetSliceColumns(d){
  if(!d.springs.available||!d.springs.anchor_local_m||!d.springs.anchor_local_m.length)return [];
  const rows=[...new Set(d.springs.anchor_local_m.map(p=>p[1].toFixed(8)))].map(Number).sort((a,b)=>a-b);
  let target=0;
  if(visualPrimary&&visualPrimary.springs.anchor_local_m&&visualPrimary.springs.anchor_local_m.length){
    const prows=[...new Set(visualPrimary.springs.anchor_local_m.map(p=>p[1].toFixed(8)))].map(Number).sort((a,b)=>a-b);
    target=prows[Math.min(Number(sliceSelect.value)||0,Math.max(0,prows.length-1))]||0;
  }
  const row=rows.reduce((best,v)=>Math.abs(v-target)<Math.abs(best-target)?v:best,rows[0]);
  return d.springs.anchor_local_m.map((_,i)=>i).filter(i=>Math.abs(d.springs.anchor_local_m[i][1]-row)<d.springs.spacing_m*.45);
}
function sideSpringPoints(d,frame){
  if(!d.springs.available)return [];
  const a=d.springs,bottom=a.bottom_m,top=a.top_m,count=a.rest_length_m.length,result=[];
  const columns=(allSprings.checked?Array.from({length:count},(_,i)=>i):datasetSliceColumns(d));
  for(const i of columns){const k=(frame*count+i)*3;result.push([bottom[k],bottom[k+2]],[top[k],top[k+2]]);}return result;
}
function sideBounds(d,p,q){
  if(!d.geometry.available||!p||!q)return [];
  const a=q[2]+q[3]+Math.PI/2+q[4]-(d.geometry.static_pitch_rad||0),out=[];
  for(const key of ["midsole","last"]){const g=d.meshCache[key];if(!g)continue;for(const x of [g.xmin,g.xmax])for(const z of [g.zmin,g.zmax])out.push(place([x,0,z],p[2],a));}return out;
}
function drawGround(context,w,v,height,color){const y=v.xy([0,height])[1];context.strokeStyle=color;context.lineWidth=1;context.beginPath();context.moveTo(0,y);context.lineTo(w,y);context.stroke();}
function drawCondition(context,d,t,v,alpha,primary){
  activateDataset(d);const st=sideState(d,t),raw=st.q,q=raw,pose=raw?joints(raw):(st.ref?joints(st.ref):null);
  // In an experiment an absent trace is an explicit stopped/empty result, not mocap.
  if(!q&&experimentMode)return {state:st};
  if(!pose)return {state:st};
  const orientation=raw||st.ref;
  context.save();context.globalAlpha=alpha;
  meshes(context,pose,orientation,v,alpha,viewSelect.value);
  if(q){
    const savedSlice=sliceColumns,savedDepth=depthColumns;
    if(d!==visualPrimary&&d.springs.available){sliceColumns=datasetSliceColumns(d);depthColumns=d.springs.anchor_local_m.map((_,i)=>i).sort((a,b)=>d.springs.anchor_local_m[a][1]-d.springs.anchor_local_m[b][1]);}
    drawSprings(context,v,st.sf);sliceColumns=savedSlice;depthColumns=savedDepth;chain(context,pose,v,primary?"#176b89":"#778a95",!primary);mounting(context,pose,orientation,v,primary);}
  if(primary&&q){
    const eq=d.tr.equilibrium&&d.tr.equilibrium[st.ti];
    if(eq){const e=v.xy(eq.slice(0,2)),p=v.xy(pose[0]);context.strokeStyle="#874daf";context.lineWidth=2;context.beginPath();context.moveTo(e[0]-6,e[1]);context.lineTo(e[0]+6,e[1]);context.moveTo(e[0],e[1]-6);context.lineTo(e[0],e[1]+6);context.stroke();context.setLineDash([3,3]);context.beginPath();context.moveTo(...p);context.lineTo(...e);context.stroke();context.setLineDash([]);}
    arrow(context,pose[0],d.tr.hip_force_n&&d.tr.hip_force_n[st.ti],v,"#874daf");
    const f=d.tr.grf_n&&d.tr.grf_n[st.ti],m=d.tr.ankle_contact_moment_nm&&d.tr.ankle_contact_moment_nm[st.ti];let cop=pose[2][0];
    if(f&&f[1]>5&&Number.isFinite(m))cop+=(m-(pose[2][1]-groundHeight(d))*f[0])/f[1];
    arrow(context,[cop,groundHeight(d)],f,v,"#c36b2d");
  }
  context.restore();return {state:st};
}
function activeDuration(d){const a=d.tr.time_s||d.r.time_s||[];return a.length?a.at(-1):0;}
function comparisonDraw(t){
  const primary=visualPrimary||visualDataset(nativePayload,primaryLabel),other=visualComparison;
  const pt=Math.max(Number(t)||0,0);
  const sets=[primary];if(experimentMode&&showComparison&&other)sets.unshift(other);
  const allDatasets=[primary];if(other)allDatasets.push(other);
  const mmScale=Math.max(0,...allDatasets.filter(d=>d.springs.available).map(d=>d.springs._baseColorMaxMm||Number(d.springs.color_max_mm)||0));
  for(const d of allDatasets)if(d.springs.available)d.springs._activeColorMaxMm=mmScale;
  activateDataset(primary);refreshSpringControls();
  for(const d of sets){activateDataset(d);const st=sideState(d,pt),pose=st.q?joints(st.q):(!experimentMode&&st.ref?joints(st.ref):null);d._points=(pose? [pose[0],pose[1],pose[2]]: [[0,0],[.2,0],[.4,0]]).concat(pose?sideBounds(d,pose,st.q||st.ref):[],sideSpringPoints(d,st.sf));}
  const all=sets.flatMap(d=>d._points),[context,w,h]=setup(canvas,430),v=projection(all.length?all:[[0,0],[.2,0]],w,h,.1);
  for(const d of sets)drawGround(context,w,v,groundHeight(d),d===primary?"#7b9d86":"#a5b0b5");
  if(other&&experimentMode&&showComparison)drawCondition(context,other,pt,v,comparisonOpacity,false);
  drawCondition(context,primary,pt,v,1,true);
  context.fillStyle="#536774";context.font="12px system-ui";context.fillText("z: up | x: forward",15,22);
  context.fillStyle="#176b89";context.fillText(primaryLabel+" (solid) — purple: controller eq/force · orange: GRF",15,42);
  if(other&&experimentMode&&showComparison)context.fillText(comparisonLabel+" (faded, opacity "+comparisonOpacity.toFixed(2)+")",15,62);
  const pst=sideState(primary,pt),ost=other?sideState(other,pt):null;
  if(!pst.q)context.fillText("No simulated trace: condition is stopped/empty; no mocap fallback is shown",15,82);
  else if(pt>activeDuration(primary)+.001)context.fillText("Simulation held at last saved state",15,82);
  if(other&&showComparison&&!ost.q)context.fillText(comparisonLabel+": no simulated trace (not drawn)",15,102);
  else if(other&&showComparison&&ost&&pt>activeDuration(other)+.001)context.fillText(comparisonLabel+": held at last saved state",15,102);
  // The detail view and deformation map are native primary visuals.  Comparator mesh
  // is drawn first in the same frame when available, so the primary remains legible.
  const [detail,dw,dh]=setup(footCanvas,310),pstate=pst.q?joints(pst.q):null;
  if(other&&showComparison&&ost&&ost.q){drawGround(detail,dw,projection([[0,0],[.2,0]],dw,dh,.025),groundHeight(other),"#a5b0b5");}
  if(pstate){activateDataset(primary);const dq=pstate,ankle=dq[2],knee=dq[1],length=Math.max(1e-9,Math.hypot(knee[0]-ankle[0],knee[1]-ankle[1])),shortShank=[ankle[0]+.09*(knee[0]-ankle[0])/length,ankle[1]+.09*(knee[1]-ankle[1])/length];let dp=[ankle,shortShank].concat(sideBounds(primary,dq,pst.q),sideSpringPoints(primary,pst.sf));if(other&&showComparison&&ost&&ost.q){const cp=joints(ost.q),cankle=cp[2],cknee=cp[1],clength=Math.max(1e-9,Math.hypot(cknee[0]-cankle[0],cknee[1]-cankle[1])),cshortShank=[cankle[0]+.09*(cknee[0]-cankle[0])/clength,cankle[1]+.09*(cknee[1]-cankle[1])/clength];dp=dp.concat([cankle,cshortShank],sideBounds(other,cp,ost.q),sideSpringPoints(other,ost.sf));}const dv=projection(dp,dw,dh,.025);if(other&&showComparison&&ost&&ost.q)drawCondition(detail,other,pt,dv,comparisonOpacity,false);drawCondition(detail,primary,pt,dv,1,true);detail.fillStyle="#95642b";detail.font="12px system-ui";detail.fillText("Gold: fullfoot last mesh",10,dh-22);detail.fillStyle="#467d73";detail.fillText(viewSelect.value==="mesh"?"Green: undeformed midsole mesh":"Spring colors: solved foundation compression",10,dh-7);}
  activateDataset(primary);if(pst.q){drawSpringMap(pst.sf);updateColumnLabel(pst.sf);}else{const [mc,mw,mh]=setup(mapCanvas,245);mc.fillStyle="#536774";mc.font="13px system-ui";mc.fillText("No simulated spring history",12,30);}
  document.getElementById("clock").textContent=pt.toFixed(3)+" / "+activeDuration(primary).toFixed(3)+" s"+(primary.springs.available&&pst.q?" | saved frame "+primary.springs.time_s[pst.sf].toFixed(3)+" s":"");
  document.getElementById("condition_status").textContent=primaryLabel+" is solid; "+(other&&showComparison?comparisonLabel+" is faded at opacity "+comparisonOpacity.toFixed(2)+". ":"")+"Green is undeformed CAD context, purple is primary controller eq/force, orange is primary GRF. Color is COMPRESSION, not force.";
}
function experimentUi(on){
  experimentMode=on;
  document.getElementById("plots").hidden=on;
  document.getElementById("metrics_details").hidden=on;document.getElementById("summary_details").hidden=on;
  if(on)document.getElementById("native_legend").textContent=primaryLabel+" (solid): simulated primary scene · "+comparisonLabel+" (faded): selected comparison scene · green: undeformed CAD context · purple: primary controller equilibrium/force · orange: primary GRF · spring colors: COMPRESSION, not force.";
  document.getElementById("show_reference").parentElement.hidden=on;
  document.getElementById("show_markers").parentElement.hidden=on;
  document.getElementById("play").parentElement.hidden=on;
  document.getElementById("report_title").hidden=on&&compactMode;
  document.getElementById("report_note").hidden=on&&compactMode;
  document.getElementById("native_legend").hidden=on&&compactMode;
  refreshSpringControls();
  if(on)document.getElementById("spring_status").textContent="Experiment mode: parent owns plots and time controls; no recorded motion overlay is used.";
}
function nativeHeight(){const main=document.querySelector("main"),rect=main.getBoundingClientRect();parent.postMessage({type:"newton-native-height",height:Math.ceil(rect.height)},"*");}
function frameTime(d,st){
  if(!st||!st.q)return null;
  if(d.springs.available&&d.springs.time_s&&d.springs.time_s.length)return Number(d.springs.time_s[st.sf]);
  if(d.tr.time_s&&d.tr.time_s.length)return Number(d.tr.time_s[st.ti]);
  return null;
}
function nativeFrameAck(frameId){
  const primary=visualPrimary||visualDataset(nativePayload,primaryLabel),pst=sideState(primary,time),other=visualComparison,ost=other?sideState(other,time):null;
  parent.postMessage({type:"newton-native-frame",frame_id:frameId,time_s:time,primary_frame_time_s:frameTime(primary,pst),comparison_frame_time_s:other?frameTime(other,ost):null,primary_label:primaryLabel,comparison_label:other?comparisonLabel:null},"*");
}
function nativeMessage(event){
  if(event.source!==parent||!event.data||typeof event.data.type!=="string")return;
  const msg=event.data;
  if(msg.type==="newton-native-request")parent.postMessage({type:"newton-native-data",request_id:msg.request_id,data:nativePayload},"*");
  else if(msg.type==="newton-native-time"&&experimentMode){const value=Number(msg.time_s);if(Number.isFinite(value)){visualPrimary=visualPrimary||visualDataset(nativePayload,primaryLabel);time=Math.max(0,value);comparisonDraw(time);nativeFrameAck(msg.frame_id);}}
  else if(msg.type==="newton-native-configure"){
    const hasComparison=Object.prototype.hasOwnProperty.call(msg,"comparison"),c=msg.comparison;
    visualPrimary=visualPrimary||visualDataset(nativePayload,msg.primary_label||"Primary");
    if(hasComparison)visualComparison=c&&typeof c==="object"?visualDataset(c,msg.comparison_label||"Comparison"):null;
    primaryLabel=String(msg.primary_label||"Primary");comparisonLabel=String(msg.comparison_label||"Comparison");
    comparisonOpacity=Math.max(0,Math.min(1,Number.isFinite(Number(msg.comparison_opacity??msg.opac))?Number(msg.comparison_opacity??msg.opac):.25));
    showComparison=msg.show_comparison!==false;experimentMode=msg.experiment===true;
    externalHeatmaps=msg.external_heatmaps===true;
    compressionMetric=msg.compression_metric==="strain"?"strain":"mm";
    metricSelect.value=compressionMetric;
    compactMode=experimentMode&&msg.compact===true;
    refreshSpringControls();
    if(experimentMode){experimentUi(true);draw=comparisonDraw;playing=false;time=0;comparisonDraw(0);nativeHeight();parent.postMessage({type:"newton-native-configured"},"*");}
    else {experimentUi(false);nativeHeight();}
  }
}
addEventListener("message",nativeMessage);addEventListener("resize",()=>{if(experimentMode){comparisonDraw(time);nativeHeight();}});
parent.postMessage({type:"newton-native-ready"},"*");
</script></html>"""
