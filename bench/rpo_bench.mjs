// RPO-Bench — an open, deterministic, reproducible benchmark for the two hard
// links of on-orbit logistics: non-cooperative CAPTURE (control) and monocular
// 6-DOF POSE (perception). Fixed scenarios, fixed seeds, bit-identical every run
// so anyone can reproduce a score and try to beat it. Baselines included.
//
//   node bench/rpo_bench.mjs            # run all baselines, print the leaderboard
//
// Institute for Physical AI @ BMI · The Charlot Lab · TR-2026-17
import { meanMotion, cwStepEuler, mppiCapture, rigidBodyStep, qRotate } from '../core/physics.mjs';
import fs from 'node:fs';

// ---- deterministic PRNG (mulberry32) ----
const mulberry32 = a => () => { a|=0; a=a+0x6D2B79F5|0; let t=Math.imul(a^a>>>15,1|a);
  t=t+Math.imul(t^t>>>7,61|t)^t; return ((t^t>>>14)>>>0)/4294967296; };
const gaussFrom = rng => { let u=0,v=0; while(u===0)u=rng(); while(v===0)v=rng();
  return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v); };

// ================= CAPTURE benchmark =================
const N_CAP = 40, n = meanMotion(450e3);
function captureScenarios() {
  const out = [];
  for (let i = 0; i < N_CAP; i++) {
    const r = mulberry32(1000 + i);
    out.push({ id: i, seed: 1000 + i, r0: [12 + r()*36, (r()*2-1)*16, (r()*2-1)*0.2, (r()*2-1)*0.2] });
  }
  return out;
}
// baseline controllers (all deterministic given the scenario)
const capBaselines = {
  MPPI: (sc) => { const r = mppiCapture(sc.r0, n, {}, sc.seed);
    return { captured: r.captured, dv: r.dvTotal, steps: r.steps }; },
  'PD': (sc) => { let s = sc.r0.slice(), dv = 0;
    for (let k = 0; k < 400; k++) {
      let ax = -0.02*s[0] - 0.18*s[2], ay = -0.02*s[1] - 0.18*s[3];
      const m = Math.hypot(ax,ay), U=0.03; if (m>U){ax*=U/m;ay*=U/m;}
      s = cwStepEuler(s, [ax,ay], n, 1.0); dv += Math.hypot(ax,ay);
      if (Math.hypot(s[0],s[1])<0.5 && Math.hypot(s[2],s[3])<0.05) return { captured:true, dv, steps:k+1 };
    } return { captured:false, dv, steps:400 }; },
  Learned: (sc) => { const P = LEARNED; if(!P) return {captured:false,dv:0,steps:0};
    let s = sc.r0.slice(), dv = 0;
    for (let k = 0; k < 400; k++) {
      const u = fwd(P, s); s = cwStepEuler(s, u, n, 1.0); dv += Math.hypot(u[0],u[1]);
      if (Math.hypot(s[0],s[1])<0.6 && Math.hypot(s[2],s[3])<0.1) return { captured:true, dv, steps:k+1 };
    } return { captured:false, dv, steps:400 }; },
};
// learned-policy forward pass
let LEARNED = null;
try { LEARNED = JSON.parse(fs.readFileSync(new URL('../policy/bc_policy.json', import.meta.url))); } catch {}
function fwd(P, sx){ const mv=(W,a)=>W[0].map((_,j)=>a.reduce((s,ai,i)=>s+ai*W[i][j],0)), relu=z=>z.map(v=>v>0?v:0);
  const x=sx.map((v,j)=>(v-P.xm[j])/P.xsd[j]);
  const a1=relu(mv(P.W1,x).map((v,j)=>v+P.b1[j])), a2=relu(mv(P.W2,a1).map((v,j)=>v+P.b2[j]));
  return mv(P.W3,a2).map((v,j)=>v+P.b3[j]).map((v,j)=>v*P.ysd[j]+P.ym[j]); }

function runCapture(name){ const scen = captureScenarios(), f = capBaselines[name];
  let cap=0, dvs=[], steps=[];
  for (const sc of scen){ const r = f(sc); if (r.captured){ cap++; dvs.push(r.dv); steps.push(r.steps); } }
  const mean = a => a.length ? a.reduce((s,v)=>s+v,0)/a.length : 0;
  return { rate: cap/scen.length, dv: mean(dvs), steps: mean(steps), n: scen.length };
}

// ================= POSE benchmark =================
const N_POSE = 30, FRAMES = 40;
const KP=[[-.6,-.5,-.7],[.6,-.5,-.7],[.6,.5,-.7],[-.6,.5,-.7],[-.6,-.5,.7],[.6,-.5,.7],[.6,.5,.7],[-.6,.5,.7],
  [-3.4,-.2,.1],[-3.4,-.2,-.1],[3.4,.2,.1],[3.4,.2,-.1],[.35,.55,1.7],[-.35,-.35,.75]];
const I=[1.0,2.6,3.2], K={f:800,cx:600,cy:360};
function qMat(q){const[w,x,y,z]=q;return[[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]];}
function rvMat(r){const th=Math.hypot(...r);if(th<1e-9)return[[1,0,0],[0,1,0],[0,0,1]];const[kx,ky,kz]=r.map(v=>v/th),c=Math.cos(th),s=Math.sin(th),C=1-c;
  return[[c+kx*kx*C,kx*ky*C-kz*s,kx*kz*C+ky*s],[ky*kx*C+kz*s,c+ky*ky*C,ky*kz*C-kx*s],[kz*kx*C-ky*s,kz*ky*C+kx*s,c+kz*kz*C]];}
const mul=(M,p)=>[M[0][0]*p[0]+M[0][1]*p[1]+M[0][2]*p[2],M[1][0]*p[0]+M[1][1]*p[1]+M[1][2]*p[2],M[2][0]*p[0]+M[2][1]*p[1]+M[2][2]*p[2]];
const matMul=(A,B)=>A.map(row=>[0,1,2].map(j=>row[0]*B[0][j]+row[1]*B[1][j]+row[2]*B[2][j]));
const transpose=M=>[[M[0][0],M[1][0],M[2][0]],[M[0][1],M[1][1],M[2][1]],[M[0][2],M[1][2],M[2][2]]];
function project(R,t,p){const pc=[mul(R,p)[0]+t[0],mul(R,p)[1]+t[1],mul(R,p)[2]+t[2]];if(pc[2]<0.2)return null;return[K.f*pc[0]/pc[2]+K.cx,K.f*pc[1]/pc[2]+K.cy,pc[2]];}
function solve6(A,b){const nn=6,M=A.map((r,i)=>[...r,b[i]]);for(let c=0;c<nn;c++){let p=c;for(let r=c+1;r<nn;r++)if(Math.abs(M[r][c])>Math.abs(M[p][c]))p=r;[M[c],M[p]]=[M[p],M[c]];if(Math.abs(M[c][c])<1e-12)return null;for(let r=0;r<nn;r++)if(r!==c){const f=M[r][c]/M[c][c];for(let k=c;k<=nn;k++)M[r][k]-=f*M[c][k];}}return M.map((r,i)=>r[nn]/r[i]);}
function pnp(dets,x0){let x=x0.slice();for(let it=0;it<8;it++){const R=rvMat(x.slice(0,3)),t=x.slice(3);const res=[],J=[];
  for(const d of dets){const pr=project(R,t,KP[d.i]);if(!pr)continue;res.push(d.u-pr[0],d.v-pr[1]);const r0=[],r1=[];
    for(let k=0;k<6;k++){const xp=x.slice();xp[k]+=1e-4;const prp=project(rvMat(xp.slice(0,3)),xp.slice(3),KP[d.i]);r0.push(prp?(prp[0]-pr[0])/1e-4:0);r1.push(prp?(prp[1]-pr[1])/1e-4:0);}J.push(r0);J.push(r1);}
  if(res.length<12)break;const A=Array.from({length:6},()=>Array(6).fill(0)),g=Array(6).fill(0);
  for(let r=0;r<res.length;r++){const jr=J[r];for(let a=0;a<6;a++){g[a]+=jr[a]*res[r];for(let b=0;b<6;b++)A[a][b]+=jr[a]*jr[b];}}
  for(let a=0;a<6;a++)A[a][a]=A[a][a]*1.25+1e-3;let dx=solve6(A,g);if(!dx||!dx.every(Number.isFinite))break;const mg=Math.hypot(...dx);if(mg>0.5)dx=dx.map(v=>v*0.5/mg);for(let k=0;k<6;k++)x[k]+=dx[k];if(mg<1e-5)break;}
  return x;}
function poseScenarios(){const out=[];for(let i=0;i<N_POSE;i++){const r=mulberry32(5000+i);
  const d=[r()*2-1,r()*2-1,r()*2-1],m=Math.hypot(...d),rate=0.18+r()*0.35;
  out.push({id:i,seed:5000+i,w0:d.map(v=>v/m*rate),noise:1.5+r()*2.5});}return out;}
function poseError(Re,te,R,tv){const Rrel=matMul(Re,transpose(R));const rerr=Math.acos(Math.max(-1,Math.min(1,(Rrel[0][0]+Rrel[1][1]+Rrel[2][2]-1)/2)));
  const terr=Math.hypot(te[0]-tv[0],te[1]-tv[1],te[2]-tv[2]);return{rerr,terr,speed:rerr+terr/Math.hypot(...tv)};}
const poseBaselines = {
  PnP: (sc) => {
    const rng = mulberry32(sc.seed*7+1); let Q=[1,0,0,0], W=sc.w0.slice(), est=[0.01,0,0,0,0,9];
    const errs=[];
    for (let frame=0; frame<FRAMES; frame++){
      const st=rigidBodyStep(Q,W,I,0.05); Q=st.q; W=st.w; const R=qMat(Q); const tv=[Math.sin(frame*0.06)*1.0,Math.cos(frame*0.05)*0.7,9];
      const dets=[]; for(let i=0;i<KP.length;i++){const pr=project(R,tv,KP[i]);if(!pr)continue;
        if(pr[2]>9.35&&rng()<0.5)continue; if(rng()<0.18)continue;
        dets.push({i,u:pr[0]+gaussFrom(rng)*sc.noise,v:pr[1]+gaussFrom(rng)*sc.noise});}
      if(dets.length>=6){const e2=pnp(dets,est);if(e2.every(Number.isFinite)&&e2[5]>2&&e2[5]<60)est=e2;}
      if(frame>=5){const e=poseError(rvMat(est.slice(0,3)),est.slice(3),R,tv);errs.push(e);} // skip lock-on frames
    }
    const mean=k=>errs.reduce((s,e)=>s+e[k],0)/errs.length;
    return { rerr:mean('rerr')*180/Math.PI, terr:mean('terr'), speed:mean('speed') };
  },
  'Centroid (naive)': (sc) => { // baseline: translation from the keypoint centroid, no rotation
    const rng = mulberry32(sc.seed*7+1); let Q=[1,0,0,0], W=sc.w0.slice(); const errs=[];
    for (let frame=0; frame<FRAMES; frame++){
      const st=rigidBodyStep(Q,W,I,0.05); Q=st.q; W=st.w; const R=qMat(Q); const tv=[Math.sin(frame*0.06)*1.0,Math.cos(frame*0.05)*0.7,9];
      let su=0,sv=0,cnt=0; for(let i=0;i<KP.length;i++){const pr=project(R,tv,KP[i]);if(pr){su+=pr[0];sv+=pr[1];cnt++;}}
      const te=[(su/cnt-K.cx)/K.f*9,(sv/cnt-K.cy)/K.f*9,9]; // back-project centroid at nominal depth
      if(frame>=5)errs.push(poseError([[1,0,0],[0,1,0],[0,0,1]],te,R,tv));
    }
    const mean=k=>errs.reduce((s,e)=>s+e[k],0)/errs.length;
    return { rerr:mean('rerr')*180/Math.PI, terr:mean('terr'), speed:mean('speed') };
  },
};
function runPose(name){ const scen=poseScenarios(), f=poseBaselines[name]; const rr=[],tt=[],ss=[];
  for(const sc of scen){const r=f(sc);rr.push(r.rerr);tt.push(r.terr);ss.push(r.speed);}
  const mean=a=>a.reduce((s,v)=>s+v,0)/a.length;
  return { rerr:mean(rr), terr:mean(tt), speed:mean(ss), n:scen.length }; }

// ================= run + leaderboard =================
const pad=(s,n)=>String(s).padEnd(n), padL=(s,n)=>String(s).padStart(n);
console.log('# RPO-Bench · deterministic, reproducible · Institute for Physical AI @ BMI\n');
console.log(`## CAPTURE  (${N_CAP} non-cooperative rendezvous scenarios, fixed seeds)`);
console.log(pad('controller',16), padL('success',9), padL('mean Δv',10), padL('mean steps',12));
const capOut={};
for (const name of ['MPPI','Learned','PD']){ const r=runCapture(name); capOut[name]=r;
  console.log(pad(name,16), padL((r.rate*100).toFixed(1)+'%',9), padL(r.dv.toFixed(2)+' m/s',10), padL(r.steps.toFixed(0),12)); }
console.log(`\n## POSE  (${N_POSE} tumbling-target tracks × ${FRAMES} frames, monocular, SPEED metric)`);
console.log(pad('estimator',18), padL('rot err',9), padL('trans err',11), padL('SPEED score',13));
const poseOut={};
for (const name of ['PnP','Centroid (naive)']){ const r=runPose(name); poseOut[name]=r;
  console.log(pad(name,18), padL(r.rerr.toFixed(2)+'°',9), padL(r.terr.toFixed(3)+' m',11), padL(r.speed.toFixed(3),13)); }
fs.writeFileSync(new URL('./results.json', import.meta.url), JSON.stringify({ capture:capOut, pose:poseOut, generated:'deterministic' }, null, 1));
console.log('\nwrote bench/results.json — reproduce with `node bench/rpo_bench.mjs` (bit-identical).');
