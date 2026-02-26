var j=Object.defineProperty;var N=(a,t,s)=>t in a?j(a,t,{enumerable:!0,configurable:!0,writable:!0,value:s}):a[t]=s;var o=(a,t,s)=>N(a,typeof t!="symbol"?t+"":t,s);(function(){const t=document.createElement("link").relList;if(t&&t.supports&&t.supports("modulepreload"))return;for(const e of document.querySelectorAll('link[rel="modulepreload"]'))i(e);new MutationObserver(e=>{for(const n of e)if(n.type==="childList")for(const c of n.addedNodes)c.tagName==="LINK"&&c.rel==="modulepreload"&&i(c)}).observe(document,{childList:!0,subtree:!0});function s(e){const n={};return e.integrity&&(n.integrity=e.integrity),e.referrerPolicy&&(n.referrerPolicy=e.referrerPolicy),e.crossOrigin==="use-credentials"?n.credentials="include":e.crossOrigin==="anonymous"?n.credentials="omit":n.credentials="same-origin",n}function i(e){if(e.ep)return;e.ep=!0;const n=s(e);fetch(e.href,n)}})();const y=2/Math.log(1+Math.sqrt(2));class A{constructor(t,s=1,i=42){o(this,"n");o(this,"j");o(this,"lattice");o(this,"temp");o(this,"exp4");o(this,"exp8");o(this,"eAcc",0);o(this,"mAcc",0);o(this,"count",0);o(this,"sweeps",0);this.n=t,this.j=s,this.temp=2,this.exp4=Math.exp(-s*4/this.temp),this.exp8=Math.exp(-s*8/this.temp);let e=i|0||42;const n=()=>(e^=e<<13,e^=e>>>17,e^=e<<5,(e>>>0)/4294967296);this.lattice=new Int8Array(t*t);for(let c=0;c<t*t;c++)this.lattice[c]=n()<.5?1:-1}setTemperature(t){this.temp=t,this.exp4=Math.exp(-this.j*4/t),this.exp8=Math.exp(-this.j*8/t),this.eAcc=0,this.mAcc=0,this.count=0}getTemperature(){return this.temp}energy(){const t=this.n,s=this.lattice;let i=0;for(let e=0;e<t;e++)for(let n=0;n<t;n++){const c=s[e*t+n];i+=c*(s[e*t+(n+1)%t]+s[(e+1)%t*t+n])}return-this.j*i}magnetization(){let t=0;for(let s=0;s<this.lattice.length;s++)t+=this.lattice[s];return t}step(){const t=this.n,s=t*t,i=this.lattice,e=this.exp4,n=this.exp8;let c=Math.random()*4294967296|0;const u=()=>(c=Math.imul(c,1664525)+1013904223|0,(c>>>0)/4294967296);for(let d=0;d<s;d++){const m=u()*t|0,v=u()*t|0,S=i[m*t+v],O=i[(m-1+t)%t*t+v]+i[(m+1)%t*t+v]+i[m*t+(v-1+t)%t]+i[m*t+(v+1)%t],M=2*S*O;let f;M<=0?f=!0:M===4?f=u()<e:M===8?f=u()<n:f=!1,f&&(i[m*t+v]=-S)}this.sweeps++;const x=s,h=this.energy()/x,b=Math.abs(this.magnetization())/x;this.eAcc+=h,this.mAcc+=b,this.count++}stats(){const t=Math.max(this.count,1);return{eMean:this.eAcc/t,mMean:this.mAcc/t,sweeps:this.sweeps}}resetStats(){this.eAcc=0,this.mAcc=0,this.count=0}setAllUp(){this.lattice.fill(1),this.resetStats()}setAllDown(){this.lattice.fill(-1),this.resetStats()}randomise(){for(let t=0;t<this.lattice.length;t++)this.lattice[t]=Math.random()<.5?1:-1;this.resetStats()}}const p=64,r=7,F=[{label:"Low T (ordered)",T:1.5,init:"up"},{label:`Critical (T≈${y.toFixed(2)})`,T:y,init:"random"},{label:"High T (disordered)",T:3.5,init:"random"}];document.querySelector("#app").innerHTML=`
<h1>2D Ising Model — Statistical Physics</h1>
<p class="subtitle">H = −J Σ s<sub>i</sub>s<sub>j</sub> &nbsp;|&nbsp;
   Metropolis-Hastings MC &nbsp;|&nbsp;
   Onsager T<sub>c</sub> = ${y.toFixed(4)}</p>

<div class="layout">

  <div class="canvas-wrap">
    <canvas id="canvas" width="${p*r}" height="${p*r}"></canvas>
    <div class="tc-line">Red = spin +1 &nbsp; Blue = spin −1</div>
  </div>

  <div class="controls">

    <!-- Stats -->
    <div class="panel">
      <div class="panel-title">Observables (running average)</div>
      <div class="stats-grid">
        <div class="stat">
          <span class="stat-label">⟨E⟩ / N²</span>
          <span class="stat-value" id="stat-e">—</span>
        </div>
        <div class="stat">
          <span class="stat-label">⟨|M|⟩ / N²</span>
          <span class="stat-value mag" id="stat-m">—</span>
        </div>
        <div class="stat">
          <span class="stat-label">Temperature T</span>
          <span class="stat-value temp" id="stat-t">—</span>
        </div>
        <div class="stat">
          <span class="stat-label">MC sweeps</span>
          <span class="stat-value" id="stat-sweeps" style="font-size:.85rem">0</span>
        </div>
      </div>
    </div>

    <!-- Temperature -->
    <div class="panel">
      <div class="panel-title">Temperature</div>
      <div class="slider-row">
        <label>T</label>
        <input id="slider-t" type="range" min="0.5" max="5.0" step="0.02" value="2.269">
        <span class="slider-val" id="val-t">2.269</span>
      </div>
      <div style="margin-top:.5rem">
        <span class="speed-label">T<sub>c</sub> = ${y.toFixed(4)}</span>
      </div>
    </div>

    <!-- Presets -->
    <div class="panel">
      <div class="panel-title">Presets</div>
      <div class="btn-row">
        ${F.map((a,t)=>`
          <button class="preset-btn" data-preset="${t}">${a.label}</button>
        `).join("")}
      </div>
    </div>

    <!-- Controls -->
    <div class="panel">
      <div class="panel-title">Simulation</div>
      <div class="btn-row" style="margin-bottom:.5rem">
        <button id="btn-play" class="primary">Pause</button>
        <button id="btn-reset">Reset</button>
        <button id="btn-up">All +1</button>
        <button id="btn-down" class="danger">All −1</button>
      </div>
      <div class="slider-row">
        <label style="min-width:4.5rem;font-size:.7rem">Sweeps/frame</label>
        <input id="slider-speed" type="range" min="1" max="32" step="1" value="4">
        <span class="slider-val" id="val-speed">×4</span>
      </div>
    </div>

  </div>
</div>
`;const q=document.getElementById("canvas"),P=q.getContext("2d"),D=P.createImageData(p*r,p*r),w=D.data;function E(a){const t=a.lattice,s=a.n;for(let i=0;i<s;i++)for(let e=0;e<s;e++){const n=t[i*s+e],c=n>0?220:60,u=n>0?60:100,x=n>0?60:220;for(let h=0;h<r;h++)for(let b=0;b<r;b++){const d=((i*r+h)*(s*r)+(e*r+b))*4;w[d]=c,w[d+1]=u,w[d+2]=x,w[d+3]=255}}P.putImageData(D,0,0)}let l=new A(p,1,42);l.setTemperature(y);let g=!0,I=4;const T=document.getElementById("slider-t"),$=document.getElementById("val-t"),C=document.getElementById("slider-speed"),R=document.getElementById("val-speed"),B=document.getElementById("btn-play"),z=document.getElementById("btn-reset"),H=document.getElementById("btn-up"),U=document.getElementById("btn-down"),k=document.getElementById("stat-e"),J=document.getElementById("stat-m"),K=document.getElementById("stat-t"),_=document.getElementById("stat-sweeps");T.addEventListener("input",()=>{const a=parseFloat(T.value);$.textContent=a.toFixed(3),l.setTemperature(a)});C.addEventListener("input",()=>{I=parseInt(C.value),R.textContent=`×${I}`});B.addEventListener("click",()=>{g=!g,B.textContent=g?"Pause":"Play",g&&L()});z.addEventListener("click",()=>{l=new A(p,1,Date.now()),l.setTemperature(parseFloat(T.value)),E(l)});H.addEventListener("click",()=>{l.setAllUp()});U.addEventListener("click",()=>{l.setAllDown()});document.querySelectorAll("[data-preset]").forEach(a=>{a.addEventListener("click",()=>{const t=parseInt(a.dataset.preset),s=F[t];l=new A(p,1,Date.now()),l.setTemperature(s.T),s.init==="up"?l.setAllUp():l.randomise(),T.value=String(s.T.toFixed(3)),$.textContent=s.T.toFixed(3),E(l)})});function L(){if(!g)return;for(let s=0;s<I;s++)l.step();E(l);const a=l.stats(),t=l.getTemperature();k.textContent=a.eMean.toFixed(4),J.textContent=a.mMean.toFixed(4),K.textContent=t.toFixed(3),_.textContent=a.sweeps.toLocaleString(),requestAnimationFrame(L)}E(l);L();
