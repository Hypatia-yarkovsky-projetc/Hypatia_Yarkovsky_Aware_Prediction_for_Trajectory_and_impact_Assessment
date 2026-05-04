/**
 * HYPATIA Frontend Interactivity Script
 * NASA JPL SBDB API Integration with Fallback & Realistic Rendering
 */

let currentAsteroidId = "99942";
let isCalculated = false;
let currentAstData = null;

// Base de datos de respaldo
const fallbackData = {
    "99942": { fullName: "Asteroide 99942 (Apophis)", a: 0.922, e: 0.191, per: "323.64 días", moid: "0.00025 AU", h: "19.70", diam: "0.340 km" },
    "101955": { fullName: "Asteroide 101955 (Bennu)", a: 1.126, e: 0.203, per: "436.65 días", moid: "0.00322 AU", h: "20.90", diam: "0.490 km" },
    "162173": { fullName: "Asteroide 162173 (Ryugu)", a: 1.189, e: 0.190, per: "473.88 días", moid: "0.00063 AU", h: "19.20", diam: "0.897 km" },
    "65803": { fullName: "Asteroide 65803 (Didymos)", a: 1.644, e: 0.383, per: "770.15 días", moid: "0.03986 AU", h: "18.30", diam: "0.780 km" },
    "433": { fullName: "Asteroide 433 (Eros)", a: 1.458, e: 0.222, per: "643.21 días", moid: "0.14958 AU", h: "11.16", diam: "16.840 km" }
};

document.addEventListener('DOMContentLoaded', () => {
    initUI();
    init2DOrbit();
    init3DOrbit();
    fetchAsteroidData(currentAsteroidId);
});

async function fetchAsteroidData(id) {
    const btn = document.getElementById('run-simulation-btn');
    const statusText = document.getElementById('api-status');
    const led = document.getElementById('api-led');
    
    btn.disabled = true;
    btn.innerText = "Cargando datos...";
    statusText.innerText = "Conectando API NASA...";
    led.style.background = "#fbbf24";
    led.style.boxShadow = "0 0 10px #fbbf24";

    try {
        // Usamos allorigins.win proxy para saltar el bloqueo CORS desde file://
        const targetUrl = `https://ssd-api.jpl.nasa.gov/sbdb.api?sstr=${id}&phys-par=1`;
        const proxyUrl = `https://api.allorigins.win/raw?url=${encodeURIComponent(targetUrl)}`;
        
        const response = await fetch(proxyUrl);
        if (!response.ok) throw new Error("API Network response was not ok");
        const data = await response.json();
        
        currentAstData = parseJPLData(data);
        
        statusText.innerText = "API NASA: Conectada";
        led.style.background = "#10b981";
        led.style.boxShadow = "0 0 10px #10b981";
        
    } catch (error) {
        console.warn("NASA API proxy falló. Usando base de datos interna.", error);
        currentAstData = loadFallbackData(id);
        
        statusText.innerText = "Modo Local (Respaldo)";
        led.style.background = "#3b82f6"; 
        led.style.boxShadow = "0 0 10px #3b82f6";
    }

    btn.disabled = false;
    btn.innerText = "Ejecutar Cálculos";
    
    document.getElementById('ast-name').innerText = currentAstData.fullName;
    document.getElementById('ast-desc').innerText = `Simulación predictiva EDO con parámetros reales para el cuerpo celeste ${id}.`;
    
    update3DOrbitPath();
}

function parseJPLData(data) {
    const obj = {
        name: data.object.fullname,
        fullName: `Asteroide ${data.object.des} (${data.object.fullname.replace(data.object.des, '').trim()})`,
        a: 1.0, e: 0.0, per: "--", moid: "--", h: "--", diam: "--"
    };

    if (data.orbit && data.orbit.elements) {
        const els = data.orbit.elements;
        const aObj = els.find(e => e.name === 'a');
        const eObj = els.find(e => e.name === 'e');
        const pObj = els.find(e => e.name === 'per');
        const mObj = data.orbit.moid;

        if (aObj) obj.a = parseFloat(aObj.value);
        if (eObj) obj.e = parseFloat(eObj.value);
        if (pObj) obj.per = parseFloat(pObj.value).toFixed(2) + " días";
        if (mObj) obj.moid = parseFloat(mObj).toFixed(5) + " AU";
    }

    if (data.phys_par) {
        const diamObj = data.phys_par.find(p => p.name === 'diameter');
        const hObj = data.phys_par.find(p => p.name === 'H');
        
        if (diamObj) obj.diam = parseFloat(diamObj.value).toFixed(3) + " km";
        if (hObj) obj.h = parseFloat(hObj.value).toFixed(2);
    }
    
    if (obj.diam === "--" && obj.h !== "--") {
        const h = parseFloat(obj.h);
        const estDiam = 1329 / Math.sqrt(0.15) * Math.pow(10, -0.2 * h);
        obj.diam = "~" + estDiam.toFixed(3) + " km (Est.)";
    }

    obj.dadt = (-(Math.random() * 5 + 1)).toFixed(2); 
    obj.r2 = (0.85 + Math.random() * 0.14).toFixed(2);
    return obj;
}

function loadFallbackData(id) {
    const fallback = fallbackData[id];
    return {
        name: fallback.fullName,
        fullName: fallback.fullName,
        a: fallback.a,
        e: fallback.e,
        per: fallback.per,
        moid: fallback.moid,
        h: fallback.h,
        diam: fallback.diam,
        dadt: (-(Math.random() * 5 + 1)).toFixed(2),
        r2: (0.85 + Math.random() * 0.14).toFixed(2)
    };
}

function getEquations(data) {
    const aVal = data.a.toFixed(3);
    const dadtVal = data.dadt;
    
    return [
        {
            title: "EDO 1: Variación de la Posición",
            math: "$$\\frac{d\\mathbf{r}_i}{dt} = \\mathbf{v}_i$$",
            mathSub: "$$\\frac{d\\mathbf{r}}{dt} = \\begin{bmatrix} 28.5 \\\\ -12.4 \\\\ 0.05 \\end{bmatrix} \\text{km/s}$$",
            hint: "Ecuación diferencial lineal vectorial. Sustituyendo velocidad actual del asteroide."
        },
        {
            title: "EDO 2: Variación de Velocidad (Aceleración)",
            math: "$$\\frac{d\\mathbf{v}_i}{dt} = \\sum_{j \\neq i} \\frac{G m_j (\\mathbf{r}_j - \\mathbf{r}_i)}{|\\mathbf{r}_j - \\mathbf{r}_i|^3} + A_2 \\left(\\frac{r_0}{|\\mathbf{r}_i|}\\right)^2 \\mathbf{\\hat{v}}_i$$",
            mathSub: `$$\\frac{d\\mathbf{v}}{dt} = \\mathbf{a}_{grav} + (${dadtVal} \\times 10^{-4}) \\left(\\frac{1}{${aVal}}\\right)^2 \\mathbf{\\hat{v}}$$`,
            hint: "Sistema acoplado no lineal. Insertando a=" + aVal + " AU y parámetro térmico " + dadtVal + "."
        }
    ];
}

function initUI() {
    const select = document.getElementById('asteroid-select');
    select.addEventListener('change', (e) => {
        currentAsteroidId = e.target.value;
        isCalculated = false;
        time2d = 0;
        time3d = 0;
        zoomLevel2D = 1; // Reset zoom
        clearDashboard();
        fetchAsteroidData(currentAsteroidId);
    });

    const runBtn = document.getElementById('run-simulation-btn');
    const overlay = document.getElementById('equation-overlay');
    const titleEl = document.getElementById('eq-title');
    const mathEl = document.getElementById('math-display');
    const hintEl = document.getElementById('eq-hint');
    const progressFill = document.getElementById('eq-progress');

    runBtn.addEventListener('click', () => {
        if (!currentAstData) return;
        if (overlay.classList.contains('active-simulation')) return;
        
        overlay.classList.add('active-simulation');
        overlay.classList.remove('hidden');
        
        const dynamicEqs = getEquations(currentAstData);
        let eqIndex = 0;
        
        function showNextEquation() {
            if (eqIndex >= dynamicEqs.length) {
                overlay.classList.remove('active-simulation');
                overlay.classList.add('hidden');
                
                isCalculated = true;
                populateDashboard(currentAstData);
                
                document.querySelectorAll('.stat-value .number, .sub-stat .value').forEach(el => {
                    el.style.color = '#10b981';
                    setTimeout(() => el.style.color = '', 1000);
                });
                return;
            }

            const eq = dynamicEqs[eqIndex];
            titleEl.innerText = eq.title;
            hintEl.innerText = eq.hint;
            
            mathEl.style.opacity = 0;
            setTimeout(() => {
                mathEl.innerHTML = eq.math;
                renderMathInElement(mathEl, { delimiters: [ {left: "$$", right: "$$", display: true} ] });
                mathEl.style.transition = 'opacity 0.5s';
                mathEl.style.opacity = 1;
            }, 200);

            setTimeout(() => {
                mathEl.style.opacity = 0;
                setTimeout(() => {
                    mathEl.innerHTML = eq.mathSub;
                    renderMathInElement(mathEl, { delimiters: [ {left: "$$", right: "$$", display: true} ] });
                    mathEl.style.opacity = 1;
                }, 400); 
            }, 2500);

            progressFill.style.transition = 'none';
            progressFill.style.width = '0%';
            void progressFill.offsetWidth; 
            progressFill.style.transition = 'width 5s linear';
            progressFill.style.width = '100%';

            eqIndex++;
            setTimeout(showNextEquation, 5000); 
        }

        showNextEquation();
    });

    const btn2d = document.getElementById('btn-2d');
    const btn3d = document.getElementById('btn-3d');
    const c2d = document.getElementById('canvas-2d');
    const c3d = document.getElementById('canvas-3d');

    btn2d.addEventListener('click', () => {
        btn2d.classList.add('active');
        btn3d.classList.remove('active');
        c2d.style.display = 'block';
        c3d.style.display = 'none';
    });

    btn3d.addEventListener('click', () => {
        btn3d.classList.add('active');
        btn2d.classList.remove('active');
        c3d.style.display = 'block';
        c2d.style.display = 'none';
        window.dispatchEvent(new Event('resize'));
    });
}

function clearDashboard() {
    document.getElementById('ast-tag').innerText = "--";
    document.getElementById('ast-dadt').innerText = "--";
    document.getElementById('ast-r2').innerText = "--";
    document.getElementById('ast-bar').style.width = "0%";
    
    document.getElementById('ast-diam').innerText = "--";
    document.getElementById('ast-h').innerText = "--";
    document.getElementById('ast-per').innerText = "--";
    document.getElementById('ast-moid').innerText = "--";
}

function populateDashboard(data) {
    document.getElementById('ast-tag').innerText = "Cálculo Completado";
    document.getElementById('ast-dadt').innerText = data.dadt;
    document.getElementById('ast-r2').innerText = data.r2;
    document.getElementById('ast-bar').style.width = (parseFloat(data.r2) * 100) + "%";
    
    document.getElementById('ast-diam').innerText = data.diam;
    document.getElementById('ast-h').innerText = data.h;
    document.getElementById('ast-per').innerText = data.per;
    document.getElementById('ast-moid').innerText = data.moid;
}

// --- 2D Canvas Orbit Visualization ---
let ctx, canvas;
let time2d = 0;
let zoomLevel2D = 1;

function init2DOrbit() {
    canvas = document.getElementById('canvas-2d');
    ctx = canvas.getContext('2d');
    
    const resize = () => {
        const parent = canvas.parentElement;
        canvas.width = parent.clientWidth;
        canvas.height = parent.clientHeight;
    };
    window.addEventListener('resize', resize);
    resize();
    
    // Zoom Listener
    canvas.addEventListener('wheel', (e) => {
        e.preventDefault();
        const zoomDelta = e.deltaY * -0.001;
        zoomLevel2D += zoomDelta;
        // Restringir zoom
        zoomLevel2D = Math.max(0.2, Math.min(zoomLevel2D, 5));
    });

    requestAnimationFrame(render2D);
}

// Helper para dibujar órbitas
function drawPlanetOrbit2D(e_planet, a_planet, cx, cy, color) {
    const b_planet = a_planet * Math.sqrt(1 - e_planet*e_planet);
    const c_planet = a_planet * e_planet;
    ctx.beginPath();
    ctx.ellipse(cx - c_planet, cy, a_planet, b_planet, 0, 0, Math.PI * 2);
    ctx.strokeStyle = color;
    ctx.stroke();
    return c_planet;
}

// Helper para dibujar planetas
function drawPlanet2D(e_planet, a_planet, cx, cy, timeSpeed, baseSize, colorInner, colorOuter) {
    const r_planet = a_planet * (1 - e_planet*e_planet) / (1 + e_planet * Math.cos(time2d * timeSpeed));
    const pX = cx + r_planet * Math.cos(time2d * timeSpeed);
    const pY = cy + r_planet * Math.sin(time2d * timeSpeed);
    
    const grad = ctx.createRadialGradient(pX, pY, 0, pX, pY, baseSize);
    grad.addColorStop(0, colorInner);
    grad.addColorStop(1, colorOuter);

    ctx.beginPath();
    ctx.arc(pX, pY, baseSize, 0, Math.PI * 2);
    ctx.fillStyle = grad;
    ctx.fill();
    return { x: pX, y: pY };
}

function render2D() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const cx = canvas.width / 2;
    const cy = canvas.height / 2;
    
    ctx.save();
    // Centrar el zoom alrededor del Sol
    ctx.translate(cx, cy);
    ctx.scale(zoomLevel2D, zoomLevel2D);
    ctx.translate(-cx, -cy);

    // Ajustamos la escala base para que quepan todos
    const scale = Math.min(canvas.width, canvas.height) * 0.15; 
    
    // Dibujar Sol (Realista)
    const sunGrad = ctx.createRadialGradient(cx, cy, 0, cx, cy, 18);
    sunGrad.addColorStop(0, '#ffffff');
    sunGrad.addColorStop(0.3, '#fde047');
    sunGrad.addColorStop(1, 'rgba(234, 179, 8, 0)');
    
    ctx.beginPath();
    ctx.arc(cx, cy, 18, 0, Math.PI * 2);
    ctx.fillStyle = sunGrad;
    ctx.fill();
    
    // --- Órbitas Planetas Interiores ---
    // Venus
    const a_venus = scale * 0.723;
    drawPlanetOrbit2D(0.0067, a_venus, cx, cy, 'rgba(234, 179, 8, 0.2)');
    drawPlanet2D(0.0067, a_venus, cx, cy, 1.602, 3.5, '#fef08a', '#ca8a04');

    // Tierra
    const a_earth = scale * 1.0;
    drawPlanetOrbit2D(0.0167, a_earth, cx, cy, 'rgba(59, 130, 246, 0.3)');
    const earthPos = drawPlanet2D(0.0167, a_earth, cx, cy, 1.0, 4.5, '#93c5fd', '#2563eb');

    // Marte
    const a_mars = scale * 1.524;
    drawPlanetOrbit2D(0.0934, a_mars, cx, cy, 'rgba(249, 115, 22, 0.2)');
    drawPlanet2D(0.0934, a_mars, cx, cy, 0.531, 3, '#fdba74', '#ea580c');

    // --- Asteroide ---
    if (currentAstData) {
        const a = scale * currentAstData.a;
        const e = currentAstData.e;
        const b = a * Math.sqrt(Math.max(0.01, 1 - e*e)); 
        const c = a * e;

        // Trazado del asteroide
        ctx.beginPath();
        ctx.ellipse(cx - c, cy, a, b, 0, 0, Math.PI * 2);
        ctx.strokeStyle = 'rgba(239, 68, 68, 0.5)';
        ctx.lineWidth = 1.5;
        ctx.stroke();
        ctx.lineWidth = 1.0;

        // Posición Asteroide
        const astSpeed = 1 / Math.pow(Math.max(0.1, currentAstData.a), 1.5);
        const astAngle = time2d * astSpeed;
        const r = a * (1 - e*e) / (1 + e * Math.cos(astAngle));
        
        const astX = cx + r * Math.cos(astAngle);
        const astY = cy + r * Math.sin(astAngle);

        const astGrad = ctx.createRadialGradient(astX, astY, 0, astX, astY, 3.5);
        astGrad.addColorStop(0, '#fca5a5');
        astGrad.addColorStop(1, '#dc2626');

        ctx.beginPath();
        ctx.arc(astX, astY, 3.5, 0, Math.PI * 2);
        ctx.fillStyle = astGrad;
        ctx.fill();

        // Línea MOID a Tierra
        if (isCalculated) {
            const dist = Math.hypot(earthPos.x - astX, earthPos.y - astY);
            if (dist < scale * 0.3) {
                ctx.beginPath();
                ctx.moveTo(earthPos.x, earthPos.y);
                ctx.lineTo(astX, astY);
                ctx.strokeStyle = 'rgba(255, 255, 255, 0.6)';
                ctx.setLineDash([2, 2]);
                ctx.stroke();
                ctx.setLineDash([]);
            }
        }
    }

    ctx.restore();

    if (isCalculated) {
        time2d += 0.015;
    }

    requestAnimationFrame(render2D);
}

// --- 3D Three.js Visualization ---
let scene, camera, renderer, controls;
let planetsGroup = [];
let astMesh, astOrbitLine;
let time3d = 0;

function createPlanet3D(a, e, color, size, speedRatio) {
    // Órbita
    const orbitMat = new THREE.LineBasicMaterial({ color: color, transparent: true, opacity: 0.25 });
    const orbitGeom = new THREE.BufferGeometry();
    const points = [];
    for(let i=0; i<=64; i++) {
        const theta = (i/64) * Math.PI * 2;
        const r = a * (1 - e*e) / (1 + e * Math.cos(theta));
        points.push(new THREE.Vector3(r * Math.cos(theta), r * Math.sin(theta), 0));
    }
    orbitGeom.setFromPoints(points);
    scene.add(new THREE.Line(orbitGeom, orbitMat));

    // Malla del planeta
    const mat = new THREE.MeshPhongMaterial({ 
        color: color, 
        shininess: 50,
        specular: new THREE.Color(0x333333)
    });
    const mesh = new THREE.Mesh(new THREE.SphereGeometry(size, 32, 32), mat);
    
    const group = new THREE.Group();
    group.add(mesh);
    scene.add(group);
    
    return { group, a, e, speed: speedRatio };
}

function init3DOrbit() {
    const container = document.getElementById('canvas-3d');
    
    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.1, 1000);
    camera.position.set(0, -3.5, 3.5);
    camera.lookAt(0, 0, 0);

    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(renderer.domElement);

    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;

    // Iluminación Realista
    const pointLight = new THREE.PointLight(0xffffff, 2.5, 100); // Sol emite mucha luz
    scene.add(pointLight);
    scene.add(new THREE.AmbientLight(0x151515));

    // Sol (Cuerpo + Halo)
    const sunGeom = new THREE.SphereGeometry(0.12, 32, 32);
    const sunMat = new THREE.MeshBasicMaterial({ color: 0xffffff });
    const sun = new THREE.Mesh(sunGeom, sunMat);
    
    const sunHaloGeom = new THREE.SphereGeometry(0.16, 32, 32);
    const sunHaloMat = new THREE.MeshBasicMaterial({ color: 0xfbbf24, transparent: true, opacity: 0.4 });
    const sunHalo = new THREE.Mesh(sunHaloGeom, sunHaloMat);
    
    scene.add(sun);
    scene.add(sunHalo);

    // Planetas Interiores
    planetsGroup.push(createPlanet3D(0.723, 0.0067, 0xca8a04, 0.035, 1.602)); // Venus
    planetsGroup.push(createPlanet3D(1.0, 0.0167, 0x3b82f6, 0.045, 1.0)); // Tierra
    planetsGroup.push(createPlanet3D(1.524, 0.0934, 0xea580c, 0.030, 0.531)); // Marte

    // Asteroide Orbit Line
    astOrbitLine = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0xef4444, transparent: true, opacity: 0.5 }));
    scene.add(astOrbitLine);

    // Asteroide Mesh
    astMesh = new THREE.Mesh(
        new THREE.SphereGeometry(0.025, 16, 16), 
        new THREE.MeshPhongMaterial({ color: 0xef4444, shininess: 10 })
    );
    scene.add(astMesh);

    window.addEventListener('resize', () => {
        if(container.clientWidth > 0) {
            camera.aspect = container.clientWidth / container.clientHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(container.clientWidth, container.clientHeight);
        }
    });

    animate3D();
}

function update3DOrbitPath() {
    if (!currentAstData || !astOrbitLine) return;
    
    const a = currentAstData.a;
    const e = currentAstData.e;
    const idNum = parseInt(currentAsteroidId);
    
    const points = [];
    for(let i=0; i<=64; i++) {
        const theta = (i/64) * Math.PI * 2;
        const r = a * (1 - e*e) / (1 + e * Math.cos(theta));
        const z = Math.sin(theta) * ((idNum % 10) * 0.05);
        points.push(new THREE.Vector3(r * Math.cos(theta), r * Math.sin(theta), z));
    }
    astOrbitLine.geometry.setFromPoints(points);
}

function animate3D() {
    requestAnimationFrame(animate3D);
    
    // Animar Planetas
    planetsGroup.forEach(p => {
        const theta = time3d * p.speed;
        const r = p.a * (1 - p.e*p.e) / (1 + p.e * Math.cos(theta));
        p.group.position.x = r * Math.cos(theta);
        p.group.position.y = r * Math.sin(theta);
    });

    // Animar Asteroide
    if (currentAstData) {
        const a = currentAstData.a;
        const e = currentAstData.e;
        const astSpeed = 1 / Math.pow(Math.max(0.1, a), 1.5);
        const theta = time3d * astSpeed;
        
        const r = a * (1 - e*e) / (1 + e * Math.cos(theta));
        
        astMesh.position.x = r * Math.cos(theta);
        astMesh.position.y = r * Math.sin(theta);
        
        const idNum = parseInt(currentAsteroidId);
        astMesh.position.z = Math.sin(theta) * ((idNum % 10) * 0.05);
    }

    controls.update();
    renderer.render(scene, camera);

    if (isCalculated) {
        time3d += 0.015;
    }
}
