// Doom Fire ASCII Algorithm 
// https://fabiensanglard.net/doom_fire_psx/

// Configuration
const FIRE_CONFIG = {
    width: 80,
    height: 40,
    debug: false,
    sourceChar: 'x',
    intensity: 20,
    decay: 8,
    spread: 3,
    wind:-1.5,
    fps: 14,
    resolutionScale: 2
};

// State
let firePixels = []; // 1D array of intensity (0-36)
let solidMap = [];   // 1D array of booleans (true = wall)
let sourceMap = [];  // 1D array of booleans (true = fire source)
let isRunning = false;
let lastFrameTime = 0;
let resolutionScale = 1.4;

// DOM Elements
let artPre = null;
let fireOverlay = null;

// Color Palette (Black -> Red -> Orange -> Yellow -> White)
// 37 colors
const FIRE_PALETTE = [
    {r:0,g:0,b:0},        {r:7,g:7,b:7},        {r:31,g:7,b:7},       {r:47,g:15,b:7},
    {r:71,g:15,b:7},      {r:87,g:23,b:7},      {r:103,g:31,b:7},     {r:119,g:31,b:7},
    {r:143,g:39,b:7},     {r:159,g:47,b:7},     {r:175,g:63,b:7},     {r:191,g:71,b:7},
    {r:199,g:71,b:7},     {r:223,g:79,b:7},     {r:223,g:87,b:7},     {r:223,g:87,b:7},
    {r:215,g:95,b:7},     {r:215,g:95,b:7},     {r:215,g:103,b:15},   {r:207,g:111,b:15},
    {r:207,g:119,b:15},   {r:207,g:127,b:15},   {r:207,g:135,b:23},   {r:199,g:135,b:23},
    {r:199,g:143,b:23},   {r:199,g:151,b:31},   {r:191,g:159,b:31},   {r:191,g:159,b:31},
    {r:191,g:167,b:39},   {r:191,g:167,b:39},   {r:191,g:175,b:47},   {r:183,g:175,b:47},
    {r:183,g:183,b:47},   {r:183,g:183,b:55},   {r:207,g:207,b:111},  {r:223,g:223,b:159},
    {r:239,g:239,b:199}
];

// Character Palette (Dark -> Light)
// Significantly expanded for smoother gradients
const FIRE_CHARS = " trmTRM";

// -----------------------------------------------------------------------------
// Initialization
// -----------------------------------------------------------------------------

function init() {
    artPre = document.getElementById('artpre');
    if (!artPre) {
        console.error("Art element #artpre not found");
        return;
    }

    // Setup Resize Listener
    window.removeEventListener('resize', handleResize);
    window.addEventListener('resize', handleResize);

    // Setup Mouse/Touch Wind Control
    const handleWindInput = (x) => {
        const width = window.innerWidth;
        const center = width / 2;
        const maxWind = 3; // Range: -3 to 3
        
        // Normalize x from 0..width to -1..1
        // Use a deadzone in center? Maybe not needed for fluid feel.
        const normalized = (x - center) / (width / 2);
        
        // Map to wind range
        let wind = normalized * maxWind;
        
        // Clamp
        wind = Math.max(-maxWind, Math.min(maxWind, wind));
        
        // Update Config
        FIRE_CONFIG.wind = wind;
        
        // Update UI
        const windSlider = document.getElementById('baseHeat');
        const windVal = document.getElementById('baseHeatValue');
        if (windSlider) windSlider.value = wind;
        if (windVal) windVal.innerText = wind.toFixed(2);
    };

    // document.addEventListener('mousemove', (e) => handleWindInput(e.clientX));
    // document.addEventListener('touchmove', (e) => {
    //     if (e.touches.length > 0) {
    //         handleWindInput(e.touches[0].clientX);
    //     }
    // }, { passive: true });

    // Setup UI Controls
    setupControls();

    // Initial Setup
    handleResize();
    
    // Start Loop
    if (!isRunning) {
        isRunning = true;
        requestAnimationFrame(loop);
    }
}

function handleResize() {
    if (!artPre) return;

    // Parse the art to determine grid size
    const text = artPre.textContent || "";
    const lines = text.split('\n');
    
    // Calculate max width
    let maxLineLength = 0;
    for (const line of lines) {
        if (line.length > maxLineLength) maxLineLength = line.length;
    }

    // Determine Resolution
    // We use the resolution slider to scale the internal grid relative to the font size
    // But to match the ASCII art exactly, we should probably stick to 1:1 or integer multiples
    const scale = FIRE_CONFIG.resolutionScale || 1.0;
    
    FIRE_CONFIG.width = Math.ceil(maxLineLength * scale);
    FIRE_CONFIG.height = Math.ceil(lines.length * scale);
    
    // ensure min size
    if (FIRE_CONFIG.width < 10) FIRE_CONFIG.width = 10;
    if (FIRE_CONFIG.height < 5) FIRE_CONFIG.height = 5;

    console.log(`Grid Size: ${FIRE_CONFIG.width}x${FIRE_CONFIG.height}`);

    // Re-initialize arrays
    const size = FIRE_CONFIG.width * FIRE_CONFIG.height;
    firePixels = new Array(size).fill(0);
    solidMap = new Array(size).fill(false);
    sourceMap = new Array(size).fill(false);

    // Parse Maps (Solid and Source)
    parseMaps(lines, scale);

    // Setup Overlay
    setupOverlay();
}

function parseMaps(lines, scale) {
    const w = FIRE_CONFIG.width;
    const h = FIRE_CONFIG.height;
    
    // Reset maps
    solidMap.fill(false);
    sourceMap.fill(false);

    sourceMap[41*w+26] = true;

    sourceMap[27*w+168] = true;
    sourceMap[28*w+169] = true;
    sourceMap[26*w+168] = true;
    sourceMap[25*w+168] = true;
    sourceMap[24*w+163] = true;
    sourceMap[20*w+165] = true;
    sourceMap[19*w+163] = true;



    sourceMap[40*w+216] = true;


    sourceMap[119*w+22] = true;
    sourceMap[119*w+26] = true;
    sourceMap[119*w+30] = true;
    sourceMap[119*w+34] = true;
    sourceMap[119*w+38] = true;
    sourceMap[119*w+40] = true;
    sourceMap[119*w+44] = true;
    sourceMap[119*w+48] = true;
    sourceMap[119*w+52] = true;
    sourceMap[119*w+56] = true;
    sourceMap[119*w+60] = true;
    sourceMap[119*w+64] = true;
    sourceMap[119*w+68] = true;
    sourceMap[119*w+72] = true;

    sourceMap[119*w+170] = true;
    sourceMap[119*w+174] = true;
    sourceMap[119*w+178] = true;
    sourceMap[119*w+182] = true;
    sourceMap[119*w+186] = true;
    sourceMap[119*w+190] = true;
    sourceMap[119*w+194] = true;
    sourceMap[119*w+198] = true;
    sourceMap[119*w+202] = true;
    sourceMap[119*w+206] = true;
    sourceMap[119*w+210] = true;
    sourceMap[119*w+214] = true;


    const srcCharLower = FIRE_CONFIG.sourceChar.toLowerCase();

    for (let y = 0; y < lines.length; y++) {
        const line = lines[y];
        for (let x = 0; x < line.length; x++) {
            const char = line[x];
            const charLower = char.toLowerCase();

            // Calculate mapped grid coordinates
            // We might map one char to multiple grid cells if scale > 1
            const startX = Math.floor(x * scale);
            const startY = Math.floor(y * scale);
            const endX = Math.floor((x + 1) * scale);
            const endY = Math.floor((y + 1) * scale);

            if (charLower === srcCharLower) {
                // It's a fire source
                for (let gy = startY; gy < endY; gy++) {
                    for (let gx = startX; gx < endX; gx++) {
                        if (gx < w && gy < h) {
                            //sourceMap[gy * w + gx] = true;
                        }

                    }
                }
            } else if (char.trim().length > 0) {
                // It's a wall (any non-whitespace, non-source char)
                for (let gy = startY; gy < endY; gy++) {
                    for (let gx = startX; gx < endX; gx++) {
                        if (gx < w && gy < h) {
                            solidMap[gy * w + gx] = true;
                        }

                    }

                }
            }
        }
    }
    
    // Add solid boundary box
    // Top and Bottom
    for (let x = 0; x < w; x++) {
        if (!sourceMap[x]) solidMap[x] = true;                 // Top row (y=0)
        const bottomIdx = (h - 1) * w + x;
        if (!sourceMap[bottomIdx]) solidMap[bottomIdx] = true;   // Bottom row (y=h-1)
    }
    // Left and Right
    for (let y = 0; y < h; y++) {
        const leftIdx = y * w;
        if (!sourceMap[leftIdx]) solidMap[leftIdx] = true;             // Left column (x=0)
        const rightIdx = y * w + (w - 1);
        if (!sourceMap[rightIdx]) solidMap[rightIdx] = true;   // Right column (x=w-1)
    }
    
    console.log(`Found ${sourceMap.filter(x=>x).length} source cells and ${solidMap.filter(x=>x).length} wall cells.`);
}

function setupOverlay() {
    if (fireOverlay) {
        fireOverlay.remove();
    }

    fireOverlay = document.createElement('pre');
    fireOverlay.id = 'doomFireOverlay';
    
    // Copy styles
    const style = window.getComputedStyle(artPre);
    fireOverlay.style.position = 'absolute';
    fireOverlay.style.top = '0';
    fireOverlay.style.left = '0';
    // Do NOT copy margin, as the overlay is absolute inside the parent
    fireOverlay.style.margin = '0'; 
    fireOverlay.style.padding = style.padding;
    fireOverlay.style.fontFamily = style.fontFamily;
    fireOverlay.style.letterSpacing = style.letterSpacing;
    fireOverlay.style.textAlign = style.textAlign;
    
    // Adjust font size and line height for resolution scale
    fireOverlay.style.fontSize = `${parseFloat(style.fontSize) / (FIRE_CONFIG.resolutionScale || 1.0)}px`;
    fireOverlay.style.lineHeight = `${parseFloat(style.lineHeight) / (FIRE_CONFIG.resolutionScale || 1.0)}px`;
    fireOverlay.style.whiteSpace = 'pre';
    fireOverlay.style.webkitFontSmoothing = 'none';
    fireOverlay.style.MozOsxFontSmoothing = 'unset';
    fireOverlay.style.textRendering = 'geometricPrecision';
    fireOverlay.style.pointerEvents = 'none'; // Click through
    fireOverlay.style.zIndex = '999';
    fireOverlay.style.width = '100%';
    fireOverlay.style.height = '100%';
    fireOverlay.style.overflow = 'hidden';
    // Transparent background
    fireOverlay.style.backgroundColor = 'rgba(0,0,0,0)'; 

    // Ensure parent is relative
    if (artPre.parentElement) {
        artPre.parentElement.style.position = 'relative';
        artPre.parentElement.appendChild(fireOverlay);
    }
}

// -----------------------------------------------------------------------------
// Simulation
// -----------------------------------------------------------------------------

function calculateFirePropagation() {
    const w = FIRE_CONFIG.width;
    const h = FIRE_CONFIG.height;
    const spread = FIRE_CONFIG.spread;
    const decayBase = FIRE_CONFIG.decay;
    const wind = Math.round(FIRE_CONFIG.wind);

    // 1. Feed Sources
    for (let i = 0; i < w * h; i++) {
        if (sourceMap[i]) {
            firePixels[i] = 40; // Max intensity
        }
    }

    // 2. Propagate
    // Iterate from x=0..w, y=1..h. 
    // Calculate new value for pixel (x, y-1) based on (x,y)
    for (let x = 0; x < w; x++) {
        for (let y = 1; y < h; y++) {
            const srcIndex = y * w + x;
            const pixelIntensity = firePixels[srcIndex];

            // Calculate random spread/wind offset
            // rand between 0 and 3 typically
            const rand = Math.floor(Math.random() * (spread + 1)); 
            const windOffset = wind; 
            
            // Destination X. 
            // If wind is positive (blowing right), fire should shift right as it goes up.
            // So dstX = x + rand + windOffset;
            let dstX = x + rand + windOffset;
            
            // Wrap or Clamp? Clamp.
            if (dstX >= w) dstX = w - 1;
            if (dstX < 0) dstX = 0;
            
            const dstY = y - 1;
            const dstIndex = dstY * w + dstX;
            
            // Decay intensity
            const decayAmount = Math.floor(Math.random() * decayBase);
            let newIntensity = pixelIntensity - decayAmount;
            if (newIntensity < 0) newIntensity = 0;
            
            // Logic to handle walls and spread
            // We use Math.max to merge fire values, allowing accumulation against walls
            
            let targetIndex = dstIndex;
            let isSolid = false;
            
            if (isSolid) {
                 // Hit a wall directly (e.g. ceiling)
                 // Bounce back / stay at source level
                 targetIndex = srcIndex;
                 
                 // Spread horizontally at SOURCE level
                 const left = y * w + Math.max(0, x - 1);
                 const right = y * w + Math.min(w - 1, x + 1);
                 
                 if (!solidMap[left]) firePixels[left] = Math.max(firePixels[left] - 1, newIntensity - 1);
                 if (!solidMap[right]) firePixels[right] = Math.max(firePixels[right] - 1, newIntensity - 1);
            } else {
                 // Destination is clear
                 // Check if destination is BLOCKED ABOVE (ceiling effect)
                 // This helps visual spread when fire rises to a ceiling
                 const aboveIndex = Math.max(0, dstY - 1) * w + dstX;
                 if (solidMap[aboveIndex]) {
                     // Spread horizontally at DEST level
                     const left = dstY * w + Math.max(0, dstX - 1);
                     const right = dstY * w + Math.min(w - 1, dstX + 1);
                     
                     if (!solidMap[left]) firePixels[left] = Math.max(firePixels[left] - 1, newIntensity - 1);
                     if (!solidMap[right]) firePixels[right] = Math.max(firePixels[right] - 1, newIntensity - 1);
                 }
            }

            // Update target (either dst or src if bounced)
            // We decay the existing value to prevent infinite heat accumulation
            // But we take Max to blend streams
            const existing = firePixels[targetIndex];
            // Simple decay for existing value
            const decayedExisting = Math.max(0, existing - decayBase); 
            
            firePixels[targetIndex] = Math.max(decayedExisting, newIntensity);
        }
    }
}

// -----------------------------------------------------------------------------
// Rendering
// -----------------------------------------------------------------------------

function render() {
    if (!fireOverlay) return;

    const w = FIRE_CONFIG.width;
    const h = FIRE_CONFIG.height;
    
    let html = '';

    // We build the string manually.
    // Optimization: array of strings join is often faster than string concatenation in older engines,
    // but V8 optimizes += well. Let's use += for simplicity.
    
    for (let y = 0; y < h; y++) {
        // Optimization: If a row is completely empty, we might skip? No, we need to preserve spacing.
        let rowHtml = '';
        for (let x = 0; x < w; x++) {
            const idx = y * w + x;
            const intensity = firePixels[idx];
            
            if (solidMap[idx]) {
                // Wall - transparent
                rowHtml += ' '; 
            } else if (intensity <= 0) {
                // Empty air - transparent
                rowHtml += ' '; 
            } else {
                // Fire pixel
                // Safe intensity clamp
                const safeInt = Math.min(36, Math.max(0, intensity));
                
                const color = FIRE_PALETTE[safeInt];
                const colorStr = `rgb(${color.r},${color.g},${color.b})`;
                
                // Map intensity to char. 
                // 0->0, 36->last
                const charIdx = Math.floor((safeInt / 36) * (FIRE_CHARS.length - 1));
                const char = FIRE_CHARS[charIdx];
                
                rowHtml += `<span style="color:${colorStr}">${char}</span>`;
            }
        }
        html += rowHtml + '\n';
    }

    fireOverlay.innerHTML = html;
    
    // Debug FPS
    const now = performance.now();
    const delta = now - lastFrameTime;
    if (delta > 1000) {
        const fps = Math.round(FIRE_CONFIG.frameCount / (delta / 1000));
        // console.log(`FPS: ${fps}`); // Optional debug
        const fpsEl = document.getElementById('fpsDisplay');
        if(fpsEl) fpsEl.innerText = fps;
        
        FIRE_CONFIG.frameCount = 0;
        lastFrameTime = now;
    }
    FIRE_CONFIG.frameCount = (FIRE_CONFIG.frameCount || 0) + 1;
}

// -----------------------------------------------------------------------------
// Loop & Controls
// -----------------------------------------------------------------------------

let lastLoopTime = 0;

function loop(timestamp) {
    if (!isRunning) return;
    
    // Calculate time elapsed since last frame
    const elapsed = timestamp - lastLoopTime;
    
    // Calculate frame duration based on target FPS
    const frameDuration = 1000 / FIRE_CONFIG.fps;
    
    if (elapsed >= frameDuration) {
        // Adjust lastLoopTime to maintain consistent pacing, accounting for drift
        // We subtract the remainder to keep the cadence aligned with the target interval
        lastLoopTime = timestamp - (elapsed % frameDuration);
        
        calculateFirePropagation();
        render();
    }
    
    requestAnimationFrame(loop);
}

function setupControls() {
    const bindSlider = (id, configKey, transform = v => parseFloat(v)) => {
        const el = document.getElementById(id);
        const valEl = document.getElementById(id + 'Value');
        if (el) {
            // Set initial value from config
            if (transform === parseInt) el.value = FIRE_CONFIG[configKey]; // integer sliders
            
            el.addEventListener('input', (e) => {
                const val = transform(e.target.value);
                FIRE_CONFIG[configKey] = val;
                if (valEl) valEl.innerText = val;
                
                // Special handling for resolution
                if (configKey === 'resolutionScale') {
                    // Debounce resize?
                    // For now direct call
                    FIRE_CONFIG.resolutionScale = val;
                    handleResize();
                }
            });
        }
    };
    
    // Map HTML IDs to Config Keys
    bindSlider('maxFire', 'intensity', parseInt);
    bindSlider('decay', 'decay', parseInt);
    bindSlider('spread', 'spread', parseInt);
    bindSlider('baseHeat', 'wind', parseFloat); // Assuming baseHeat slider reused for wind
    bindSlider('resolution', 'resolutionScale', parseFloat);
    bindSlider('targetFPS', 'fps', parseInt);
    
    const sourceInput = document.getElementById('sourceChar');
    if (sourceInput) {
        sourceInput.addEventListener('input', (e) => {
            if(e.target.value.length > 0) {
                FIRE_CONFIG.sourceChar = e.target.value[0];
                handleResize(); // Re-parse map
            }
        });
    }
    
    const pauseBtn = document.getElementById('pauseButton');
    if(pauseBtn) {
        pauseBtn.addEventListener('click', () => {
            isRunning = !isRunning;
            if(isRunning) loop();
        });
    }
}

// Boot
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}

    init();