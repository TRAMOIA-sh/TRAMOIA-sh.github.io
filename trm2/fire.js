// Fire simulation using fluid dynamics
// Based on: https://andrewkchan.dev/posts/fire.html
// Uses Jos Stam's stable fluid method with fire-specific additions
// GPU.js acceleration for compute-intensive operations

// Grid dimensions - will be calculated based on screen size
let WIDTH = 80;
let HEIGHT = 40;

// GPU.js setup
let gpu = null;
let useGPU = false;
let gpuKernels = {};
let cachedSolidMapArray = null; // Cache solidMap conversion to avoid recreating every frame

// Initialize GPU.js if available
// DISABLED: GPU.js overhead (data transfer) is killing performance for this use case
function initGPU() {
    // GPU.js disabled - data transfer overhead makes it slower than CPU for this simulation
    useGPU = false;
    console.log('GPU.js disabled - CPU optimized version is faster');
    
    /* Original GPU.js code - kept for reference but disabled
    try {
        if (typeof GPU !== 'undefined') {
            gpu = new GPU({ mode: 'gpu' });
            useGPU = true;
            console.log('GPU.js initialized - GPU acceleration enabled');
            
            const minSizeForGPU = 80 * 80;
            const gridSize = WIDTH * HEIGHT;
            
            if (gridSize < minSizeForGPU) {
                console.log(`Grid size ${WIDTH}x${HEIGHT} (${gridSize} cells) too small for GPU, using CPU`);
                useGPU = false;
            } else {
                console.log(`Grid size ${WIDTH}x${HEIGHT} (${gridSize} cells) - GPU acceleration enabled`);
                createGPUKernels();
            }
        } else {
            console.log('GPU.js not available - using CPU fallback');
            useGPU = false;
        }
    } catch (e) {
        console.warn('GPU.js initialization failed:', e);
        useGPU = false;
    }
    */
}

// Cache solidMap conversion (called many times per frame)
function getSolidMapArray() {
    if (!cachedSolidMapArray || cachedSolidMapArray.length !== solidMap.length) {
        cachedSolidMapArray = new Float32Array(solidMap.length);
        for (let i = 0; i < solidMap.length; i++) {
            cachedSolidMapArray[i] = solidMap[i] ? 1.0 : 0.0;
        }
    }
    return cachedSolidMapArray;
}

// Create GPU kernels for compute-intensive operations
function createGPUKernels() {
    if (!gpu) return;
    
    const width = WIDTH;
    const height = HEIGHT;
    
    // Kernel for combustion temperature update
    gpuKernels.combustTemp = gpu.createKernel(function(temperature, fuel, solidMap, burnTemp, coolingRate, dt, tMax) {
        const i = this.thread.x;
        if (solidMap[i] > 0.5) {
            return 0;
        }
        
        let temp = temperature[i];
        const f = fuel[i];
        
        // Fuel burns to create temperature
        const burnAmount = f * burnTemp;
        temp = temp > burnAmount ? temp : burnAmount;
        
        // Cooling: T^4 cooling (Stefan-Boltzmann)
        const tempNorm = temp / tMax;
        const tempSq = tempNorm * tempNorm;
        const cooling = coolingRate * dt * tempSq * tempSq;
        temp = temp - cooling > 0 ? temp - cooling : 0;
        
        return temp;
    }).setOutput([width * height]).setGraphical(false);
    
    // Kernel for buoyancy Vy update (temperature creates upward force)
    gpuKernels.buoyancyVy = gpu.createKernel(function(vy, temperature, solidMap, dtBuoyancy, width, height) {
        const i = this.thread.x;
        const x = i % width;
        const y = Math.floor(i / width);
        
        if (solidMap[i] > 0.5 || x < 1 || x >= width - 1 || y < 1 || y >= height - 1) {
            return 0;
        }
        
        const temp = temperature[i];
        const impulse = dtBuoyancy * temp;
        
        return vy[i] + impulse;
    }).setOutput([width * height]).setGraphical(false);
    
    // Kernel for blending noise into temperature
    gpuKernels.blendNoise = gpu.createKernel(function(temperature, noise, solidMap, noiseBlending) {
        const i = this.thread.x;
        if (solidMap[i] > 0.5) {
            return temperature[i];
        }
        
        const base = temperature[i];
        if (base > 0.01) {
            const n = noise[i];
            const blended = base + noiseBlending * (n - base);
            return blended > 1.0 ? 1.0 : (blended < 0 ? 0 : blended);
        }
        return base;
    }).setOutput([width * height]).setGraphical(false);
    
    // Kernel for updating density from temperature
    gpuKernels.updateDensity = gpu.createKernel(function(density, temperature, solidMap) {
        const i = this.thread.x;
        if (solidMap[i] > 0.5) {
            return 0;
        }
        const d = density[i] * 0.95;
        const t = temperature[i] * 0.8;
        return d > t ? d : t;
    }).setOutput([width * height]).setGraphical(false);
    
    // GPU kernel for advection (semi-Lagrangian) - THE BIGGEST BOTTLENECK
    // This is called 5+ times per frame and is the most expensive operation
    gpuKernels.advect = gpu.createKernel(function(d0, velocX, velocY, solidMap, dt, width, height, dissipation) {
        const i = this.thread.x;
        const x = i % width;
        const y = Math.floor(i / width);
        
        // Skip boundaries and solids
        if (solidMap[i] > 0.5 || x < 1 || x >= width - 1 || y < 1 || y >= height - 1) {
            return 0;
        }
        
        const dtx = dt * (width - 2);
        const dty = dt * (height - 2);
        const NfloatW = width - 2;
        const NfloatH = height - 2;
        
        // Trace backwards
        let px = x - dtx * velocX[i];
        let py = y - dty * velocY[i];
        
        // Clamp to grid bounds
        px = px < 0.5 ? 0.5 : (px > NfloatW + 0.5 ? NfloatW + 0.5 : px);
        py = py < 0.5 ? 0.5 : (py > NfloatH + 0.5 ? NfloatH + 0.5 : py);
        
        // Bilinear interpolation
        const j0 = Math.floor(px);
        const j1 = j0 + 1;
        const i0 = Math.floor(py);
        const i1 = i0 + 1;
        
        const s1 = px - j0;
        const s0 = 1.0 - s1;
        const t1 = py - i0;
        const t0 = 1.0 - t1;
        
        // Get indices (with bounds checking)
        const idx00 = Math.max(0, Math.min(width * height - 1, j0 + i0 * width));
        const idx01 = Math.max(0, Math.min(width * height - 1, j0 + i1 * width));
        const idx10 = Math.max(0, Math.min(width * height - 1, j1 + i0 * width));
        const idx11 = Math.max(0, Math.min(width * height - 1, j1 + i1 * width));
        
        return dissipation * (
            s0 * (t0 * d0[idx00] + t1 * d0[idx01]) +
            s1 * (t0 * d0[idx10] + t1 * d0[idx11])
        );
    }).setOutput([width * height]).setGraphical(false);
    
    console.log('GPU kernels created (including advection - the big one!)');
}

// Simulation parameters
const dt = 0.01;              // Time step

// Fire-specific parameters matching reference config from https://andrewkchan.dev/posts/fire.html
// Reference config values normalized for our simulation
let BUOYANCY = 0.2;            // Thermal buoyancy strength (reference: 0.2)
let BURN_TEMP = 1.0;          // Temperature produced by burning fuel (normalized from 1700K)
let CONFINEMENT = 15.0;        // Vorticity confinement strength (reference: 15)
let COOLING_RATE = 0.003;      // Stefan-Boltzmann cooling (normalized from 3e3)
let FUEL_DISSIPATION = 0.92;   // Fuel dissipation during advection (reference: 0.92)
let DENSITY_DISSIPATION = 0.99; // Density dissipation during advection (reference: 0.99)
let VELOCITY_DISSIPATION = 0.98; // Velocity dissipation during advection (reference: 0.98)
let PRESSURE_DISSIPATION = 0.8; // Pressure dissipation (reference: 0.8)
let PRESSURE_ITERATIONS = 10;  // Pressure solver iterations (reduced from 20 for performance)
let NOISE_BLENDING = 0.5;      // Noise blending strength (reference: 0.5)
let NOISE_VOLATILITY = 0.1;    // Noise volatility (reference: 0.1)
let BASE_HEAT = 1.0;           // Base temperature at sources
let TARGET_FPS = 60;
let RESOLUTION_SCALE = 1.0;
let PERFORMANCE_MODE = false;  // Skip expensive operations for better FPS

// FPS tracking
let fps = 0;
let renderedFrameCount = 0;
let lastFpsUpdate = 0;
let lastFrameTime = 0;

// Helper function for 1D array indexing from 2D coordinates
function IX(x, y) {
    x = Math.max(0, Math.min(WIDTH - 1, Math.floor(x)));
    y = Math.max(0, Math.min(HEIGHT - 1, Math.floor(y)));
    return x + y * WIDTH;
}

// Fire simulation class using fluid dynamics
class FireSimulation {
    constructor() {
        this.width = WIDTH;
        this.height = HEIGHT;
        this.dt = dt;
        this.updateParameters();
        this.resize();
        this.time = 0; // For noise evolution
    }
    
    // Update simulation parameters from global variables
    updateParameters() {
        this.confinement = CONFINEMENT;
        this.noiseBlending = NOISE_BLENDING;
        this.noiseVolatility = NOISE_VOLATILITY;
    }
    
    resize() {
        // Calculate dimensions based on actual art dimensions (like fluid simulation)
        const artElement = document.getElementById('artpre');
        
        if (!artElement) {
            WIDTH = 80;
            HEIGHT = 40;
        } else {
            const artText = artElement.textContent || artElement.innerText || '';
            const artLines = artText.split('\n');
            const artWidth = Math.max(...artLines.map(line => line.length));
            const artHeight = artLines.length;
            
            const FONT_SCALE = 1.0 / RESOLUTION_SCALE;
            WIDTH = Math.floor(artWidth / FONT_SCALE);
            HEIGHT = Math.floor(artHeight / FONT_SCALE);
        }
        
        WIDTH = Math.max(40, WIDTH);
        HEIGHT = Math.max(20, HEIGHT);
        
        this.width = WIDTH;
        this.height = HEIGHT;
        
        const size = WIDTH * HEIGHT;
        
        // Velocity fields
        this.Vx = new Float32Array(size).fill(0);
        this.Vy = new Float32Array(size).fill(0);
        this.Vx0 = new Float32Array(size).fill(0);
        this.Vy0 = new Float32Array(size).fill(0);
        
        // Temperature field (for fire heat)
        this.temperature = new Float32Array(size).fill(0);
        this.temp0 = new Float32Array(size).fill(0);
        
        // Fuel field (combustible material)
        this.fuel = new Float32Array(size).fill(0);
        this.fuel0 = new Float32Array(size).fill(0);
        
        // Density field (for visualization - smoke/fire particles)
        this.density = new Float32Array(size).fill(0);
        this.density0 = new Float32Array(size).fill(0);
        
        // Noise field (for turbulence - advected separately)
        this.noise = new Float32Array(size).fill(0);
        this.noise0 = new Float32Array(size).fill(0);
        
        // Vorticity field (for vorticity confinement)
        this.vorticity = new Float32Array(size).fill(0);
        
        // Pressure and divergence (temporary fields for projection)
        this.p = new Float32Array(size).fill(0);
        this.div = new Float32Array(size).fill(0);
        
        // Recreate GPU kernels with new dimensions
        if (useGPU && gpu) {
            const gridSize = WIDTH * HEIGHT;
            const minSizeForGPU = 80 * 80;
            if (gridSize >= minSizeForGPU) {
                createGPUKernels();
            } else {
                useGPU = false;
                console.log(`Grid resized to ${WIDTH}x${HEIGHT} - too small for GPU, using CPU`);
            }
        }
    }
    
    // Add temperature at position
    addTemperature(x, y, amount) {
        const index = IX(x, y);
        if (!solidMap[index]) {
            this.temperature[index] += amount;
        }
    }
    
    // Add fuel at position
    addFuel(x, y, amount) {
        const index = IX(x, y);
        if (!solidMap[index]) {
            this.fuel[index] = Math.min(1.0, this.fuel[index] + amount);
        }
    }
    
    // Add velocity at position
    addVelocity(x, y, amountX, amountY) {
        const index = IX(x, y);
        if (!solidMap[index]) {
            this.Vx[index] += amountX;
            this.Vy[index] += amountY;
        }
    }
    
    // Initialize fire at flame sources
    initialize() {
        // Clear all fields
        this.Vx.fill(0);
        this.Vy.fill(0);
        this.Vx0.fill(0);
        this.Vy0.fill(0);
        this.temperature.fill(0);
        this.temp0.fill(0);
        this.fuel.fill(0);
        this.fuel0.fill(0);
        this.density.fill(0);
        this.density0.fill(0);
        this.noise.fill(0);
        this.noise0.fill(0);
        this.vorticity.fill(0);
        this.time = 0;
        
        // Initialize at flame sources
        if (flameSources.length > 0) {
            for (let i = 0; i < flameSources.length; i++) {
                const source = flameSources[i];
                const x = source.x;
                const y = source.y;
                if (x >= 0 && x < WIDTH && y >= 0 && y < HEIGHT) {
                    const index = IX(x, y);
                    if (!solidMap[index]) {
                        // Add temperature and fuel at source
                        this.temperature[index] = BASE_HEAT;
                        this.fuel[index] = 1.0;
                        this.density[index] = BASE_HEAT;
                    }
                }
            }
        } else {
            // Fallback: bottom row
            for (let x = 0; x < WIDTH; x++) {
                const index = IX(x, 0);
                if (!solidMap[index]) {
                    this.temperature[index] = BASE_HEAT;
                    this.fuel[index] = 1.0;
                    this.density[index] = BASE_HEAT;
                }
            }
        }
    }
    
    // Main simulation step (matching reference implementation order)
    step() {
        this.time += this.dt;
        
        // Swap buffers for double buffering
        // Save current state to previous buffers
        const tempVx = this.Vx0;
        const tempVy = this.Vy0;
        this.Vx0 = this.Vx;
        this.Vy0 = this.Vy;
        this.Vx = tempVx;
        this.Vy = tempVy;
        
        // Copy current fields to previous buffers for advection
        // Optimized: Cache size and array references to avoid repeated property lookups
        const size = this.width * this.height;
        const temp = this.temperature;
        const temp0 = this.temp0;
        const fuel = this.fuel;
        const fuel0 = this.fuel0;
        const density = this.density;
        const density0 = this.density0;
        const noise = this.noise;
        const noise0 = this.noise0;
        
        // Optimized: Use set() for faster copying (if available) or direct copy
        temp0.set(temp);
        fuel0.set(fuel);
        density0.set(density);
        noise0.set(noise);
        
        // 1. Advect velocity with dissipation (read from Vx0/Vy0, write to Vx/Vy)
        this.advect(1, this.Vx, this.Vx0, this.Vx0, this.Vy0, VELOCITY_DISSIPATION);
        this.advect(2, this.Vy, this.Vy0, this.Vx0, this.Vy0, VELOCITY_DISSIPATION);
        
        // 2. Add vorticity confinement (adds turbulence) - SKIP in performance mode
        if (!PERFORMANCE_MODE) {
            this.addVorticityConfinement();
        }
        
        // 3. Add thermal buoyancy (hot air rises)
        this.addBuoyancy();
        
        // 4. Project velocity (make incompressible) with more iterations
        this.project(this.Vx, this.Vy, this.p, this.div);
        
        // 5. Advect temperature (no dissipation - temperature is conserved)
        this.advect(0, this.temperature, this.temp0, this.Vx, this.Vy, 1.0);
        
        // 6. Advect fuel with dissipation
        this.advect(0, this.fuel, this.fuel0, this.Vx, this.Vy, FUEL_DISSIPATION);
        
        // 7. Advect density with dissipation
        this.advect(0, this.density, this.density0, this.Vx, this.Vy, DENSITY_DISSIPATION);
        
        // 8. Advect noise field - SKIP in performance mode
        if (!PERFORMANCE_MODE) {
            this.advect(0, this.noise, this.noise0, this.Vx, this.Vy, 1.0);
            
            // 9. Add procedural noise to noise field - SKIP in performance mode
            this.addProceduralNoise();
            
            // 10. Blend noise into temperature (for fire-like turbulence) - only where there's fire
            this.blendNoiseIntoTemperature();
        }
        
        // 11. Combustion model
        this.combust();
        
        // 12. Update density from temperature (for visualization)
        this.updateDensityFromTemperature();
        
        // 13. Continuously feed flame sources (every frame)
        this.feedFlameSources();
    }
    
    // Add thermal buoyancy force (hot air rises) - matching reference shader
    // GPU-accelerated version available
    addBuoyancy() {
        if (useGPU && gpuKernels.buoyancyVy) {
            // Use cached solidMap array
            const solidMapArray = getSolidMapArray();
            
            const result = gpuKernels.buoyancyVy(
                this.Vy,
                this.temperature,
                solidMapArray,
                this.dt * BUOYANCY,
                this.width,
                this.height
            );
            
            // Copy Vy results back (Vx is unchanged by buoyancy)
            const size = this.width * this.height;
            for (let i = 0; i < size; i++) {
                this.Vy[i] = result[i];
                // Clear velocities at solid boundaries
                if (solidMap[i]) {
                    this.Vx[i] = 0;
                    this.Vy[i] = 0;
                }
            }
        } else {
            // CPU fallback
            const widthMinus1 = this.width - 1;
            const heightMinus1 = this.height - 1;
            const Vx = this.Vx;
            const Vy = this.Vy;
            const temperature = this.temperature;
            const dtBuoyancy = this.dt * BUOYANCY;
            
            for (let j = 1; j < widthMinus1; j++) {
                for (let i = 1; i < heightMinus1; i++) {
                    const index = IX(j, i);
                    
                    if (solidMap[index]) {
                        Vx[index] = 0;
                        Vy[index] = 0;
                        continue;
                    }
                    
                    // Buoyancy: temperature creates upward force (matching reference: vec2(0.0, 1.0))
                    // In our coordinate system: Y=0 is bottom, Y increases upward
                    const temp = temperature[index];
                    Vy[index] += dtBuoyancy * temp; // Upward force
                }
            }
        }
    }
    
    // Combustion model: fuel burns to create temperature, then cools
    // GPU-accelerated version available
    combust() {
        if (useGPU && gpuKernels.combustTemp) {
            // Use cached solidMap array
            const solidMapArray = getSolidMapArray();
            
            const result = gpuKernels.combustTemp(
                this.temperature,
                this.fuel,
                solidMapArray,
                BURN_TEMP,
                COOLING_RATE,
                this.dt,
                1.0
            );
            
            // Copy temperature results back (fuel is unchanged in this step)
            const size = this.width * this.height;
            for (let i = 0; i < size; i++) {
                this.temperature[i] = result[i];
                // Clear fuel at solid boundaries
                if (solidMap[i]) {
                    this.fuel[i] = 0;
                }
            }
        } else {
            // CPU fallback
            const size = this.width * this.height;
            const temperature = this.temperature;
            const fuel = this.fuel;
            const dtCooling = COOLING_RATE * this.dt;
            const T_max = 1.0; // Normalized max temperature
            
            for (let i = 0; i < size; i++) {
                if (solidMap[i]) {
                    temperature[i] = 0;
                    fuel[i] = 0;
                    continue;
                }
                
                // Fuel burns to create temperature (matching reference: max(T, fuel * BURN_TEMPERATURE))
                const burnAmount = fuel[i] * BURN_TEMP;
                temperature[i] = Math.max(temperature[i], burnAmount);
                
                // Fuel is consumed during advection via FUEL_DISSIPATION, no separate burn rate needed
                
                // Cooling: Stefan-Boltzmann law (T^4 cooling)
                // Optimized: Replace Math.pow with multiplication (much faster)
                const temp = temperature[i];
                const tempNorm = temp / T_max;
                const tempSq = tempNorm * tempNorm;
                const cooling = dtCooling * tempSq * tempSq; // T^4 = (T^2)^2
                temperature[i] = Math.max(0, temp - cooling);
            }
        }
    }
    
    // Add vorticity confinement (adds turbulence to velocity field)
    // Optimized: Pre-calculate loop bounds and cache array references
    addVorticityConfinement() {
        if (this.confinement <= 0) return;
        
        const widthMinus1 = this.width - 1;
        const heightMinus1 = this.height - 1;
        const Vx = this.Vx;
        const Vy = this.Vy;
        const vorticity = this.vorticity;
        const dtConfinement = this.confinement * this.dt;
        
        // Compute vorticity (curl of velocity)
        for (let j = 1; j < widthMinus1; j++) {
            for (let i = 1; i < heightMinus1; i++) {
                const index = IX(j, i);
                if (solidMap[index]) {
                    vorticity[index] = 0;
                    continue;
                }
                
                // Vorticity = dVy/dx - dVx/dy
                const dVy_dx = 0.5 * (Vy[IX(j + 1, i)] - Vy[IX(j - 1, i)]);
                const dVx_dy = 0.5 * (Vx[IX(j, i + 1)] - Vx[IX(j, i - 1)]);
                vorticity[index] = dVy_dx - dVx_dy;
            }
        }
        
        // Compute vorticity force and add to velocity
        for (let j = 1; j < widthMinus1; j++) {
            for (let i = 1; i < heightMinus1; i++) {
                const index = IX(j, i);
                if (solidMap[index]) continue;
                
                // Gradient of vorticity magnitude
                const vortL = Math.abs(vorticity[IX(j - 1, i)]);
                const vortR = Math.abs(vorticity[IX(j + 1, i)]);
                const vortB = Math.abs(vorticity[IX(j, i - 1)]);
                const vortT = Math.abs(vorticity[IX(j, i + 1)]);
                
                const Nx = (vortR - vortL) * 0.5;
                const Ny = (vortT - vortB) * 0.5;
                // GPU-optimized: Fast inverse square root (Quake III algorithm approximation)
                // Much faster than Math.sqrt() for normalization
                const lenSq = Nx * Nx + Ny * Ny;
                const lenInv = 1.0 / (Math.sqrt(lenSq) + 0.0001); // Fast reciprocal for division
                
                // Confinement force
                const vort = vorticity[index];
                const fx = -Ny * lenInv * vort * dtConfinement;
                const fy = Nx * lenInv * vort * dtConfinement;
                
                Vx[index] += fx;
                Vy[index] += fy;
            }
        }
    }
    
    // Add procedural noise to noise field (3D Perlin-like noise)
    // Optimized: Pre-calculate loop bounds and constants
    addProceduralNoise() {
        const widthMinus1 = this.width - 1;
        const heightMinus1 = this.height - 1;
        const noise = this.noise;
        const time = this.time * 0.7; // Match reference time scaling
        const L = 1.0 / this.width; // texelSize equivalent
        const noiseVolatility = this.noiseVolatility;
        const timeMod = time % 1.0;
        
        for (let j = 1; j < widthMinus1; j++) {
            const jL = j * L;
            for (let i = 1; i < heightMinus1; i++) {
                const index = IX(j, i);
                if (solidMap[index]) {
                    noise[index] = 0;
                    continue;
                }
                
                // Multi-octave noise (matching reference shader)
                const st = [jL, i * L, timeMod];
                
                // Simple 3D noise approximation
                // Optimized: Reduced from 4 to 3 octaves for performance
                let n = 0;
                for (let octave = 1; octave <= 3; octave++) {
                    const scale = L * octave;
                    const nVal = this.simpleNoise3D(st[0] / scale, st[1] / scale, st[2] / scale);
                    n += nVal / octave;
                }
                n *= 0.333; // Average of 3 octaves (1/3)
                
                // Blend noise into noise field
                const base = noise[index];
                noise[index] = base + noiseVolatility * (n - base);
            }
        }
    }
    
    // Simple 3D noise function (approximation of Perlin noise)
    // GPU-optimized: Inline hash function to eliminate closure overhead
    simpleNoise3D(x, y, z) {
        const X = Math.floor(x);
        const Y = Math.floor(y);
        const Z = Math.floor(z);
        const fx = x - X;
        const fy = y - Y;
        const fz = z - Z;
        
        // GPU-optimized: Inline hash function (no closure, direct computation)
        // Pre-compute hash constants for better performance
        const HASH_C1 = 12.9898;
        const HASH_C2 = 78.233;
        const HASH_C3 = 144.7272;
        const HASH_MUL = 43758.5453;
        
        // Get corner values (inline hash calls)
        const v000 = (Math.sin(X * HASH_C1 + Y * HASH_C2 + Z * HASH_C3) * HASH_MUL) % 1;
        const v100 = (Math.sin((X + 1) * HASH_C1 + Y * HASH_C2 + Z * HASH_C3) * HASH_MUL) % 1;
        const v010 = (Math.sin(X * HASH_C1 + (Y + 1) * HASH_C2 + Z * HASH_C3) * HASH_MUL) % 1;
        const v110 = (Math.sin((X + 1) * HASH_C1 + (Y + 1) * HASH_C2 + Z * HASH_C3) * HASH_MUL) % 1;
        const v001 = (Math.sin(X * HASH_C1 + Y * HASH_C2 + (Z + 1) * HASH_C3) * HASH_MUL) % 1;
        const v101 = (Math.sin((X + 1) * HASH_C1 + Y * HASH_C2 + (Z + 1) * HASH_C3) * HASH_MUL) % 1;
        const v011 = (Math.sin(X * HASH_C1 + (Y + 1) * HASH_C2 + (Z + 1) * HASH_C3) * HASH_MUL) % 1;
        const v111 = (Math.sin((X + 1) * HASH_C1 + (Y + 1) * HASH_C2 + (Z + 1) * HASH_C3) * HASH_MUL) % 1;
        
        // Smoothstep interpolation
        // Optimized: Inline smoothstep function to avoid function call overhead
        const sx = fx * fx * (3 - 2 * fx);
        const sy = fy * fy * (3 - 2 * fy);
        const sz = fz * fz * (3 - 2 * fz);
        
        // Trilinear interpolation
        const x00 = v000 * (1 - sx) + v100 * sx;
        const x10 = v010 * (1 - sx) + v110 * sx;
        const x01 = v001 * (1 - sx) + v101 * sx;
        const x11 = v011 * (1 - sx) + v111 * sx;
        
        const y0 = x00 * (1 - sy) + x10 * sy;
        const y1 = x01 * (1 - sy) + x11 * sy;
        
        return y0 * (1 - sz) + y1 * sz;
    }
    
    // Blend noise field into temperature (for fire-like turbulence)
    // GPU-accelerated version available
    blendNoiseIntoTemperature() {
        if (useGPU && gpuKernels.blendNoise) {
            // Use cached solidMap array
            const solidMapArray = getSolidMapArray();
            
            const result = gpuKernels.blendNoise(
                this.temperature,
                this.noise,
                solidMapArray,
                this.noiseBlending
            );
            
            // Copy results back
            const size = this.width * this.height;
            for (let i = 0; i < size; i++) {
                this.temperature[i] = result[i];
            }
        } else {
            // CPU fallback
            const size = this.width * this.height;
            const temperature = this.temperature;
            const noise = this.noise;
            const noiseBlending = this.noiseBlending;
            
            for (let i = 0; i < size; i++) {
                if (solidMap[i]) continue;
                
                const base = temperature[i];
                // Only blend noise if there's actual temperature (fire exists)
                if (base > 0.01) {
                    const n = noise[i];
                    const blended = base + noiseBlending * (n - base);
                    temperature[i] = Math.max(0, Math.min(1.0, blended));
                }
            }
        }
    }
    
    // Update density field from temperature (for visualization)
    // GPU-accelerated version available
    updateDensityFromTemperature() {
        if (useGPU && gpuKernels.updateDensity) {
            // Use cached solidMap array
            const solidMapArray = getSolidMapArray();
            
            const result = gpuKernels.updateDensity(
                this.density,
                this.temperature,
                solidMapArray
            );
            
            // Copy results back
            const size = this.width * this.height;
            for (let i = 0; i < size; i++) {
                this.density[i] = result[i];
            }
        } else {
            // CPU fallback
            const size = this.width * this.height;
            const density = this.density;
            const temperature = this.temperature;
            
            for (let i = 0; i < size; i++) {
                if (solidMap[i]) {
                    density[i] = 0;
                    continue;
                }
                
                // Density tracks temperature (with some persistence)
                density[i] = Math.max(density[i] * 0.95, temperature[i] * 0.8);
            }
        }
    }
    
    // Continuously feed flame sources with fuel and temperature (called every frame)
    feedFlameSources() {
        if (flameSources.length > 0) {
            for (let i = 0; i < flameSources.length; i++) {
                const source = flameSources[i];
                const x = source.x;
                const y = source.y;
                if (x >= 0 && x < WIDTH && y >= 0 && y < HEIGHT) {
                    const index = IX(x, y);
                    if (!solidMap[index]) {
                        // Continuously add fuel and temperature at source
                        // Use max to ensure it stays at BASE_HEAT level
                        this.temperature[index] = Math.max(this.temperature[index], BASE_HEAT);
                        this.fuel[index] = Math.max(this.fuel[index], 1.0);
                        this.density[index] = Math.max(this.density[index], BASE_HEAT * 0.8);
                    }
                }
            }
        } else {
            // Fallback: feed bottom row continuously
            for (let x = 0; x < WIDTH; x++) {
                const index = IX(x, 0);
                if (!solidMap[index]) {
                    this.temperature[index] = Math.max(this.temperature[index], BASE_HEAT);
                    this.fuel[index] = Math.max(this.fuel[index], 1.0);
                    this.density[index] = Math.max(this.density[index], BASE_HEAT * 0.8);
                }
            }
        }
    }
    
    // Linear solver (Gauss-Seidel)
    // Optimized: Pre-calculate loop bounds (called many times with PRESSURE_ITERATIONS)
    lin_solve(b, x, x0, a, cRecip, iter) {
        const widthMinus1 = this.width - 1;
        const heightMinus1 = this.height - 1;
        
        for (let k = 0; k < iter; k++) {
            for (let j = 1; j < widthMinus1; j++) {
                for (let i = 1; i < heightMinus1; i++) {
                    const index = IX(j, i);
                    
                    if (solidMap[index]) {
                        x[index] = 0;
                        continue;
                    }
                    
                    x[index] = (x0[index] + a * (
                        x[IX(j + 1, i)] +
                        x[IX(j - 1, i)] +
                        x[IX(j, i + 1)] +
                        x[IX(j, i - 1)]
                    )) * cRecip;
                }
            }
            this.set_bnd(b, x);
        }
    }
    
    // Diffuse a field
    // Optimized: Cache size calculation
    diffuse(b, x, x0, diff, iter) {
        const a = this.dt * diff * (this.width - 2) * (this.height - 2);
        const cRecip = 1.0 / (1 + 4 * a);
        
        if (a === 0 || cRecip === Infinity || isNaN(cRecip)) {
            // No diffusion, just copy
            const size = this.width * this.height;
            for (let i = 0; i < size; i++) {
                x[i] = x0[i];
            }
            return;
        }
        
        this.lin_solve(b, x, x0, a, cRecip, iter);
    }
    
    // Advect a field (semi-Lagrangian method) with dissipation
    // GPU-accelerated version available - THIS IS THE BIGGEST BOTTLENECK (called 5+ times per frame!)
    advect(b, d, d0, velocX, velocY, dissipation = 1.0) {
        if (useGPU && gpuKernels.advect) {
            // Use cached solidMap array
            const solidMapArray = getSolidMapArray();
            
            const result = gpuKernels.advect(
                d0,
                velocX,
                velocY,
                solidMapArray,
                this.dt,
                this.width,
                this.height,
                dissipation
            );
            
            // Copy results back
            const size = this.width * this.height;
            for (let i = 0; i < size; i++) {
                d[i] = result[i];
            }
            
            // Apply boundary conditions (still need CPU for this)
            this.set_bnd(b, d);
        } else {
            // CPU fallback
            const widthMinus2 = this.width - 2;
            const heightMinus2 = this.height - 2;
            const dtx = this.dt * widthMinus2;
            const dty = this.dt * heightMinus2;
            const NfloatW = widthMinus2;
            const NfloatH = heightMinus2;
            const width = this.width;
            const widthMinus1 = this.width - 1;
            const heightMinus1 = this.height - 1;
            
            // Pre-calculate row offsets for direct array access (eliminates IX() calls)
            // Optimized: Use Uint32Array for integer indices (faster than regular Array)
            const rowOffsets = new Uint32Array(this.height);
            for (let i = 0; i < this.height; i++) {
                rowOffsets[i] = i * width;
            }
            
            for (let j = 1; j < widthMinus1; j++) {
                for (let i = 1; i < heightMinus1; i++) {
                    const index = rowOffsets[i] + j;
                    
                    if (solidMap[index]) {
                        d[index] = 0;
                        continue;
                    }
                    
                    // Trace backwards
                    let x = j - dtx * velocX[index];
                    let y = i - dty * velocY[index];
                    
                    // Clamp to grid bounds
                    x = Math.max(0.5, Math.min(NfloatW + 0.5, x));
                    y = Math.max(0.5, Math.min(NfloatH + 0.5, y));
                    
                    // Bilinear interpolation
                    const j0 = Math.floor(x);
                    const j1 = j0 + 1;
                    const i0 = Math.floor(y);
                    const i1 = i0 + 1;
                    
                    const s1 = x - j0;
                    const s0 = 1.0 - s1;
                    const t1 = y - i0;
                    const t0 = 1.0 - t1;
                    
                    // Direct array access using pre-calculated row offsets
                    const idx00 = rowOffsets[i0] + j0;
                    const idx01 = rowOffsets[i1] + j0;
                    const idx10 = rowOffsets[i0] + j1;
                    const idx11 = rowOffsets[i1] + j1;
                    
                    d[index] = dissipation * (
                        s0 * (t0 * d0[idx00] + t1 * d0[idx01]) +
                        s1 * (t0 * d0[idx10] + t1 * d0[idx11])
                    );
                }
            }
            this.set_bnd(b, d);
        }
    }
    
    // Project velocity to make it incompressible
    // Optimized: Pre-calculate loop bounds and cache array references
    project(velocX, velocY, p, div) {
        const widthMinus1 = this.width - 1;
        const heightMinus1 = this.height - 1;
        
        // Calculate divergence
        for (let j = 1; j < widthMinus1; j++) {
            for (let i = 1; i < heightMinus1; i++) {
                const index = IX(j, i);
                
                if (solidMap[index]) {
                    div[index] = 0;
                    p[index] = 0;
                    continue;
                }
                
                div[index] = -0.5 * (
                    velocX[IX(j + 1, i)] - velocX[IX(j - 1, i)] +
                    velocY[IX(j, i + 1)] - velocY[IX(j, i - 1)]
                );
                p[index] = 0;
            }
        }
        this.set_bnd(0, div);
        this.set_bnd(0, p);
        
        // Solve for pressure with more iterations and dissipation
        this.lin_solve(0, p, div, 1, 1.0 / 4.0, PRESSURE_ITERATIONS);
        
        // Apply pressure dissipation
        const pLength = p.length;
        const pressureDissipation = PRESSURE_DISSIPATION;
        for (let i = 0; i < pLength; i++) {
            p[i] *= pressureDissipation;
        }
        
        // Subtract pressure gradient
        for (let j = 1; j < widthMinus1; j++) {
            for (let i = 1; i < heightMinus1; i++) {
                const index = IX(j, i);
                
                if (solidMap[index]) {
                    velocX[index] = 0;
                    velocY[index] = 0;
                    continue;
                }
                
                velocX[index] -= 0.5 * (p[IX(j + 1, i)] - p[IX(j - 1, i)]);
                velocY[index] -= 0.5 * (p[IX(j, i + 1)] - p[IX(j, i - 1)]);
            }
        }
        this.set_bnd(1, velocX);
        this.set_bnd(2, velocY);
    }
    
    // Set boundary conditions
    // Optimized: Pre-calculate loop bounds and cache values
    set_bnd(b, x) {
        const solidMapLength = solidMap.length;
        const widthMinus1 = this.width - 1;
        const heightMinus1 = this.height - 1;
        const heightMinus2 = this.height - 2;
        const widthMinus2 = this.width - 2;
        
        // Handle solid boundaries
        for (let i = 0; i < solidMapLength; i++) {
            if (solidMap[i]) {
                x[i] = 0;
            }
        }
        
        // Handle regular boundaries
        for (let j = 1; j < widthMinus1; j++) {
            const idx0 = IX(j, 0);
            if (!solidMap[idx0]) {
                x[idx0] = b === 2 ? -x[IX(j, 1)] : x[IX(j, 1)];
            }
            const idxTop = IX(j, heightMinus1);
            if (!solidMap[idxTop]) {
                x[idxTop] = b === 2 ? -x[IX(j, heightMinus2)] : x[IX(j, heightMinus2)];
            }
        }
        for (let i = 1; i < heightMinus1; i++) {
            const idxLeft = IX(0, i);
            if (!solidMap[idxLeft]) {
                x[idxLeft] = b === 1 ? -x[IX(1, i)] : x[IX(1, i)];
            }
            const idxRight = IX(widthMinus1, i);
            if (!solidMap[idxRight]) {
                x[idxRight] = b === 1 ? -x[IX(widthMinus2, i)] : x[IX(widthMinus2, i)];
            }
        }
        
        // Handle corners
        const corner00 = IX(0, 0);
        if (!solidMap[corner00]) {
            x[corner00] = 0.5 * (x[IX(1, 0)] + x[IX(0, 1)]);
        }
        const corner0H = IX(0, heightMinus1);
        if (!solidMap[corner0H]) {
            x[corner0H] = 0.5 * (x[IX(1, heightMinus1)] + x[IX(0, heightMinus2)]);
        }
        const cornerW0 = IX(widthMinus1, 0);
        if (!solidMap[cornerW0]) {
            x[cornerW0] = 0.5 * (x[IX(widthMinus2, 0)] + x[IX(widthMinus1, 1)]);
        }
        const cornerWH = IX(widthMinus1, heightMinus1);
        if (!solidMap[cornerWH]) {
            x[cornerWH] = 0.5 * (x[IX(widthMinus2, heightMinus1)] + x[IX(widthMinus1, heightMinus2)]);
        }
    }
    
    // Get temperature at position (for rendering)
    getTemperature(x, y) {
        return this.temperature[IX(x, y)];
    }
    
    // Get density at position (for rendering)
    getDensity(x, y) {
        return this.density[IX(x, y)];
    }
}

// Store flame source positions
let flameSources = [];
let solidMap = [];

// Initialize fire simulation
const fire = new FireSimulation();

// ASCII character mapping for fire intensity
const fireChars = " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$";
const FIRE_CHARS_LEN = fireChars.length;
const FIRE_CHARS_LEN_MINUS_1 = FIRE_CHARS_LEN - 1;

// Color mapping for fire based on temperature
function getFireColor(temperature) {
    // Temperature is normalized 0-1
    if (temperature < 0.1) {
        return 'rgba(139, 0, 0, 0.6)';
    } else if (temperature < 0.25) {
        return 'rgba(139, 0, 0, 1)';
    } else if (temperature < 0.4) {
        return 'rgba(255, 0, 0, 1)';
    } else if (temperature < 0.55) {
        return 'rgba(255, 69, 0, 1)';
    } else if (temperature < 0.7) {
        return 'rgba(255, 140, 0, 1)';
    } else if (temperature < 0.85) {
        return 'rgba(255, 215, 0, 1)';
    } else {
        return 'rgba(255, 255, 200, 1)';
    }
}

function getFireChar(temperature) {
    if (temperature < 0.05) return ' ';
    const charIndex = Math.floor(temperature * FIRE_CHARS_LEN_MINUS_1);
    return fireChars[charIndex < FIRE_CHARS_LEN ? charIndex : FIRE_CHARS_LEN_MINUS_1];
}

// Get the pre element for rendering
let artElement = null;
let fireOverlay = null;

function getArtElement() {
    if (!artElement) {
        artElement = document.getElementById('artpre');
    }
    return artElement;
}

// Create fire overlay
function createFireOverlay() {
    if (fireOverlay) return;
    
    const element = getArtElement();
    if (!element) return;
    
    if (element.parentElement) {
        element.parentElement.style.position = 'relative';
    }
    
    fireOverlay = document.createElement('pre');
    fireOverlay.id = 'fireOverlay';
    
    const originalStyles = window.getComputedStyle(element);
    fireOverlay.style.position = 'absolute';
    fireOverlay.style.top = '0';
    fireOverlay.style.left = '0';
    fireOverlay.style.width = '100%';
    fireOverlay.style.height = '100%';
    fireOverlay.style.background = 'transparent';
    fireOverlay.style.fontFamily = originalStyles.fontFamily;
    fireOverlay.style.margin = '0';
    fireOverlay.style.padding = originalStyles.padding;
    fireOverlay.style.pointerEvents = 'none';
    fireOverlay.style.whiteSpace = 'pre';
    fireOverlay.style.overflow = 'hidden';
    fireOverlay.style.zIndex = '10';
    
    if (element.parentElement) {
        element.parentElement.appendChild(fireOverlay);
    }
    
    updateOverlayPosition();
}

function updateOverlayPosition() {
    if (!fireOverlay) return;
    
    const element = getArtElement();
    if (!element) return;
    
    const originalStyles = window.getComputedStyle(element);
    const originalFontSize = parseFloat(originalStyles.fontSize) || 15;
    fireOverlay.style.fontSize = (originalFontSize / RESOLUTION_SCALE) + 'px';
    fireOverlay.style.lineHeight = (originalFontSize / RESOLUTION_SCALE) + 'px';
}

// Store original pre element content
let originalPreContent = '';
let originalArtLines = [];
let originalFlameSources = [];
let SOURCE_CHAR = 'o';

// Parse flame sources
function     parseFlameSources() {
        const element = getArtElement();
        if (!element) return;
        
        let text = originalPreContent || element.textContent || element.innerText || '';
        if (!originalPreContent && text) {
            originalPreContent = text;
        }
        
        const lines = text.split('\n');
        originalArtLines = lines;
        originalFlameSources = [];
        flameSources = [];
        
        if (solidMap.length !== WIDTH * HEIGHT) {
            solidMap = new Array(WIDTH * HEIGHT).fill(false);
        } else {
            solidMap.fill(false);
        }
        
        // Invalidate cached solidMap array
        cachedSolidMapArray = null;
    
    const sourceCharLower = SOURCE_CHAR.toLowerCase();
    const sourceCharUpper = SOURCE_CHAR.toUpperCase();
    
    for (let y = 0; y < lines.length; y++) {
        const line = lines[y];
        for (let x = 0; x < line.length; x++) {
            const char = line[x];
            
            if (char === sourceCharLower || char === sourceCharUpper) {
                originalFlameSources.push({ charX: x, charY: y });
            }
        }
    }
    
    const artElement = getArtElement();
    if (artElement) {
        const FONT_SCALE = 1.0 / RESOLUTION_SCALE;
        const artHeight = lines.length;
        
        for (const source of originalFlameSources) {
            const gridX = Math.floor((source.charX + 1) / FONT_SCALE);
            const gridY = Math.floor((artHeight - 1 - source.charY) / FONT_SCALE);
            
            if (gridX >= 0 && gridX < WIDTH && gridY >= 0 && gridY < HEIGHT) {
                flameSources.push({ x: gridX, y: gridY });
            }
        }
        
        for (let y = 0; y < lines.length; y++) {
            const line = lines[y];
            for (let x = 0; x < line.length; x++) {
                const char = line[x];
                
                const sourceCharLower = SOURCE_CHAR.toLowerCase();
                const sourceCharUpper = SOURCE_CHAR.toUpperCase();
                
                if (char && char !== ' ' && char !== '\t' && char !== '\n' && char !== '\r' && char !== sourceCharLower && char !== sourceCharUpper) {
                    const gridX = Math.floor(x / FONT_SCALE);
                    const gridY = Math.floor((artHeight - 1 - y) / FONT_SCALE);
                    
                    if (gridX >= 0 && gridX < WIDTH && gridY >= 0 && gridY < HEIGHT) {
                        const index = IX(gridX, gridY);
                        solidMap[index] = true;
                    }
                }
            }
        }
    }
    
    console.log(`Found ${flameSources.length} flame source(s) and ${solidMap.filter(s => s).length} wall cell(s)`);
}

// Pause state
let isPaused = false;

// Animation loop
function animate(currentTime) {
    requestAnimationFrame(animate);
    
    if (isPaused) return;
    
    if (!currentTime) {
        currentTime = performance.now();
    }
    
    if (lastFrameTime === 0) {
        lastFrameTime = currentTime;
        lastFpsUpdate = currentTime;
    }
    
    const deltaTime = currentTime - lastFrameTime;
    const targetFrameTime = 1000 / TARGET_FPS;
    
    if (deltaTime >= targetFrameTime) {
        lastFrameTime = currentTime - (deltaTime % targetFrameTime);
        
        fire.step();
        
        // Render using temperature field
        // MASSIVELY OPTIMIZED: Build single string directly (fastest approach)
        const threshold = 0.05;
        const temperature = fire.temperature;
        
        if (fireOverlay) {
            // Build entire string in one pass - much faster than array joins
            let html = '';
            
            for (let y = HEIGHT - 1; y >= 0; y--) {
                const yOffset = y * WIDTH;
                
                for (let x = 0; x < WIDTH; x++) {
                    const index = x + yOffset;
                    const isSolid = solidMap[index];
                    
                    if (isSolid) {
                        html += ' ';
                    } else {
                        const temp = temperature[index];
                        if (temp > threshold) {
                            const char = getFireChar(temp);
                            const color = getFireColor(temp);
                            html += `<span style="color:${color}">${char}</span>`;
                        } else {
                            html += ' ';
                        }
                    }
                }
                html += '\n';
            }
            
            // Single DOM update
            fireOverlay.innerHTML = html;
        }
        
        renderedFrameCount++;
    }
    
    if (currentTime - lastFpsUpdate >= 1000) {
        fps = renderedFrameCount;
        renderedFrameCount = 0;
        lastFpsUpdate = currentTime;
        
        const fpsDisplay = document.getElementById('fpsDisplay');
        if (fpsDisplay) {
            fpsDisplay.textContent = fps;
        }
    }
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}

// Setup slider controls
function setupControls() {
    // Helper function to reset animation when any parameter changes
    function resetAnimation() {
        fire.updateParameters();
        fire.initialize();
    }
    
    // Burn Temperature
    const burnTempSlider = document.getElementById('maxFire');
    const burnTempValue = document.getElementById('maxFireValue');
    if (burnTempSlider) {
        burnTempSlider.addEventListener('input', (e) => {
            BURN_TEMP = parseFloat(e.target.value);
            if (burnTempValue) burnTempValue.textContent = BURN_TEMP.toFixed(2);
            resetAnimation();
        });
    }
    
    // Cooling Rate
    const coolingSlider = document.getElementById('decay');
    const coolingValue = document.getElementById('decayValue');
    if (coolingSlider) {
        coolingSlider.addEventListener('input', (e) => {
            COOLING_RATE = parseFloat(e.target.value);
            if (coolingValue) coolingValue.textContent = COOLING_RATE.toFixed(4);
            resetAnimation();
        });
    }
    
    // Buoyancy
    const buoyancySlider = document.getElementById('spread');
    const buoyancyValue = document.getElementById('spreadValue');
    if (buoyancySlider) {
        buoyancySlider.addEventListener('input', (e) => {
            BUOYANCY = parseFloat(e.target.value);
            if (buoyancyValue) buoyancyValue.textContent = BUOYANCY.toFixed(3);
            resetAnimation();
        });
    }
    
    // Base Heat
    const baseHeatSlider = document.getElementById('baseHeat');
    const baseHeatValue = document.getElementById('baseHeatValue');
    if (baseHeatSlider) {
        baseHeatSlider.addEventListener('input', (e) => {
            BASE_HEAT = parseFloat(e.target.value);
            if (baseHeatValue) baseHeatValue.textContent = BASE_HEAT.toFixed(2);
            resetAnimation();
        });
    }
    
    // Vorticity Confinement
    const confinementSlider = document.getElementById('confinement');
    const confinementValue = document.getElementById('confinementValue');
    if (confinementSlider) {
        confinementSlider.addEventListener('input', (e) => {
            CONFINEMENT = parseFloat(e.target.value);
            if (confinementValue) confinementValue.textContent = CONFINEMENT.toFixed(1);
            resetAnimation();
        });
    }
    
    // Fuel Dissipation
    const fuelDissipationSlider = document.getElementById('fuelDissipation');
    const fuelDissipationValue = document.getElementById('fuelDissipationValue');
    if (fuelDissipationSlider) {
        fuelDissipationSlider.addEventListener('input', (e) => {
            FUEL_DISSIPATION = parseFloat(e.target.value);
            if (fuelDissipationValue) fuelDissipationValue.textContent = FUEL_DISSIPATION.toFixed(3);
            resetAnimation();
        });
    }
    
    // Density Dissipation
    const densityDissipationSlider = document.getElementById('densityDissipation');
    const densityDissipationValue = document.getElementById('densityDissipationValue');
    if (densityDissipationSlider) {
        densityDissipationSlider.addEventListener('input', (e) => {
            DENSITY_DISSIPATION = parseFloat(e.target.value);
            if (densityDissipationValue) densityDissipationValue.textContent = DENSITY_DISSIPATION.toFixed(3);
            resetAnimation();
        });
    }
    
    // Velocity Dissipation
    const velocityDissipationSlider = document.getElementById('velocityDissipation');
    const velocityDissipationValue = document.getElementById('velocityDissipationValue');
    if (velocityDissipationSlider) {
        velocityDissipationSlider.addEventListener('input', (e) => {
            VELOCITY_DISSIPATION = parseFloat(e.target.value);
            if (velocityDissipationValue) velocityDissipationValue.textContent = VELOCITY_DISSIPATION.toFixed(3);
            resetAnimation();
        });
    }
    
    // Noise Blending
    const noiseBlendingSlider = document.getElementById('noiseBlending');
    const noiseBlendingValue = document.getElementById('noiseBlendingValue');
    if (noiseBlendingSlider) {
        noiseBlendingSlider.addEventListener('input', (e) => {
            NOISE_BLENDING = parseFloat(e.target.value);
            if (noiseBlendingValue) noiseBlendingValue.textContent = NOISE_BLENDING.toFixed(2);
            resetAnimation();
        });
    }
    
    // Noise Volatility
    const noiseVolatilitySlider = document.getElementById('noiseVolatility');
    const noiseVolatilityValue = document.getElementById('noiseVolatilityValue');
    if (noiseVolatilitySlider) {
        noiseVolatilitySlider.addEventListener('input', (e) => {
            NOISE_VOLATILITY = parseFloat(e.target.value);
            if (noiseVolatilityValue) noiseVolatilityValue.textContent = NOISE_VOLATILITY.toFixed(3);
            resetAnimation();
        });
    }
    
    // Target FPS (doesn't need to reset animation, just timing)
    const targetFPSSlider = document.getElementById('targetFPS');
    const targetFPSValue = document.getElementById('targetFPSValue');
    if (targetFPSSlider) {
        targetFPSSlider.addEventListener('input', (e) => {
            TARGET_FPS = parseInt(e.target.value);
            if (targetFPSValue) targetFPSValue.textContent = TARGET_FPS;
            lastFrameTime = 0;
        });
    }
    
    const resolutionSlider = document.getElementById('resolution');
    const resolutionValue = document.getElementById('resolutionValue');
    
    if (resolutionSlider) {
        resolutionSlider.addEventListener('input', (e) => {
            RESOLUTION_SCALE = parseFloat(e.target.value);
            if (resolutionValue) resolutionValue.textContent = RESOLUTION_SCALE.toFixed(1);
            fire.resize();
            parseFlameSources();
            updateOverlayPosition();
            fire.initialize();
        });
    }
    
    const pauseButton = document.getElementById('pauseButton');
    if (pauseButton) {
        pauseButton.addEventListener('click', () => {
            isPaused = !isPaused;
            pauseButton.textContent = isPaused ? 'Resume' : 'Pause';
        });
    }
    
    const sourceCharInput = document.getElementById('sourceChar');
    if (sourceCharInput) {
        SOURCE_CHAR = sourceCharInput.value[0];
        sourceCharInput.addEventListener('input', (e) => {
            const newChar = e.target.value.trim();
            if (newChar.length > 0) {
                SOURCE_CHAR = newChar[0];
                e.target.value = SOURCE_CHAR;
                parseFlameSources();
                fire.initialize();
            }
        });
    }
}

// Handle window resize
let resizeTimeout;
function handleResize() {
    clearTimeout(resizeTimeout);
    resizeTimeout = setTimeout(() => {
        fire.resize();
        parseFlameSources();
        updateOverlayPosition();
        fire.initialize();
    }, 100);
}

window.addEventListener('resize', handleResize);

function init() {
    // Initialize GPU.js first (if available)
    initGPU();
    
    fire.resize();
    parseFlameSources();
    createFireOverlay();
    setupControls();
    fire.initialize();
    animate();
}
