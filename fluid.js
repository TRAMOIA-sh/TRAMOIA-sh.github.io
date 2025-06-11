// fluid simulation in ascii
// using Jos Stam's "Real-Time Fluid Dynamics for Games"
// https://www.dgp.toronto.edu/people/stam/reality/Research/pdf/GDC03.pdfd
// https://mikeash.com/pyblog/fluid-simulation-for-dummies.html

// Calculate fluid grid size based on actual art dimensions
const tempArtElement = document.getElementById('artpre');
const tempArtText = tempArtElement.innerText;
const tempArtLines = tempArtText.split('\n');
const WIDTH = Math.max(...tempArtLines.map(line => line.length));
// const WIDTH = window.innerWidth;
const HEIGHT = tempArtLines.length;

// Calculate scaling factor for higher resolution
const FONT_SCALE = 0.5; // This will make the font size effectively 8px (assuming original is 16px)
const SCALED_WIDTH = Math.floor(WIDTH / FONT_SCALE);
const SCALED_HEIGHT = Math.floor(HEIGHT / FONT_SCALE);

console.log('Fluid grid dimensions (based on art):', SCALED_WIDTH, 'x', SCALED_HEIGHT, '=', SCALED_WIDTH * SCALED_HEIGHT, 'cells');

// --- Fluid Simulation Code ---

// Simulation parameters
// const N = 128;      // Grid size (N x N) - Power of 2 preferable, keep reasonable
const SCALE = 1;   // How large the visualization plane is in the scene
const iter = 4;    // Increased iterations for more accurate solving
const dt = 0.01;   // Smaller time step for more stability
const diffusion = 0.00000001; // Further reduced diffusion for more cohesive behavior
const viscosity = 0.00000010; // Reduced viscosity for less aggressive movement

// Helper function for 1D array indexing from 2D coordinates
function IX(x, y) {
    // Clamp coordinates to grid bounds
    x = Math.max(0, Math.min(SCALED_WIDTH - 1, Math.floor(x)));
    y = Math.max(0, Math.min(SCALED_HEIGHT - 1, Math.floor(y)));
    return x + y * SCALED_WIDTH; // Use SCALED_WIDTH for row stride
}

// Fluid Cube class to hold the state and simulation steps
class FluidCube {
    constructor(dt, diffusion, viscosity) {
        this.width = SCALED_WIDTH;
        this.height = SCALED_HEIGHT;
        this.dt = dt;
        this.diff = diffusion;
        this.visc = viscosity;

        // Simulation data arrays (using Float32Array for performance)
        const size = SCALED_WIDTH * SCALED_HEIGHT;
        this.s = new Float32Array(size).fill(0);        // Previous density
        this.density = new Float32Array(size).fill(0); // Current density

        this.Vx = new Float32Array(size).fill(0);       // Current X velocity
        this.Vy = new Float32Array(size).fill(0);       // Current Y velocity

        this.Vx0 = new Float32Array(size).fill(0);      // Previous X velocity
        this.Vy0 = new Float32Array(size).fill(0);      // Previous Y velocity
    }

    addDensity(x, y, amount) {
        this.density[IX(x, y)] += amount;
    }

    addVelocity(x, y, amountX, amountY) {
        let index = IX(x, y);
        this.Vx[index] += amountX;
        this.Vy[index] += amountY;
    }

    // --- Core Simulation Steps ---
    /**
     * Advances the fluid simulation by one time step.
     */
    step() {
        const N_actual = this.width;
        const visc = this.visc;
        const diff = this.diff;
        const dt = this.dt;
        const Vx = this.Vx;
        const Vy = this.Vy;
        const Vx0 = this.Vx0;
        const Vy0 = this.Vy0;
        const s = this.s; // Previous density
        const density = this.density;

        // Add constant gravity force to all cells
        // const gravity = -0.01; // Constant downward force
        // for (let i = 0; i < this.width * this.height; i++) {
        //     Vy[i] += gravity * dt; // Apply gravity to vertical velocity
        // }

        // 1. Diffuse Velocity
        this.diffuse(1, Vx0, Vx, visc, iter); // Diffuse Vx into Vx0
        this.diffuse(2, Vy0, Vy, visc, iter); // Diffuse Vy into Vy0

        // 2. Project Velocity (make incompressible)
        // Vx0, Vy0 are now the diffused velocities
        // p and div fields are reused (passed as Vx and Vy temporarily for project)
        this.project(Vx0, Vy0, Vx, Vy); // Project Vx0, Vy0; results back in Vx0, Vy0; uses Vx, Vy for temp storage

        // 3. Advect Velocity
        // Advect the projected velocity (Vx0, Vy0) using itself
        // Result goes into Vx, Vy
        this.advect(1, Vx, Vx0, Vx0, Vy0); // Advect Vx0 based on Vx0, Vy0 -> store in Vx
        this.advect(2, Vy, Vy0, Vx0, Vy0); // Advect Vy0 based on Vx0, Vy0 -> store in Vy

        // 4. Project Velocity again (correcting advection)
        // Vx, Vy are now the advected velocities
        // p and div fields reused again (passed as Vx0, Vy0 temporarily)
        this.project(Vx, Vy, Vx0, Vy0); // Project Vx, Vy; results back in Vx, Vy; uses Vx0, Vy0 for temp storage

        // --- Density Step ---

        // 5. Diffuse Density
        this.diffuse(0, s, density, diff, iter); // Diffuse density into s

        // 6. Advect Density
        // Advect the diffused density (s) using the final velocity field (Vx, Vy)
        // Result goes back into density array
        this.advect(0, density, s, Vx, Vy); // Advect s based on Vx, Vy -> store in density
    }

     /**
      * Solves a linear system using Gauss-Seidel relaxation.
      * Modified to respect solid boundaries and only process active cells.
      */
     lin_solve(b, x, x0, a, cRecip, iter) {
        for (let k = 0; k < iter; k++) {
            for (let j = 1; j < this.width - 1; j++) { // x-coord
                for (let i = 1; i < this.height - 1; i++) { // y-coord
                    const index = IX(j, i);
                    
                    // Skip inactive cells for performance
                    if (!activeMap[index]) continue;
                    
                    // Skip solid cells
                    if (solidMap[index]) {
                        x[index] = 0;
                        continue;
                    }
                    
                    x[index] =
                        (x0[index] +
                            a * (x[IX(j + 1, i)] +
                                 x[IX(j - 1, i)] +
                                 x[IX(j, i + 1)] +
                                 x[IX(j, i - 1)]
                                )) * cRecip;
                }
            }
            this.set_bnd(b, x);
        }
    }

    /**
     * Simulates the diffusion of a quantity (density or velocity).
     * @param {number} b - Boundary condition type.
     * @param {Float32Array} x - Current state array (updated in place).
     * @param {Float32Array} x0 - Previous state array.
     * @param {number} diff - Diffusion rate.
     * @param {number} iter - Number of solver iterations.
     */
     diffuse(b, x, x0, diff, iter) {
        // const N_actual = this.width;
        // Use width/height in factor calculation
        const a = this.dt * diff * (this.width - 2) * (this.height - 2);
        // Avoid division by zero or very small numbers if a is close to 0
        const cRecip = 1.0 / (1 + 4 * a); // Assuming dx=dy=1, Gauss-Seidel uses 4 neighbors
        if (a === 0 || cRecip === Infinity || isNaN(cRecip)) {
            // If no diffusion, simply copy x0 to x for the internal cells
            for (let j = 1; j < this.width - 1; j++) {
                for (let i = 1; i < this.height - 1; i++) {
                    x[IX(j, i)] = x0[IX(j, i)];
                }
            }
            this.set_bnd(b, x);
        } else {
            this.lin_solve(b, x, x0, a, cRecip, iter);
        }
    }

    /**
     * Simulates the advection (transport) of a quantity through the velocity field.
     * Modified to respect solid boundaries and only process active cells.
     */
     advect(b, d, d0, velocX, velocY) {
        let i0, i1, j0, j1;

        const dtx = this.dt * (this.width - 2); // Scale dt by grid size for stable advection
        const dty = this.dt * (this.height - 2);

        let s0, s1, t0, t1;
        let tmp1, tmp2, x, y;

        const NfloatW = this.width;
        const NfloatH = this.height;
        for (let j = 1; j < this.width - 1; j++) { // x-coord
            for (let i = 1; i < this.height - 1; i++) { // y-coord
                const index = IX(j, i);
                
                // Skip inactive cells for performance
                if (!activeMap[index]) {
                    d[index] = d0[index] * 0.99; // Slight decay for inactive cells
                    continue;
                }
                
                // Skip solid cells completely
                if (solidMap[index]) {
                    d[index] = 0;
                    continue;
                }
                
                // Calculate back-traced position
                tmp1 = dtx * velocX[index];
                tmp2 = dty * velocY[index];
                x = j - tmp1;
                y = i - tmp2;

                // Clamp coordinates to grid bounds (0.5 to N-1.5 representing cell centers)
                x = Math.max(0.5, Math.min(NfloatW - 1.5, x));
                y = Math.max(0.5, Math.min(NfloatH - 1.5, y));

                // Get integer and fractional parts for interpolation
                j0 = Math.floor(x);
                j1 = j0 + 1;
                i0 = Math.floor(y);
                i1 = i0 + 1;

                // Check if we're trying to interpolate from solid areas
                const idx00 = IX(j0, i0);
                const idx01 = IX(j0, i1);
                const idx10 = IX(j1, i0);
                const idx11 = IX(j1, i1);
                
                // If any interpolation source is solid, use reflection instead of decay
                if (solidMap[idx00] || solidMap[idx01] || solidMap[idx10] || solidMap[idx11]) {
                    // Calculate reflection: find which direction we're coming from
                    const fromX = j + tmp1; // Where we came from
                    const fromY = i + tmp2;
                    
                    // Reflect the traced position away from solid
                    let reflectedX = x;
                    let reflectedY = y;
                    
                    if (solidMap[idx00] || solidMap[idx10]) { // Left side hit
                        reflectedX = j + (j - x); // Reflect horizontally
                    }
                    if (solidMap[idx00] || solidMap[idx01]) { // Bottom hit
                        reflectedY = i + (i - y); // Reflect vertically
                    }
                    
                    // Clamp reflected position
                    reflectedX = Math.max(0.5, Math.min(NfloatW - 1.5, reflectedX));
                    reflectedY = Math.max(0.5, Math.min(NfloatH - 1.5, reflectedY));
                    
                    // Get new interpolation indices for reflected position
                    j0 = Math.floor(reflectedX);
                    j1 = j0 + 1;
                    i0 = Math.floor(reflectedY);
                    i1 = i0 + 1;
                    
                    const ridx00 = IX(j0, i0);
                    const ridx01 = IX(j0, i1);
                    const ridx10 = IX(j1, i0);
                    const ridx11 = IX(j1, i1);
                    
                    // If reflected position is also solid, just use current value with slight decay
                    if (solidMap[ridx00] || solidMap[ridx01] || solidMap[ridx10] || solidMap[ridx11]) {
                        d[index] = d0[index] * 0.95;
                        continue;
                    }
                    
                    // Use reflected position for interpolation
                    s1 = reflectedX - j0;
                    s0 = 1.0 - s1;
                    t1 = reflectedY - i0;
                    t0 = 1.0 - t1;
                    
                    d[index] = 
                        s0 * (t0 * d0[ridx00] + t1 * d0[ridx01]) +
                        s1 * (t0 * d0[ridx10] + t1 * d0[ridx11]);
                    continue;
                }

                s1 = x - j0;
                s0 = 1.0 - s1;
                t1 = y - i0;
                t0 = 1.0 - t1;

                // Normal bilinear interpolation (only when no solids are involved)
                d[index] =
                    s0 * (t0 * d0[idx00] + t1 * d0[idx01]) +
                    s1 * (t0 * d0[idx10] + t1 * d0[idx11]);
            }
        }
        this.set_bnd(b, d);
    }

     /**
      * Enforces mass conservation by making the velocity field divergence-free.
      * Modified to respect solid boundaries and only process active cells.
      */
     project(velocX, velocY, p, div) {
        // Calculate divergence (using central differences, h=1)
        for (let j = 1; j < this.width - 1; j++) {
            for (let i = 1; i < this.height - 1; i++) {
                const index = IX(j, i);
                
                // Skip inactive cells for performance
                if (!activeMap[index]) {
                    div[index] = 0;
                    p[index] = 0;
                    continue;
                }
                
                // Skip solid cells
                if (solidMap[index]) {
                    div[index] = 0;
                    p[index] = 0;
                    continue;
                }
                
                div[index] = -0.5 * (velocX[IX(j + 1, i)] - velocX[IX(j - 1, i)] +
                                       velocY[IX(j, i + 1)] - velocY[IX(j, i - 1)]);
                p[index] = 0; // Initialize pressure
            }
        }
        this.set_bnd(0, div);
        this.set_bnd(0, p);

        // Solve for pressure using lin_solve (Poisson equation)
        this.lin_solve(0, p, div, 1, 1.0 / 4.0, iter);

        // Subtract pressure gradient from velocity field (using central differences, h=1)
        for (let j = 1; j < this.width - 1; j++) {
            for (let i = 1; i < this.height - 1; i++) {
                const index = IX(j, i);
                
                // Skip inactive cells for performance
                if (!activeMap[index]) continue;
                
                // Skip solid cells
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

    /**
     * Sets boundary conditions for a given field.
     * Modified to respect solid boundaries.
     */
     set_bnd(b, x) {
        // First handle solid constraints
        for (let i = 0; i < solidMap.length; i++) {
            if (solidMap[i]) {
                x[i] = 0;
            }
        }
        
        // Then handle regular boundaries
        for (let j = 1; j < this.width - 1; j++) { 
            if (!solidMap[IX(j, 0)]) {
                x[IX(j, 0)] = b === 2 ? -x[IX(j, 1)] : x[IX(j, 1)];
            }
            if (!solidMap[IX(j, this.height - 1)]) {
                x[IX(j, this.height - 1)] = b === 2 ? -x[IX(j, this.height - 2)] : x[IX(j, this.height - 2)];
            }
        }
        for (let i = 1; i < this.height - 1; i++) {
            if (!solidMap[IX(0, i)]) {
                x[IX(0, i)] = b === 1 ? -x[IX(1, i)] : x[IX(1, i)];
            }
            if (!solidMap[IX(this.width - 1, i)]) {
                x[IX(this.width - 1, i)] = b === 1 ? -x[IX(this.width - 2, i)] : x[IX(this.width - 2, i)];
            }
        }

        // Handle corners (only if they're not solid)
        if (!solidMap[IX(0, 0)]) {
            x[IX(0, 0)] = 0.5 * (x[IX(1, 0)] + x[IX(0, 1)]);
        }
        if (!solidMap[IX(0, this.height - 1)]) {
            x[IX(0, this.height - 1)] = 2.5 * (x[IX(1, this.height - 1)] + x[IX(0, this.height - 2)]);
        }
        if (!solidMap[IX(this.width - 1, 0)]) {
            x[IX(this.width - 1, 0)] = 0.5 * (x[IX(this.width - 4, 0)] + x[IX(this.width - 1, 1)]);
        }
        if (!solidMap[IX(this.width - 1, this.height - 1)]) {
            x[IX(this.width - 1, this.height - 1)] = 0.5 * (x[IX(this.width - 4, this.height - 1)] + x[IX(this.width - 1, this.height - 2)]);
        }
     }
}


const ansi_art = document.getElementById('artpre').innerText;

const fluid = new FluidCube(dt, diffusion, viscosity);

// Create a solid map to track which cells are obstacles
const solidMap = new Array(SCALED_WIDTH * SCALED_HEIGHT).fill(false);

// Add active cell tracking for optimization
const activeMap = new Array(SCALED_WIDTH * SCALED_HEIGHT).fill(false);
const ACTIVITY_THRESHOLD = 0.01; // Minimum density/velocity to consider a cell "active"

// Function to update active cells based on current fluid state
function updateActiveCells() {
    // Reset active map
    activeMap.fill(false);
    
    for (let i = 0; i < SCALED_WIDTH * SCALED_HEIGHT; i++) {
        // A cell is active if it has significant density, velocity, or is near solid boundaries
        const hasSignificantDensity = fluid.density[i] > ACTIVITY_THRESHOLD;
        const hasSignificantVelocity = Math.abs(fluid.Vx[i]) > ACTIVITY_THRESHOLD || Math.abs(fluid.Vy[i]) > ACTIVITY_THRESHOLD;
        
        if (hasSignificantDensity || hasSignificantVelocity || solidMap[i]) {
            activeMap[i] = true;
            
            // Also mark neighboring cells as active (fluid can spread)
            const x = i % SCALED_WIDTH;
            const y = Math.floor(i / SCALED_WIDTH);
            
            for (let dx = -1; dx <= 1; dx++) {
                for (let dy = -1; dy <= 1; dy++) {
                    const nx = x + dx;
                    const ny = y + dy;
                    if (nx >= 0 && nx < SCALED_WIDTH && ny >= 0 && ny < SCALED_HEIGHT) {
                        const neighborIndex = IX(nx, ny);
                        activeMap[neighborIndex] = true;
                    }
                }
            }
        }
    }
}

// Parse original art to create solid map
function parseArtForSolids() {
    const solids = [];
    const dropPoints = []; // Array to store drop point coordinates
    
    // Direct 1:1 mapping between art and fluid grid
    for (let y = 0; y < HEIGHT && y < tempArtLines.length; y++) {
        for (let x = 0; x < WIDTH && x < tempArtLines[y].length; x++) {
            const char = tempArtLines[y][x];
            
            // Only treat specific visible characters as solid
            // Exclude all whitespace (spaces, tabs, newlines, etc.)
            if (char && char !== ' ' && char !== '\t' && char !== '\n' && char !== '\r') {
                // Scale coordinates to match the higher resolution grid
                const scaledX = Math.floor(x / FONT_SCALE);
                const scaledY = Math.floor((HEIGHT - 1 - y) / FONT_SCALE);
                
                if (char === '°') {
                    // Store drop point coordinates
                    dropPoints.push({x: scaledX, y: scaledY});
                } else {
                    // Only treat actual printable characters as solid
                    solids.push({x: scaledX, y: scaledY});
                    // Mark this position as solid in the map
                    solidMap[IX(scaledX, scaledY)] = true;
                }
            }
        }
    }
    
    return { solids, dropPoints };
}

// Function to enforce solid constraints with proper bounce-back
function enforceSolidConstraints() {
    // Apply bounce-back boundary conditions only to active cells
    for (let j = 1; j < SCALED_WIDTH - 1; j++) {
        for (let i = 1; i < SCALED_HEIGHT - 1; i++) {
            const index = IX(j, i);
            
            // Skip inactive cells for performance
            if (!activeMap[index]) continue;
            
            // Skip if this cell is solid
            if (solidMap[index]) {
                fluid.Vx[index] = 0;
                fluid.Vy[index] = 0;
                fluid.Vx0[index] = 0;
                fluid.Vy0[index] = 0;
                fluid.density[index] = 0;
                fluid.s[index] = 0;
                continue;
            }
            
            // Check for solid neighbors and apply bounce-back
            const leftSolid = solidMap[IX(j - 1, i)];
            const rightSolid = solidMap[IX(j + 1, i)];
            const bottomSolid = solidMap[IX(j, i - 1)];
            const topSolid = solidMap[IX(j, i + 1)];
            
            // Bounce-back horizontal velocity if hitting vertical walls
            if (leftSolid && fluid.Vx[index] < 0) {
                fluid.Vx[index] = -fluid.Vx[index] * 0.8; // Some energy loss on bounce
                fluid.density[index] += Math.abs(fluid.Vx[index]) * 0.3; // Convert some velocity to density
            }
            if (rightSolid && fluid.Vx[index] > 0) {
                fluid.Vx[index] = -fluid.Vx[index] * 0.8;
                fluid.density[index] += Math.abs(fluid.Vx[index]) * 0.3; // Convert some velocity to density
            }
            
            // Bounce-back vertical velocity if hitting horizontal walls
            if (bottomSolid && fluid.Vy[index] < 0) {
                fluid.Vy[index] = -fluid.Vy[index] * 0.8;
                fluid.density[index] += Math.abs(fluid.Vy[index]) * 0.3; // Convert some velocity to density
            }
            if (topSolid && fluid.Vy[index] > 0) {
                fluid.Vy[index] = -fluid.Vy[index] * 0.8;
                fluid.density[index] += Math.abs(fluid.Vy[index]) * 0.3; // Convert some velocity to density
            }
            
            // Handle corner cases (diagonal bounces)
            if ((leftSolid || rightSolid) && (topSolid || bottomSolid)) {
                // If hitting a corner, reverse both components with more energy loss
                fluid.Vx[index] = -fluid.Vx[index] * 0.6;
                fluid.Vy[index] = -fluid.Vy[index] * 0.6;
                fluid.density[index] += Math.abs(fluid.Vx[index]) * 0.3; // Convert some velocity to density
                fluid.density[index] += Math.abs(fluid.Vy[index]) * 0.3; // Convert some velocity to density
            }
        }
    }
}

// Debug function to check solid map
function debugSolidMap() {
    let solidCount = 0;
    for (let i = 0; i < solidMap.length; i++) {
        if (solidMap[i]) solidCount++;
    }
    console.log('Total solid cells:', solidCount, 'out of', solidMap.length);
    console.log('Drop points:', dropPoints.length);
}

// Create solid obstacles and get drop points from the art
const { solids: artSolids, dropPoints } = parseArtForSolids();

// Debug the solid map
// debugSolidMap();

// Function to add a fluid drop at a random drop point
function addRandomDrop() {
    if (dropPoints.length === 0) return;
    
    // Pick a random drop point
    const dropPoint = dropPoints[Math.floor(Math.random() * dropPoints.length)];
    
    // Check if drop point is not solid
    if (!solidMap[IX(dropPoint.x, dropPoint.y)]) {
        // Add fluid at the drop point
        fluid.addDensity(dropPoint.x, dropPoint.y, 10.0);
        fluid.addVelocity(dropPoint.x, dropPoint.y, 0, -25); // Reduced velocity
    }
}

// --- Visualization Setup ---
// Define ASCII character map based on density (using the longer, original ramp)
const asciiChars = " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"; // Original long ramp

// Create density-dependent colored overlays
const createDensityOverlays = () => {
    // Make the parent container relative for absolute positioning
    tempArtElement.parentElement.style.position = 'relative';
    
    // Define density ranges and their corresponding colors
    const densityLayers = [
        { minDensity: 0, maxDensity: 0.20, color: 'rgba(0, 255, 255,0.5)', chars: " .'`^\",:;Il!i" },
        { minDensity: 0.20, maxDensity: 0.70, color: 'rgba(100, 150, 255,1)', chars: "><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0" },
        { minDensity: 0.70, maxDensity: 1.0, color: 'rgba(0, 20, 200,1)', chars: "Zmwqpdbkhao*#MW&8%B@$" }
    ];
    
    const overlays = [];
    
    densityLayers.forEach((layer, index) => {
        const overlay = document.createElement('pre');
        overlay.id = `fluidOverlay${index}`;
        
        // Copy exact styles from original element
        const originalStyles = window.getComputedStyle(tempArtElement);
        overlay.style.position = 'absolute';
        overlay.style.top = tempArtElement.offsetTop + 'px';
        overlay.style.left = tempArtElement.offsetLeft + 'px';
        overlay.style.width = tempArtElement.offsetWidth + 'px';
        overlay.style.height = tempArtElement.offsetHeight + 'px';
        overlay.style.background = 'transparent';
        overlay.style.color = layer.color;
        overlay.style.fontFamily = originalStyles.fontFamily;
        overlay.style.fontSize = (parseFloat(originalStyles.fontSize) * FONT_SCALE) + 'px';
        overlay.style.lineHeight = (parseFloat(originalStyles.lineHeight) * FONT_SCALE) + 'px';
        overlay.style.margin = '0';
        overlay.style.padding = originalStyles.padding;
        overlay.style.pointerEvents = 'none';
        overlay.style.whiteSpace = 'pre';
        overlay.style.overflow = 'hidden';
        overlay.style.zIndex = `${10 + index}`;
        
        tempArtElement.parentElement.appendChild(overlay);
        overlays.push({ element: overlay, layer: layer });
    });
    
    return overlays;
};

// Create overlay pre element that sits on top
const fluidOverlays = createDensityOverlays();

// Add debugging variables outside the animation loop
let frameCount = 0;
let layerUsageCount = [0, 0, 0];
let maxDensitySeen = 0;

// --- Animation Loop ---
function animate() {
    requestAnimationFrame(animate);

    // Add random drops with random intervals
    if (Math.random() < 0.6) {
        addRandomDrop();
    }
    
    // Update active cells before simulation for optimization
    updateActiveCells();
    
    // Enforce solid constraints before simulation step
    enforceSolidConstraints();
    
    // --- Step the Fluid Simulation ---
    fluid.step();
    
    // // Enforce solid constraints after each simulation step
    enforceSolidConstraints();

    // --- Update ASCII Visualization on all overlays ---
    // Reset debug counters every 60 frames
    frameCount++;
    if (frameCount % 60 === 0) {
        layerUsageCount = [0, 0, 0];
        maxDensitySeen = 0;
        
        // Log performance statistics
        const activeCells = activeMap.reduce((count, active) => count + (active ? 1 : 0), 0);
        console.log(`Active cells: ${activeCells}/${activeMap.length} (${(activeCells/activeMap.length*100).toFixed(1)}%)`);
    }

    // Generate separate ASCII strings for each density layer
    fluidOverlays.forEach((overlayData, layerIndex) => {
        let asciiString = "";
        
        for (let i = SCALED_HEIGHT - 1; i >= 0; i--) {
            for (let j = 0; j < SCALED_WIDTH; j++) {
                const index = IX(j, i);
                
                // Quick check for inactive cells
                if (!activeMap[index]) {
                    asciiString += ' ';
                    continue;
                }
                
                const densityValue = fluid.density[index];
                
                if (isNaN(densityValue) || densityValue === undefined) {
                    asciiString += ' ';
                    continue;
                }
                
                const normalizedDensity = Math.min(Math.max(densityValue / 2.0, 0), 1);
                const layer = overlayData.layer;
                
                // Track max density for debugging
                if (densityValue > maxDensitySeen) {
                    maxDensitySeen = densityValue;
                }
                
                // Only show characters if density is within this layer's range
                const isInRange = layerIndex === fluidOverlays.length - 1 
                    ? (normalizedDensity >= layer.minDensity && normalizedDensity <= layer.maxDensity) // Include 1.0 in the last layer
                    : (normalizedDensity >= layer.minDensity && normalizedDensity < layer.maxDensity);
                
                if (isInRange) {
                    // Track layer usage for debugging
                    layerUsageCount[layerIndex]++;
                    
                    // Map density within this layer's range to character index
                    const layerRange = layer.maxDensity - layer.minDensity;
                    const densityInLayer = (normalizedDensity - layer.minDensity) / layerRange;
                    const charIndex = Math.floor(densityInLayer * (layer.chars.length - 1));
                    const safeCharIndex = Math.max(0, Math.min(charIndex, layer.chars.length - 1));
                    const char = layer.chars[safeCharIndex];
                    asciiString += char !== undefined ? char : ' ';
                } else {
                    asciiString += ' ';
                }
            }
            asciiString += '\n';
        }
        
        overlayData.element.textContent = asciiString;
    });
}

animate(); // Start the animation loop
