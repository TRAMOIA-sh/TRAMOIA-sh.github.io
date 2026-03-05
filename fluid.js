// fluid simulation in ascii
// using Jos Stam's "Real-Time Fluid Dynamics for Games"
// https://www.dgp.toronto.edu/people/stam/reality/Research/pdf/GDC03.pdfd
// https://mikeash.com/pyblog/fluid-simulation-for-dummies.html

window.addEventListener('load', () => {

// add _ to padding the artpre element based on window width
const artPreElement = document.getElementById('artpre');
const spanCalculated = document.createElement('span');
spanCalculated.textContent = '_';
spanCalculated.style.fontSize = '16px';
spanCalculated.style.lineHeight = '16px';
spanCalculated.style.display = 'inline-block';
spanCalculated.style.width = '16px';
spanCalculated.style.position = 'absolute';
spanCalculated.style.top = '0';
spanCalculated.style.fontFamily = 'ibmcga';
spanCalculated.style.height = '16px';
spanCalculated.style.backgroundColor = 'transparent';
spanCalculated.style.color = 'transparent';
document.body.appendChild(spanCalculated);
const spanWidth = spanCalculated.offsetWidth;
const spanHeight = spanCalculated.offsetHeight;
// Add a buffer to handle potential rounding errors or font loading race conditions
const howManyCharactersWindowFit = Math.ceil(window.innerWidth / spanWidth) + 4;
document.body.removeChild(spanCalculated);

// Measure original width before adding underscores to calculate correct centering
const originalLines = artPreElement.innerText.split('\n');
const originalWidth = Math.max(...originalLines.map(l => l.length));

let multiplier;
if (window.innerWidth > 1024) {
    multiplier = 1.79;  // for desktop
} else {
    multiplier = 0.0; // for mobile
}
artPreElement.innerHTML += `_`.repeat(howManyCharactersWindowFit*multiplier);

// Calculate how many underscores were added for centering
const numPaddingUnderscores = Math.floor(howManyCharactersWindowFit * multiplier);

// Calculate padding spaces to add (center justify)
// Use the original width to ensure the art is centered in the expanded container
let numSpacesPadding = 0;
if (multiplier > 0) {
    // Center the art: Padding = (TotalWidth - ArtWidth) / 2
    numSpacesPadding = Math.ceil((numPaddingUnderscores - originalWidth)-1)/2;
    if (numSpacesPadding < 0) numSpacesPadding = 0;
}

// Pad each line inside artpre with spaces (on the left)
function padArtPreLines() {
    // Get list of lines as HTML (preserving color tags)
    // We'll operate on .innerHTML to not break <g1>, <g2>, <ww> etc.
    let lines = artPreElement.innerHTML.split('\n');
    for (let i = 0; i < lines.length-2; i++) {
        // Only pad lines that aren't trivially empty
        // But do not touch special decoration lines if any (optional, can be removed)
        if (lines[i].trim().length > 0) {
            lines[i] = ' '.repeat(numSpacesPadding) + lines[i];
        }
    }
    artPreElement.innerHTML = lines.join('\n');
}

// Run this after the underscores are added
padArtPreLines();


// Calculate fluid grid size based on actual art dimensions
const tempArtElement = document.getElementById('artpre');
const tempArtText = tempArtElement.innerText;
const tempArtLines = tempArtText.split('\n');
const WIDTH = Math.max(...tempArtLines.map(line => line.length));
// const WIDTH = window.innerWidth;
const HEIGHT = tempArtLines.length;

// Calculate scaling factor for higher resolution
const FONT_SCALE = 0.6; // This will make the font size effectively 8px (assuming original is 16px)
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
        const width = this.width;
        const height = this.height;
        const widthMinus1 = width - 1;
        const heightMinus1 = height - 1;
        
        for (let k = 0; k < iter; k++) {
            for (let j = 1; j < widthMinus1; j++) { // x-coord
                for (let i = 1; i < heightMinus1; i++) { // y-coord
                    const index = j + i * width; // IX(j, i) = j + i * width
                    
                    // Skip inactive cells for performance
                    if (!activeMap[index]) continue;
                    
                    // Skip solid cells
                    if (solidMap[index]) {
                        x[index] = 0;
                        continue;
                    }
                    
                    // Cache neighbor indices (avoid repeated IX() calls)
                    const idxRight = (j + 1) + i * width;
                    const idxLeft = (j - 1) + i * width;
                    const idxTop = j + (i + 1) * width;
                    const idxBottom = j + (i - 1) * width;
                    
                    x[index] =
                        (x0[index] +
                            a * (x[idxRight] +
                                 x[idxLeft] +
                                 x[idxTop] +
                                 x[idxBottom]
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
        const widthMinus2 = this.width - 2;
        const heightMinus2 = this.height - 2;
        // Use width/height in factor calculation
        const a = this.dt * diff * widthMinus2 * heightMinus2;
        // Avoid division by zero or very small numbers if a is close to 0
        const cRecip = 1.0 / (1 + 4 * a); // Assuming dx=dy=1, Gauss-Seidel uses 4 neighbors
        if (a === 0 || cRecip === Infinity || cRecip !== cRecip) { // NaN check without isNaN
            // If no diffusion, simply copy x0 to x for the internal cells
            const width = this.width;
            const height = this.height;
            for (let j = 1; j < width - 1; j++) {
                for (let i = 1; i < height - 1; i++) {
                    const index = j + i * width; // IX(j, i) = j + i * width
                    x[index] = x0[index];
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

        const width = this.width;
        const height = this.height;
        const widthMinus1 = width - 1;
        const heightMinus1 = height - 1;
        const widthMinus2 = width - 2;
        const heightMinus2 = height - 2;
        
        const dtx = this.dt * widthMinus2; // Scale dt by grid size for stable advection
        const dty = this.dt * heightMinus2;

        let s0, s1, t0, t1;
        let tmp1, tmp2, x, y;

        const NfloatW = width;
        const NfloatH = height;
        const NfloatWMinus1_5 = width - 1.5;
        const NfloatHMinus1_5 = height - 1.5;
        
        for (let j = 1; j < widthMinus1; j++) { // x-coord
            for (let i = 1; i < heightMinus1; i++) { // y-coord
                const index = j + i * width; // IX(j, i) = j + i * width
                
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
                x = x < 0.5 ? 0.5 : (x > NfloatWMinus1_5 ? NfloatWMinus1_5 : x);
                y = y < 0.5 ? 0.5 : (y > NfloatHMinus1_5 ? NfloatHMinus1_5 : y);

                // Get integer and fractional parts for interpolation
                j0 = Math.floor(x);
                j1 = j0 + 1;
                i0 = Math.floor(y);
                i1 = i0 + 1;

                // Check if we're trying to interpolate from solid areas
                // Cache width multiplication for index calculations (IX(x,y) = x + y * width)
                const idx00 = j0 + i0 * width;
                const idx01 = j0 + i1 * width;
                const idx10 = j1 + i0 * width;
                const idx11 = j1 + i1 * width;
                
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
                    reflectedX = reflectedX < 0.5 ? 0.5 : (reflectedX > NfloatWMinus1_5 ? NfloatWMinus1_5 : reflectedX);
                    reflectedY = reflectedY < 0.5 ? 0.5 : (reflectedY > NfloatHMinus1_5 ? NfloatHMinus1_5 : reflectedY);
                    
                    // Get new interpolation indices for reflected position
                    j0 = Math.floor(reflectedX);
                    j1 = j0 + 1;
                    i0 = Math.floor(reflectedY);
                    i1 = i0 + 1;
                    
                    // Cache width multiplication for reflected indices (IX(x,y) = x + y * width)
                    const ridx00 = j0 + i0 * width;
                    const ridx01 = j0 + i1 * width;
                    const ridx10 = j1 + i0 * width;
                    const ridx11 = j1 + i1 * width;
                    
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
        const width = this.width;
        const height = this.height;
        const widthMinus1 = width - 1;
        const heightMinus1 = height - 1;
        
        // Calculate divergence (using central differences, h=1)
        for (let j = 1; j < widthMinus1; j++) {
            for (let i = 1; i < heightMinus1; i++) {
                const index = j + i * width; // IX(j, i) = j + i * width
                
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
                
                // Cache neighbor indices (IX(x,y) = x + y * width)
                const idxRight = (j + 1) + i * width;
                const idxLeft = (j - 1) + i * width;
                const idxTop = j + (i + 1) * width;
                const idxBottom = j + (i - 1) * width;
                
                div[index] = -0.5 * (velocX[idxRight] - velocX[idxLeft] +
                                       velocY[idxTop] - velocY[idxBottom]);
                p[index] = 0; // Initialize pressure
            }
        }
        this.set_bnd(0, div);
        this.set_bnd(0, p);

        // Solve for pressure using lin_solve (Poisson equation)
        this.lin_solve(0, p, div, 1, 1.0 / 4.0, iter);

        // Subtract pressure gradient from velocity field (using central differences, h=1)
        for (let j = 1; j < widthMinus1; j++) {
            for (let i = 1; i < heightMinus1; i++) {
                const index = j + i * width; // IX(j, i) = j + i * width
                
                // Skip inactive cells for performance
                if (!activeMap[index]) continue;
                
                // Skip solid cells
                if (solidMap[index]) {
                    velocX[index] = 0;
                    velocY[index] = 0;
                    continue;
                }
                
                // Cache neighbor indices (IX(x,y) = x + y * width)
                const idxRight = (j + 1) + i * width;
                const idxLeft = (j - 1) + i * width;
                const idxTop = j + (i + 1) * width;
                const idxBottom = j + (i - 1) * width;
                
                velocX[index] -= 0.5 * (p[idxRight] - p[idxLeft]);
                velocY[index] -= 0.5 * (p[idxTop] - p[idxBottom]);
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
        const width = this.width;
        const height = this.height;
        const widthMinus1 = width - 1;
        const heightMinus1 = height - 1;
        const widthMinus2 = width - 2;
        const heightMinus2 = height - 2;
        
        // First handle solid constraints - only iterate through known solid cells
        // Instead of iterating through entire array, handle solids during boundary processing
        
        // Then handle regular boundaries
        for (let j = 1; j < widthMinus1; j++) {
            const idxTop = j + 0 * width; // IX(j, 0)
            const idxTopInner = j + 1 * width; // IX(j, 1)
            const idxBottom = j + heightMinus1 * width; // IX(j, height-1)
            const idxBottomInner = j + heightMinus2 * width; // IX(j, height-2)
            
            if (!solidMap[idxTop]) {
                x[idxTop] = b === 2 ? -x[idxTopInner] : x[idxTopInner];
            } else {
                x[idxTop] = 0;
            }
            if (!solidMap[idxBottom]) {
                x[idxBottom] = b === 2 ? -x[idxBottomInner] : x[idxBottomInner];
            } else {
                x[idxBottom] = 0;
            }
        }
        for (let i = 1; i < heightMinus1; i++) {
            const iWidth = i * width;
            const idxLeft = 0 + iWidth; // IX(0, i)
            const idxLeftInner = 1 + iWidth; // IX(1, i)
            const idxRight = widthMinus1 + iWidth; // IX(width-1, i)
            const idxRightInner = widthMinus2 + iWidth; // IX(width-2, i)
            
            if (!solidMap[idxLeft]) {
                x[idxLeft] = b === 1 ? -x[idxLeftInner] : x[idxLeftInner];
            } else {
                x[idxLeft] = 0;
            }
            if (!solidMap[idxRight]) {
                x[idxRight] = b === 1 ? -x[idxRightInner] : x[idxRightInner];
            } else {
                x[idxRight] = 0;
            }
        }

        // Handle corners (only if they're not solid)
        const idx00 = 0;
        const idx01 = width;
        const idx10 = 1;
        if (!solidMap[idx00]) {
            x[idx00] = 0.5 * (x[idx10] + x[idx01]);
        } else {
            x[idx00] = 0;
        }
        
        const idx0H = heightMinus1 * width;
        const idx0HInner = 1 + idx0H;
        const idx0HInner2 = (heightMinus2) * width;
        if (!solidMap[idx0H]) {
            x[idx0H] = 2.5 * (x[idx0HInner] + x[idx0HInner2]);
        } else {
            x[idx0H] = 0;
        }
        
        const idxW0 = widthMinus1;
        const idxW0Inner = widthMinus2;
        const idxW01 = widthMinus1 + width;
        if (!solidMap[idxW0]) {
            x[idxW0] = 0.5 * (x[idxW0Inner] + x[idxW01]);
        } else {
            x[idxW0] = 0;
        }
        
        const idxWH = widthMinus1 + idx0H;
        const idxWHInner1 = widthMinus2 + idx0H;
        const idxWHInner2 = widthMinus1 + idx0HInner2;
        if (!solidMap[idxWH]) {
            x[idxWH] = 0.5 * (x[idxWHInner1] + x[idxWHInner2]);
        } else {
            x[idxWH] = 0;
        }
        
        // Handle all solid cells in one pass (only those not on boundaries)
        for (let j = 1; j < widthMinus1; j++) {
            for (let i = 1; i < heightMinus1; i++) {
                const idx = j + i * width; // IX(j, i) = j + i * width
                if (solidMap[idx]) {
                    x[idx] = 0;
                }
            }
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
    const width = SCALED_WIDTH;
    const height = SCALED_HEIGHT;
    const size = width * height;
    
    // Reset active map
    activeMap.fill(false);
    
    for (let i = 0; i < size; i++) {
        // A cell is active if it has significant density, velocity, or is near solid boundaries
        const density = fluid.density[i];
        const vx = fluid.Vx[i];
        const vy = fluid.Vy[i];
        const hasSignificantDensity = density > ACTIVITY_THRESHOLD;
        const hasSignificantVelocity = (vx > ACTIVITY_THRESHOLD || vx < -ACTIVITY_THRESHOLD) || 
                                      (vy > ACTIVITY_THRESHOLD || vy < -ACTIVITY_THRESHOLD);
        
        if (hasSignificantDensity || hasSignificantVelocity || solidMap[i]) {
            activeMap[i] = true;
            
            // Also mark neighboring cells as active (fluid can spread)
            // Cache calculations
            const x = i % width;
            const y = (i - x) / width; // Equivalent to Math.floor(i / width) but faster
            
            // Optimize neighbor marking - only check valid neighbors (avoid nested loops)
            const hasLeft = x > 0;
            const hasRight = x < width - 1;
            const hasTop = y > 0;
            const hasBottom = y < height - 1;
            
            if (hasTop) {
                if (hasLeft) activeMap[i - width - 1] = true;
                activeMap[i - width] = true;
                if (hasRight) activeMap[i - width + 1] = true;
            }
            if (hasLeft) activeMap[i - 1] = true;
            if (hasRight) activeMap[i + 1] = true;
            if (hasBottom) {
                if (hasLeft) activeMap[i + width - 1] = true;
                activeMap[i + width] = true;
                if (hasRight) activeMap[i + width + 1] = true;
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
    const width = SCALED_WIDTH;
    const height = SCALED_HEIGHT;
    const widthMinus1 = width - 1;
    const heightMinus1 = height - 1;
    
    // Apply bounce-back boundary conditions only to active cells
    for (let j = 1; j < widthMinus1; j++) {
        for (let i = 1; i < heightMinus1; i++) {
            const index = j + i * width; // IX(j, i) = j + i * width
            
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
            
            // Check for solid neighbors and apply bounce-back (cache neighbor indices, IX(x,y) = x + y * width)
            const idxLeft = (j - 1) + i * width;
            const idxRight = (j + 1) + i * width;
            const idxBottom = j + (i - 1) * width;
            const idxTop = j + (i + 1) * width;
            const leftSolid = solidMap[idxLeft];
            const rightSolid = solidMap[idxRight];
            const bottomSolid = solidMap[idxBottom];
            const topSolid = solidMap[idxTop];
            
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
    
    // Check if drop point is not solid (cache index calculation)
    const dropIndex = dropPoint.x + dropPoint.y * SCALED_WIDTH;
    if (!solidMap[dropIndex]) {
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

// --- Mouse Interaction ---
let lastMouseX = -1;
let lastMouseY = -1;

function handleMouseMove(e) {
    const rect = tempArtElement.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    // Check if inside
    if (x < 0 || x > rect.width || y < 0 || y > rect.height) {
        lastMouseX = -1;
        lastMouseY = -1;
        return;
    }

    // Map to grid coordinates
    const gridX = Math.floor((x / rect.width) * SCALED_WIDTH);
    // Visual Y increases downwards (0 at top). Simulation Y increases upwards (SCALED_HEIGHT-1 at top).
    const visualGridY = Math.floor((y / rect.height) * SCALED_HEIGHT);
    const gridY = SCALED_HEIGHT - 1 - visualGridY;

    if (lastMouseX !== -1 && lastMouseY !== -1) {
        // Calculate velocity
        const dx = x - lastMouseX;
        const dy = y - lastMouseY;

        // Scale velocity to be meaningful for simulation
        // Subtle force: Reduced multiplier from 5.0 to 1.5
        const force = 0.2; 
        const velX = dx * force;
        const velY = -dy * force; // Screen Y down is positive, Sim Y up is positive. So dy>0 (down) -> velY<0 (down)

        // Add velocity and minimal density at current position
        const radius = 1;
        addForce(gridX, gridY, velX, velY, radius);
    }

    lastMouseX = x;
    lastMouseY = y;
}

function addForce(cx, cy, vx, vy, r) {
     // Loop around center
     for (let i = -r; i <= r; i++) {
         for (let j = -r; j <= r; j++) {
             const x = cx + i;
             const y = cy + j;
             
             // Bounds check
             if (x >= 0 && x < SCALED_WIDTH && y >= 0 && y < SCALED_HEIGHT) {
                 const index = x + y * SCALED_WIDTH;
                 // Check solid
                 if (!solidMap[index]) {
                     fluid.addVelocity(x, y, vx, vy);
                     // Add very subtle density trail to see the interaction
                     const speed = Math.sqrt(vx*vx + vy*vy);
                     // Reduced density injection: max 1.0 instead of 5.0
                     fluid.addDensity(x, y, Math.min(speed * 0.05, 1.0)); 
                 }
             }
         }
     }
}

document.addEventListener('mousemove', handleMouseMove);

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
    const width = SCALED_WIDTH;
    const height = SCALED_HEIGHT;
    const densityArray = fluid.density;
    const densityScale = 1.0 / 2.0; // Pre-calculate 1/2.0
    const isLastLayer = fluidOverlays.length - 1;
    
    fluidOverlays.forEach((overlayData, layerIndex) => {
        const layer = overlayData.layer;
        const layerChars = layer.chars;
        const charsLengthMinus1 = layerChars.length - 1;
        const layerRange = layer.maxDensity - layer.minDensity;
        const layerRangeRecip = 1.0 / layerRange;
        const isLast = layerIndex === isLastLayer;
        const minDensity = layer.minDensity;
        const maxDensity = layer.maxDensity;
        
        let asciiString = "";
        
        for (let i = height - 1; i >= 0; i--) {
            for (let j = 0; j < width; j++) {
                const index = j + i * width; // IX(j, i) = j + i * width
                
                // Quick check for inactive cells
                if (!activeMap[index]) {
                    asciiString += ' ';
                    continue;
                }
                
                const densityValue = densityArray[index];
                
                // Check for NaN/undefined (faster than isNaN check)
                if (densityValue !== densityValue || densityValue === undefined) {
                    asciiString += ' ';
                    continue;
                }
                
                // Track max density for debugging (only when needed)
                if (densityValue > maxDensitySeen) {
                    maxDensitySeen = densityValue;
                }
                
                // Normalize density (optimize Math.min/Math.max)
                const normalizedDensity = densityValue * densityScale;
                const clampedDensity = normalizedDensity < 0 ? 0 : (normalizedDensity > 1 ? 1 : normalizedDensity);
                
                // Only show characters if density is within this layer's range
                const isInRange = isLast 
                    ? (clampedDensity >= minDensity && clampedDensity <= maxDensity)
                    : (clampedDensity >= minDensity && clampedDensity < maxDensity);
                
                if (isInRange) {
                    // Track layer usage for debugging
                    layerUsageCount[layerIndex]++;
                    
                    // Map density within this layer's range to character index
                    const densityInLayer = (clampedDensity - minDensity) * layerRangeRecip;
                    const charIndex = Math.floor(densityInLayer * charsLengthMinus1);
                    const safeCharIndex = charIndex < 0 ? 0 : (charIndex > charsLengthMinus1 ? charsLengthMinus1 : charIndex);
                    const char = layerChars[safeCharIndex];
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
});
