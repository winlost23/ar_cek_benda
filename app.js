/**
 * AR Smart Vision - Deteksi Objek Furnitur, Perangkat Komputer, dan Benda Ruangan
 * Target: Kursi, Meja, Lemari, Mouse Komputer, Laptop, PC, Monitor, Speaker, Gelas, Lampu, Pintu
 * Menggunakan TensorFlow.js (COCO-SSD + MobileNet + Heuristik Visual) dengan AR HUD Overlay
 */

const video = document.getElementById('webcam');
const canvas = document.getElementById('ar-canvas');
const ctx = canvas.getContext('2d');

const loadingOverlay = document.getElementById('loading-overlay');
const loadingStatus = document.getElementById('loading-status');
const progressFill = document.getElementById('progress-fill');
const fpsCounter = document.getElementById('fps-counter');
const hudReticle = document.getElementById('hud-reticle');

// Focus Card Elements
const focusCard = document.getElementById('focus-card');
const focusIcon = document.getElementById('focus-icon');
const focusTitle = document.getElementById('focus-title');
const focusDesc = document.getElementById('focus-desc');
const focusScore = document.getElementById('focus-score');

// Daftar Kategori Target
const TARGET_CATEGORIES = [
    'kursi',
    'meja',
    'lemari',
    'mouse',
    'laptop',
    'pc',
    'monitor',
    'speaker',
    'gelas',
    'lampu',
    'pintu'
];

// Counters & Cards DOM Map
const countElements = {};
const cardElements = {};
TARGET_CATEGORIES.forEach(cat => {
    countElements[cat] = document.getElementById(`count-${cat}`);
    cardElements[cat] = document.getElementById(`card-${cat}`);
});

// Controls
const btnFilterMode = document.getElementById('btn-filter-mode');
const btnVoice = document.getElementById('btn-voice');
const btnSwitchCam = document.getElementById('btn-switch-cam');
const confidenceSlider = document.getElementById('confidence-slider');
const confVal = document.getElementById('conf-val');
const uploadInput = document.getElementById('upload-input');
const uploadedImage = document.getElementById('uploaded-image');
const btnBackCam = document.getElementById('btn-back-cam');
const btnSnapshot = document.getElementById('btn-snapshot');

// State
let model = null;
let mobilenetModel = null;
let isTargetOnly = true;     // Default: Fokus ke 11 benda target
let isVoiceEnabled = false;
let minConfidence = 0.40;
let currentFacingMode = 'environment'; // Gunakan kamera belakang secara default di HP
let isStaticImageMode = false;
let stream = null;
let lastSpokenObject = '';
let lastSpokenTime = 0;

// Offscreen Canvas untuk analisis pencahayaan (Lampu) & crop MobileNet
const analysisCanvas = document.createElement('canvas');
analysisCanvas.width = 96;
analysisCanvas.height = 72;
const analysisCtx = analysisCanvas.getContext('2d', { willReadFrequently: true });

const cropCanvas = document.createElement('canvas');
cropCanvas.width = 224;
cropCanvas.height = 224;
const cropCtx = cropCanvas.getContext('2d');

// State hasil deteksi tambahan MobileNet (diperbarui secara berkala)
let lastMobileNetCheck = 0;
let latestMobileNetResult = null;
let isMobileNetBusy = false;

// FPS Calculation
let lastFrameTime = performance.now();
let frameCount = 0;
let currentFps = 0;

/**
 * Definisi Metadata untuk 11 Kategori Benda Target
 */
const CATEGORY_META = {
    kursi: {
        category: 'kursi',
        label: 'KURSI',
        subLabel: 'Kursi / Tempat Duduk',
        icon: '🪑',
        color: '#10b981',
        desc: 'Furnitur tempat duduk terdeteksi di area AR.'
    },
    meja: {
        category: 'meja',
        label: 'MEJA',
        subLabel: 'Meja (Table / Desk)',
        icon: '🪵',
        color: '#f59e0b',
        desc: 'Permukaan meja kerja atau meja makan terdeteksi.'
    },
    lemari: {
        category: 'lemari',
        label: 'LEMARI',
        subLabel: 'Lemari / Kabinet Penyimpanan',
        icon: '🗄️',
        color: '#8b5cf6',
        desc: 'Unit lemari atau rak penyimpanan terdeteksi.'
    },
    mouse: {
        category: 'mouse',
        label: 'MOUSE KOMPUTER',
        subLabel: 'Mouse Komputer (Mouse)',
        icon: '🖱️',
        color: '#06b6d4',
        desc: 'Perangkat penunjuk mouse komputer terdeteksi.'
    },
    laptop: {
        category: 'laptop',
        label: 'LAPTOP',
        subLabel: 'Laptop / Notebook Komputer',
        icon: '💻',
        color: '#3b82f6',
        desc: 'Komputer jinjing (laptop) terdeteksi.'
    },
    pc: {
        category: 'pc',
        label: 'PC / KOMPUTER',
        subLabel: 'PC Desktop (CPU / Keyboard)',
        icon: '🖳',
        color: '#6366f1',
        desc: 'Perangkat PC desktop / komputer kerja terdeteksi.'
    },
    monitor: {
        category: 'monitor',
        label: 'MONITOR',
        subLabel: 'Monitor / Layar Tampilan',
        icon: '🖥️',
        color: '#0ea5e9',
        desc: 'Layar monitor komputer atau display visual terdeteksi.'
    },
    speaker: {
        category: 'speaker',
        label: 'SPEAKER',
        subLabel: 'Speaker Audio / Pengeras Suara',
        icon: '🔊',
        color: '#ec4899',
        desc: 'Perangkat pengeras suara (speaker) terdeteksi.'
    },
    gelas: {
        category: 'gelas',
        label: 'GELAS',
        subLabel: 'Gelas / Cangkir Minum',
        icon: '🥛',
        color: '#14b8a6',
        desc: 'Wadah minum (gelas/cangkir/mug) terdeteksi.'
    },
    lampu: {
        category: 'lampu',
        label: 'LAMPU',
        subLabel: 'Lampu / Sumber Pencahayaan',
        icon: '💡',
        color: '#eab308',
        desc: 'Lampu ruangan, lampu meja, atau sumber cahaya terdeteksi.'
    },
    pintu: {
        category: 'pintu',
        label: 'PINTU',
        subLabel: 'Pintu Ruangan (Door)',
        icon: '🚪',
        color: '#f97316',
        desc: 'Akses pintu ruangan atau panel pintu terdeteksi.'
    }
};

/**
 * Pemetaan langsung kelas COCO-SSD ke Kategori Target
 */
const COCO_DIRECT_MAP = {
    // 1. KURSI
    'chair': CATEGORY_META.kursi,
    'couch': {
        ...CATEGORY_META.kursi,
        label: 'KURSI / SOFA',
        subLabel: 'Sofa / Kursi Ruang Tamu',
        icon: '🛋️'
    },
    'bench': {
        ...CATEGORY_META.kursi,
        label: 'KURSI BANGKU',
        subLabel: 'Bangku Panjang (Bench)'
    },

    // 2. MEJA
    'dining table': CATEGORY_META.meja,

    // 3. LEMARI
    'refrigerator': CATEGORY_META.lemari,
    'oven': {
        ...CATEGORY_META.lemari,
        label: 'LEMARI KABINET',
        subLabel: 'Kabinet / Lemari Bawah'
    },
    'microwave': {
        ...CATEGORY_META.lemari,
        label: 'LEMARI RAK',
        subLabel: 'Kompartemen / Kabinet'
    },
    'book': {
        ...CATEGORY_META.lemari,
        label: 'RAK / LEMARI BUKU',
        subLabel: 'Bagian Lemari Rak Buku',
        icon: '📚'
    },

    // 4. MOUSE KOMPUTER
    'mouse': CATEGORY_META.mouse,

    // 5. LAPTOP
    'laptop': CATEGORY_META.laptop,

    // 6. PC (Komputer Desktop / Keyboard / CPU)
    'keyboard': {
        ...CATEGORY_META.pc,
        label: 'PC (KEYBOARD)',
        subLabel: 'PC Desktop / Keyboard Komputer'
    },

    // 7. MONITOR
    'tv': CATEGORY_META.monitor,

    // 8. GELAS
    'cup': CATEGORY_META.gelas,
    'wine glass': {
        ...CATEGORY_META.gelas,
        label: 'GELAS KACA',
        subLabel: 'Gelas Kaca (Glass)'
    },

    // 9. LAMPU (Traffic light di dalam ruangan umumnya adalah lampu gantung/dinding/LED)
    'traffic light': CATEGORY_META.lampu
};

/**
 * Pemetaan kata kunci kelas MobileNet (ImageNet 1000 kelas) ke 11 Kategori Target
 */
const MOBILENET_KEYWORDS = [
    { keywords: ['computer mouse', 'mouse'], category: 'mouse' },
    { keywords: ['laptop', 'notebook computer', 'notebook'], category: 'laptop' },
    { keywords: ['desktop computer', 'computer keyboard', 'space bar', 'typewriter keyboard', 'hard disc', 'modem'], category: 'pc' },
    { keywords: ['monitor', 'crt screen', 'screen', 'television', 'home theater'], category: 'monitor' },
    { keywords: ['loudspeaker', 'speaker', 'subwoofer', 'radio', 'cassette player', 'tape player', 'ipod', 'microphone'], category: 'speaker' },
    { keywords: ['water glass', 'beer glass', 'goblet', 'coffee mug', 'cup', 'espresso', 'measuring cup', 'beaker', 'pitcher', 'water bottle', 'pop bottle'], category: 'gelas' },
    { keywords: ['table lamp', 'lampshade', 'lamp shade', 'spotlight', 'chandelier', 'candle', 'torch', 'jack-o\'-lantern', 'street sign', 'traffic light'], category: 'lampu' },
    { keywords: ['sliding door', 'shoji', 'window screen', 'window shade', 'turnstile', 'prison', 'safe'], category: 'pintu' },
    { keywords: ['barber chair', 'folding chair', 'rocking chair', 'studio couch', 'park bench', 'throne'], category: 'kursi' },
    { keywords: ['desk', 'dining table', 'pool table', 'pedestal'], category: 'meja' },
    { keywords: ['wardrobe', 'closet', 'bookcase', 'china cabinet', 'medicine chest', 'file cabinet', 'filing cabinet', 'chiffonier', 'entertainment center', 'refrigerator', 'icebox'], category: 'lemari' }
];

function matchMobileNetClass(className) {
    const lower = className.toLowerCase();
    for (const entry of MOBILENET_KEYWORDS) {
        for (const kw of entry.keywords) {
            if (lower.includes(kw)) {
                return entry.category;
            }
        }
    }
    return null;
}

/**
 * Mengambil statistik kecerahan & warna pada area bounding box (di-downsample ke analysisCanvas)
 */
function getRegionVisualStats(rawX, rawY, rawW, rawH, sourceWidth, sourceHeight) {
    const ax = Math.max(0, Math.floor((rawX / sourceWidth) * analysisCanvas.width));
    const ay = Math.max(0, Math.floor((rawY / sourceHeight) * analysisCanvas.height));
    const aw = Math.max(1, Math.min(analysisCanvas.width - ax, Math.floor((rawW / sourceWidth) * analysisCanvas.width)));
    const ah = Math.max(1, Math.min(analysisCanvas.height - ay, Math.floor((rawH / sourceHeight) * analysisCanvas.height)));

    try {
        const imgData = analysisCtx.getImageData(ax, ay, aw, ah).data;
        let totalLum = 0;
        let brightPixels = 0;
        let darkPixels = 0;
        const count = aw * ah;

        for (let i = 0; i < imgData.length; i += 4) {
            const r = imgData[i];
            const g = imgData[i + 1];
            const b = imgData[i + 2];
            const lum = 0.299 * r + 0.587 * g + 0.114 * b;
            totalLum += lum;
            if (lum > 225) brightPixels++;
            if (lum < 75) darkPixels++;
        }

        return {
            meanLum: totalLum / Math.max(count, 1),
            brightRatio: brightPixels / Math.max(count, 1),
            darkRatio: darkPixels / Math.max(count, 1)
        };
    } catch (e) {
        return { meanLum: 128, brightRatio: 0, darkRatio: 0 };
    }
}

/**
 * Klasifikasi cerdas: Mengubah hasil prediksi mentah menjadi informasi AR untuk 11 Benda Target
 */
function classifyObject(prediction, sourceWidth, sourceHeight, allPredictions) {
    const rawClass = prediction.class.toLowerCase();
    const [x, y, width, height] = prediction.bbox;
    const aspectRatio = height / Math.max(width, 1);
    const relativeArea = (width * height) / Math.max(sourceWidth * sourceHeight, 1);
    const visual = getRegionVisualStats(x, y, width, height, sourceWidth, sourceHeight);

    const hasMonitorOrLaptopNearby = allPredictions.some(p => {
        const c = p.class.toLowerCase();
        return c === 'tv' || c === 'laptop' || c === 'keyboard' || c === 'mouse';
    });

    // 1. Heuristik LAMPU: Jika objek bulat/gantung di bagian atas/tengah ruangan dan sangat terang (menyala)
    if (
        ['clock', 'frisbee', 'sports ball', 'umbrella', 'bowl', 'vase', 'cup', 'wine glass', 'potted plant', 'traffic light'].includes(rawClass) &&
        (visual.brightRatio > 0.22 || visual.meanLum > 205) &&
        relativeArea < 0.35
    ) {
        return {
            ...CATEGORY_META.lampu,
            isTarget: true
        };
    }

    // 2. Heuristik PINTU: Objek tinggi vertikal seperti panel pintu ruangan
    if (
        (rawClass === 'surfboard' || rawClass === 'door') ||
        ((rawClass === 'refrigerator' || rawClass === 'dining table' || rawClass === 'bed' || rawClass === 'tv') && aspectRatio > 1.82 && relativeArea > 0.10)
    ) {
        return {
            ...CATEGORY_META.pintu,
            isTarget: true
        };
    }

    // 3. Heuristik SPEAKER: Objek berbentuk kotak speaker (sering terdeteksi sebagai remote/cell phone/vase/toaster/suitcase kecil)
    if (
        (rawClass === 'toaster') ||
        ((rawClass === 'remote' || rawClass === 'cell phone') && aspectRatio > 1.35 && visual.darkRatio > 0.35) ||
        (rawClass === 'vase' && visual.darkRatio > 0.45 && hasMonitorOrLaptopNearby) ||
        ((rawClass === 'suitcase' || rawClass === 'handbag') && relativeArea < 0.14 && aspectRatio >= 1.1 && aspectRatio <= 2.2 && visual.darkRatio > 0.4)
    ) {
        return {
            ...CATEGORY_META.speaker,
            isTarget: true
        };
    }

    // 4. Heuristik MOUSE KOMPUTER tambahan: 'remote' kecil bulat/oval di dekat laptop/keyboard/monitor
    if (rawClass === 'remote' && relativeArea < 0.06 && hasMonitorOrLaptopNearby) {
        return {
            ...CATEGORY_META.mouse,
            isTarget: true
        };
    }

    // 5. Heuristik PC (CPU Casing / Tower Komputer): Kotak vertikal sedang di dekat monitor/meja
    if (
        (rawClass === 'suitcase' || rawClass === 'microwave' || rawClass === 'oven' || rawClass === 'refrigerator') &&
        hasMonitorOrLaptopNearby &&
        relativeArea >= 0.04 &&
        relativeArea <= 0.30 &&
        aspectRatio >= 1.05 &&
        aspectRatio <= 2.3
    ) {
        return {
            ...CATEGORY_META.pc,
            label: 'PC (CPU TOWER)',
            subLabel: 'Unit Casing PC Desktop',
            isTarget: true
        };
    }

    // 6. Heuristik GELAS tambahan: 'bottle' atau 'vase' bening/kecil berukuran gelas minum
    if ((rawClass === 'bottle' || rawClass === 'vase') && aspectRatio < 2.3 && relativeArea < 0.18) {
        return {
            ...CATEGORY_META.gelas,
            label: rawClass === 'bottle' ? 'GELAS / BOTOL MINUM' : 'GELAS',
            subLabel: 'Wadah Minum / Gelas',
            isTarget: true
        };
    }

    // 7. Jika langsung cocok di COCO_DIRECT_MAP
    if (COCO_DIRECT_MAP[rawClass]) {
        // Khusus 'dining table' yang tinggi vertikal -> bisa jadi Lemari
        if (rawClass === 'dining table' && aspectRatio > 1.50) {
            return {
                ...CATEGORY_META.lemari,
                subLabel: 'Lemari / Rak Vertikal',
                isTarget: true
            };
        }
        return {
            ...COCO_DIRECT_MAP[rawClass],
            isTarget: true
        };
    }

    // 8. Heuristik tambahan untuk Meja & Lemari dari objek besar di ruangan
    if (rawClass === 'bed' && aspectRatio < 0.85) {
        return {
            ...CATEGORY_META.meja,
            label: 'MEJA / BIDANG DATAR',
            subLabel: 'Permukaan Furnitur Datar',
            isTarget: true
        };
    }

    if ((rawClass === 'suitcase' || rawClass === 'handbag') && relativeArea >= 0.14 && aspectRatio > 1.2) {
        return {
            ...CATEGORY_META.lemari,
            label: 'LEMARI / KABINET',
            subLabel: 'Unit Lemari Penyimpanan',
            isTarget: true
        };
    }

    // Objek lain (non-target)
    return {
        category: 'other',
        label: rawClass.toUpperCase(),
        subLabel: `Objek Lain (${rawClass})`,
        icon: '📦',
        color: '#64748b',
        desc: 'Objek umum di sekitar ruangan.',
        isTarget: false
    };
}

/**
 * Deteksi Sumber Cahaya / Lampu Menyala langsung dari kanvas analisis (untuk bohlam / lampu plafon / lampu meja terang)
 */
function detectBrightLampBoxes(sourceWidth, sourceHeight, existingPredictions) {
    const w = analysisCanvas.width;
    const h = analysisCanvas.height;
    let imgData;
    try {
        imgData = analysisCtx.getImageData(0, 0, w, h).data;
    } catch (e) {
        return [];
    }

    // Cari titik paling terang di 65% bagian atas gambar
    const maxScanY = Math.floor(h * 0.68);
    let sumX = 0, sumY = 0, count = 0;
    let minX = w, minY = h, maxX = 0, maxY = 0;
    let midLumSum = 0;

    for (let y = 2; y < maxScanY; y++) {
        for (let x = 2; x < w - 2; x++) {
            const idx = (y * w + x) * 4;
            const r = imgData[idx];
            const g = imgData[idx + 1];
            const b = imgData[idx + 2];
            const lum = 0.299 * r + 0.587 * g + 0.114 * b;
            midLumSum += lum;

            if (lum > 244 && r > 235 && g > 230) {
                sumX += x;
                sumY += y;
                count++;
                if (x < minX) minX = x;
                if (x > maxX) maxX = x;
                if (y < minY) minY = y;
                if (y > maxY) maxY = y;
            }
        }
    }

    const avgAmbientLum = midLumSum / Math.max(maxScanY * w, 1);
    const totalPixels = w * h;
    // Hanya anggap lampu jika ada titik cahaya terkonsentrasi (bukan seluruh layar putih silau)
    if (count >= 4 && count < totalPixels * 0.09 && avgAmbientLum < 195) {
        const bw = (maxX - minX + 1);
        const bh = (maxY - minY + 1);
        if (bw < w * 0.45 && bh < h * 0.45) {
            const padX = Math.max(4, bw * 0.35);
            const padY = Math.max(4, bh * 0.35);
            const realX = Math.max(0, ((minX - padX) / w) * sourceWidth);
            const realY = Math.max(0, ((minY - padY) / h) * sourceHeight);
            const realW = Math.min(sourceWidth - realX, ((bw + padX * 2) / w) * sourceWidth);
            const realH = Math.min(sourceHeight - realY, ((bh + padY * 2) / h) * sourceHeight);

            const cx = realX + realW / 2;
            const cy = realY + realH / 2;

            // Pastikan belum tumpang tindih dengan prediksi lain
            const overlaps = existingPredictions.some(p => {
                const [px, py, pw, ph] = p.bbox;
                return cx >= px && cx <= px + pw && cy >= py && cy <= py + ph;
            });

            if (!overlaps) {
                return [{
                    bbox: [realX, realY, realW, realH],
                    class: 'traffic light',
                    score: Math.min(0.88, Math.max(minConfidence + 0.08, 0.62)),
                    isCustomLamp: true
                }];
            }
        }
    }
    return [];
}

/**
 * Menjalankan klasifikasi MobileNet pada bagian tengah kamera/gambar untuk membantu mengenali
 * Speaker, Lampu, Pintu, PC, Monitor, Mouse, Laptop, Gelas, dll.
 */
async function updateMobileNetClassification(sourceElement, sourceWidth, sourceHeight, forceWait = false) {
    if (!mobilenetModel || !sourceWidth || !sourceHeight) return;
    const now = performance.now();
    if (!forceWait && (isMobileNetBusy || now - lastMobileNetCheck < 480)) return;

    isMobileNetBusy = true;
    lastMobileNetCheck = now;

    try {
        // Ambil area tengah (reticle focus area) sebesar 60% dari dimensi terkecil
        const cropW = sourceWidth * 0.58;
        const cropH = sourceHeight * 0.58;
        const sx = (sourceWidth - cropW) / 2;
        const sy = (sourceHeight - cropH) / 2;

        cropCtx.clearRect(0, 0, 224, 224);
        cropCtx.drawImage(sourceElement, sx, sy, cropW, cropH, 0, 0, 224, 224);

        const results = await mobilenetModel.classify(cropCanvas, 4);
        let matched = null;

        for (const item of results) {
            const cat = matchMobileNetClass(item.className);
            if (cat && item.probability >= Math.max(0.16, minConfidence * 0.55)) {
                matched = {
                    category: cat,
                    rawName: item.className.split(',')[0],
                    score: Math.min(0.95, item.probability + 0.22),
                    bbox: [sx + cropW * 0.08, sy + cropH * 0.08, cropW * 0.84, cropH * 0.84],
                    timestamp: performance.now()
                };
                break;
            }
        }

        latestMobileNetResult = matched;
    } catch (err) {
        // Abaikan jika frame belum siap
    } finally {
        isMobileNetBusy = false;
    }
}

/**
 * Inisialisasi Kamera WebRTC
 */
async function setupCamera() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
    }

    const constraints = {
        audio: false,
        video: {
            facingMode: currentFacingMode,
            width: { ideal: 1280 },
            height: { ideal: 720 }
        }
    };

    try {
        stream = await navigator.mediaDevices.getUserMedia(constraints);
        video.srcObject = stream;
        return new Promise((resolve) => {
            video.onloadedmetadata = () => {
                video.play();
                resizeCanvas();
                resolve();
            };
        });
    } catch (err) {
        console.warn('Gagal mengakses kamera utama, mencoba fallback kamera:', err);
        stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        video.srcObject = stream;
        return new Promise((resolve) => {
            video.onloadedmetadata = () => {
                video.play();
                resizeCanvas();
                resolve();
            };
        });
    }
}

function resizeCanvas() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
}

window.addEventListener('resize', () => {
    resizeCanvas();
    if (isStaticImageMode) {
        detectStaticImage();
    }
});

/**
 * Menggambar Kotak AR 3D-style & Pin Holografik di atas objek
 */
function drawAROverlay(rawPredictions, sourceWidth, sourceHeight) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Hitung skala object-fit: cover agar bounding box pas dengan layar
    const scale = Math.max(canvas.width / sourceWidth, canvas.height / sourceHeight);
    const offsetX = (canvas.width - sourceWidth * scale) / 2;
    const offsetY = (canvas.height - sourceHeight * scale) / 2;

    // Inisialisasi tally untuk 11 kategori
    const tally = {};
    TARGET_CATEGORIES.forEach(cat => { tally[cat] = 0; });

    let centerLockedObject = null;
    let minCenterDist = Infinity;

    const screenCenterX = canvas.width / 2;
    const screenCenterY = canvas.height / 2;

    // Gabungkan prediksi COCO-SSD + Deteksi Lampu + MobileNet Center Assistance
    const combinedPredictions = [...rawPredictions];

    // Tambah deteksi sumber cahaya lampu jika ada
    const lampBoxes = detectBrightLampBoxes(sourceWidth, sourceHeight, combinedPredictions);
    combinedPredictions.push(...lampBoxes);

    // Cek apakah hasil MobileNet masih segar (< 1.5 detik) dan belum diwakili oleh COCO-SSD di tengah
    if (latestMobileNetResult && (isStaticImageMode || performance.now() - latestMobileNetResult.timestamp < 1500)) {
        const mnCat = latestMobileNetResult.category;
        const alreadyDetectedCenter = combinedPredictions.some(p => {
            if (p.score < minConfidence * 0.75) return false;
            const info = classifyObject(p, sourceWidth, sourceHeight, combinedPredictions);
            return info.category === mnCat;
        });

        if (!alreadyDetectedCenter && latestMobileNetResult.score >= minConfidence) {
            combinedPredictions.push({
                bbox: latestMobileNetResult.bbox,
                class: mnCat,
                score: latestMobileNetResult.score,
                forcedInfo: {
                    ...CATEGORY_META[mnCat],
                    subLabel: `${CATEGORY_META[mnCat].label} (${latestMobileNetResult.rawName})`,
                    isTarget: true
                }
            });
        }
    }

    combinedPredictions.forEach(pred => {
        if (pred.score < minConfidence) return;

        const info = pred.forcedInfo || classifyObject(pred, sourceWidth, sourceHeight, combinedPredictions);
        if (isTargetOnly && !info.isTarget) return;

        // Tambah hitungan jika termasuk salah satu dari 11 benda target
        if (tally[info.category] !== undefined) {
            tally[info.category]++;
        }

        // Transformasi koordinat ke ukuran layar
        const [rawX, rawY, rawW, rawH] = pred.bbox;
        const x = rawX * scale + offsetX;
        const y = rawY * scale + offsetY;
        const w = rawW * scale;
        const h = rawH * scale;

        const objCenterX = x + w / 2;
        const objCenterY = y + h / 2;

        // Cek apakah objek berada dekat dengan titik tengah reticle AR
        const distToCenter = Math.hypot(objCenterX - screenCenterX, objCenterY - screenCenterY);
        if (distToCenter < minCenterDist && distToCenter < Math.min(canvas.width, canvas.height) * 0.42) {
            minCenterDist = distToCenter;
            centerLockedObject = { ...info, score: pred.score };
        }

        const confidencePct = Math.round(pred.score * 100);
        const color = info.color;

        ctx.save();

        // 1. Semi-transparent AR Fill
        ctx.fillStyle = hexToRgba(color, 0.12);
        ctx.fillRect(x, y, w, h);

        // 2. Dashed Outer Frame
        ctx.strokeStyle = hexToRgba(color, 0.45);
        ctx.lineWidth = 1.5;
        ctx.setLineDash([6, 6]);
        ctx.strokeRect(x, y, w, h);
        ctx.setLineDash([]);

        // 3. Sudut Siku AR (Futuristic Corner Brackets)
        const cornerLen = Math.min(26, w * 0.25, h * 0.25);
        ctx.strokeStyle = color;
        ctx.lineWidth = 3.5;
        ctx.shadowColor = color;
        ctx.shadowBlur = 10;

        // Top-Left
        ctx.beginPath();
        ctx.moveTo(x, y + cornerLen);
        ctx.lineTo(x, y);
        ctx.lineTo(x + cornerLen, y);
        ctx.stroke();

        // Top-Right
        ctx.beginPath();
        ctx.moveTo(x + w - cornerLen, y);
        ctx.lineTo(x + w, y);
        ctx.lineTo(x + w, y + cornerLen);
        ctx.stroke();

        // Bottom-Left
        ctx.beginPath();
        ctx.moveTo(x, y + h - cornerLen);
        ctx.lineTo(x, y + h);
        ctx.lineTo(x + cornerLen, y + h);
        ctx.stroke();

        // Bottom-Right
        ctx.beginPath();
        ctx.moveTo(x + w - cornerLen, y + h);
        ctx.lineTo(x + w, y + h);
        ctx.lineTo(x + w, y + h - cornerLen);
        ctx.stroke();

        // 4. Titik Anchor Tengah & Garis Panggilan AR (AR Callout Pin)
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(objCenterX, objCenterY, 5, 0, Math.PI * 2);
        ctx.fill();

        ctx.beginPath();
        ctx.arc(objCenterX, objCenterY, 11, 0, Math.PI * 2);
        ctx.strokeStyle = hexToRgba(color, 0.7);
        ctx.lineWidth = 1.5;
        ctx.stroke();

        // 5. Label Pill AR di bagian atas Bounding Box
        const badgeText = `${info.icon} ${info.label}  ${confidencePct}%`;
        ctx.font = '700 14px "Outfit", sans-serif';
        const textWidth = ctx.measureText(badgeText).width;
        const pillW = textWidth + 24;
        const pillH = 32;
        const pillX = Math.max(10, Math.min(x, canvas.width - pillW - 10));
        const pillY = Math.max(75, y - pillH - 8);

        // Garis penghubung dari titik tengah ke badge
        ctx.beginPath();
        ctx.moveTo(objCenterX, objCenterY);
        ctx.lineTo(pillX + pillW / 2, pillY + pillH);
        ctx.strokeStyle = hexToRgba(color, 0.5);
        ctx.lineWidth = 1.5;
        ctx.stroke();

        // Background Pill
        ctx.shadowBlur = 12;
        ctx.fillStyle = 'rgba(15, 23, 42, 0.88)';
        roundRect(ctx, pillX, pillY, pillW, pillH, 9);
        ctx.fill();

        ctx.lineWidth = 2;
        ctx.strokeStyle = color;
        ctx.stroke();

        // Teks Label
        ctx.shadowBlur = 0;
        ctx.fillStyle = '#ffffff';
        ctx.fillText(badgeText, pillX + 12, pillY + 21);

        // Estimasi jarak visual sederhana berdasarkan tinggi relatif di layar
        const estDistanceMeters = Math.max(0.3, (1.4 / Math.sqrt((w * h) / (canvas.width * canvas.height))) * 0.45).toFixed(1);
        const distLabel = `~${estDistanceMeters}m`;
        ctx.font = '400 11px "Space Mono", monospace';
        ctx.fillStyle = hexToRgba('#ffffff', 0.8);
        ctx.fillText(distLabel, x + 8, y + h - 8);

        ctx.restore();
    });

    // Update UI Statistik & Focus Card
    updateDashboard(tally, centerLockedObject);
}

/**
 * Memperbarui angka counter untuk semua 11 benda serta kartu fokus di tengah
 */
function updateDashboard(tally, centerLockedObject) {
    TARGET_CATEGORIES.forEach(cat => {
        if (countElements[cat]) {
            countElements[cat].textContent = tally[cat] || 0;
        }
        if (cardElements[cat]) {
            cardElements[cat].classList.toggle('detected', (tally[cat] || 0) > 0);
        }
    });

    if (centerLockedObject) {
        hudReticle.classList.add('locked');
        focusCard.classList.remove('hidden');
        focusCard.style.borderColor = centerLockedObject.color;
        focusIcon.textContent = centerLockedObject.icon;
        focusTitle.textContent = centerLockedObject.subLabel;
        focusDesc.textContent = centerLockedObject.desc;
        focusScore.textContent = `${Math.round(centerLockedObject.score * 100)}%`;

        // Suara Asisten AR jika diaktifkan
        speakObject(centerLockedObject.label);
    } else {
        hudReticle.classList.remove('locked');
        focusCard.classList.add('hidden');
    }
}

/**
 * Text-to-Speech Bahasa Indonesia untuk menyebutkan objek yang dikenali
 */
function speakObject(objectName) {
    if (!isVoiceEnabled || !('speechSynthesis' in window)) return;

    const now = Date.now();
    // Beri jeda agar suara tidak tumpang tindih
    if (objectName === lastSpokenObject && now - lastSpokenTime < 5000) return;
    if (now - lastSpokenTime < 3000) return;

    lastSpokenObject = objectName;
    lastSpokenTime = now;

    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(`Terdeteksi ${objectName.toLowerCase()}`);
    utterance.lang = 'id-ID';
    utterance.rate = 1.05;
    window.speechSynthesis.speak(utterance);
}

/**
 * Loop Deteksi Real-time dari Kamera
 */
async function detectFrame() {
    if (isStaticImageMode) return;

    if (video.readyState === 4 && model) {
        const vw = video.videoWidth;
        const vh = video.videoHeight;

        // Salin frame ke kanvas analisis kecil untuk pemeriksaan kecerahan/warna
        analysisCtx.drawImage(video, 0, 0, analysisCanvas.width, analysisCanvas.height);

        // Jalankan MobileNet secara periodik di latar belakang
        updateMobileNetClassification(video, vw, vh, false);

        // Deteksi objek menggunakan COCO-SSD (ambil kandidat hingga skor 0.20)
        const predictions = await model.detect(video, 20, 0.20);
        drawAROverlay(predictions, vw, vh);

        // Hitung FPS
        frameCount++;
        const now = performance.now();
        if (now - lastFrameTime >= 1000) {
            currentFps = frameCount;
            frameCount = 0;
            lastFrameTime = now;
            fpsCounter.textContent = `${String(currentFps).padStart(2, '0')} FPS | AR AKTIF`;
        }
    }

    requestAnimationFrame(detectFrame);
}

/**
 * Deteksi pada Foto yang Diunggah Pengguna (untuk pengujian tanpa kamera/ruangan)
 */
async function detectStaticImage() {
    if (!model || !uploadedImage.src) return;

    resizeCanvas();
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const imgW = uploadedImage.naturalWidth;
    const imgH = uploadedImage.naturalHeight;
    if (!imgW || !imgH) return;

    // Salin gambar ke kanvas analisis
    analysisCtx.drawImage(uploadedImage, 0, 0, analysisCanvas.width, analysisCanvas.height);

    // Jalankan klasifikasi MobileNet untuk gambar statis
    await updateMobileNetClassification(uploadedImage, imgW, imgH, true);

    // Gambar foto ke canvas dengan proporsi cover
    const scale = Math.max(canvas.width / imgW, canvas.height / imgH);
    const offsetX = (canvas.width - imgW * scale) / 2;
    const offsetY = (canvas.height - imgH * scale) / 2;

    const predictions = await model.detect(uploadedImage, 25, 0.20);
    drawAROverlay(predictions, imgW, imgH);

    // Gambar image sebagai background di bawah overlay AR
    ctx.save();
    ctx.globalCompositeOperation = 'destination-over';
    ctx.drawImage(uploadedImage, offsetX, offsetY, imgW * scale, imgH * scale);
    ctx.restore();
}

/**
 * Helper: Menggambar persegi panjang bersudut melengkung
 */
function roundRect(ctx, x, y, width, height, radius) {
    ctx.beginPath();
    ctx.moveTo(x + radius, y);
    ctx.lineTo(x + width - radius, y);
    ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
    ctx.lineTo(x + width, y + height - radius);
    ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
    ctx.lineTo(x + radius, y + height);
    ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
    ctx.lineTo(x, y + radius);
    ctx.quadraticCurveTo(x, y, x + radius, y);
    ctx.closePath();
}

function hexToRgba(hex, alpha) {
    const cleanHex = hex.replace('#', '');
    const r = parseInt(cleanHex.substring(0, 2), 16);
    const g = parseInt(cleanHex.substring(2, 4), 16);
    const b = parseInt(cleanHex.substring(4, 6), 16);
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

/**
 * Event Listeners Kontrol UI
 */
btnFilterMode.addEventListener('click', () => {
    isTargetOnly = !isTargetOnly;
    btnFilterMode.classList.toggle('active', isTargetOnly);
    btnFilterMode.innerHTML = isTargetOnly
        ? '<span>🎯 Target: 11 Benda</span>'
        : '<span>🌐 Semua Objek</span>';
    if (isStaticImageMode) detectStaticImage();
});

btnVoice.addEventListener('click', () => {
    isVoiceEnabled = !isVoiceEnabled;
    btnVoice.classList.toggle('active', isVoiceEnabled);
    btnVoice.innerHTML = isVoiceEnabled
        ? '<span>🔊 Suara Aktif</span>'
        : '<span>🔇 Suara</span>';
    if (isVoiceEnabled) {
        speakObject('Asisten suara aktif');
    }
});

btnSwitchCam.addEventListener('click', async () => {
    currentFacingMode = currentFacingMode === 'environment' ? 'user' : 'environment';
    isStaticImageMode = false;
    video.classList.remove('hidden');
    btnBackCam.classList.add('hidden');
    await setupCamera();
});

confidenceSlider.addEventListener('input', (e) => {
    const val = parseInt(e.target.value, 10);
    minConfidence = val / 100;
    confVal.textContent = `${val}%`;
    if (isStaticImageMode) detectStaticImage();
});

uploadInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
        uploadedImage.onload = () => {
            isStaticImageMode = true;
            video.classList.add('hidden');
            btnBackCam.classList.remove('hidden');
            fpsCounter.textContent = 'MODE FOTO | AR AKTIF';
            detectStaticImage();
        };
        uploadedImage.src = event.target.result;
    };
    reader.readAsDataURL(file);
});

btnBackCam.addEventListener('click', () => {
    isStaticImageMode = false;
    video.classList.remove('hidden');
    btnBackCam.classList.add('hidden');
    detectFrame();
});

btnSnapshot.addEventListener('click', () => {
    const snapCanvas = document.createElement('canvas');
    snapCanvas.width = canvas.width;
    snapCanvas.height = canvas.height;
    const snapCtx = snapCanvas.getContext('2d');

    if (!isStaticImageMode && video.readyState === 4) {
        const scale = Math.max(canvas.width / video.videoWidth, canvas.height / video.videoHeight);
        const offsetX = (canvas.width - video.videoWidth * scale) / 2;
        const offsetY = (canvas.height - video.videoHeight * scale) / 2;
        snapCtx.drawImage(video, offsetX, offsetY, video.videoWidth * scale, video.videoHeight * scale);
    }
    snapCtx.drawImage(canvas, 0, 0);

    const link = document.createElement('a');
    link.download = `ar-scanner-${Date.now()}.png`;
    link.href = snapCanvas.toDataURL('image/png');
    link.click();
});

/**
 * Inisialisasi Aplikasi Utama
 */
async function initApp() {
    try {
        progressFill.style.width = '30%';
        loadingStatus.textContent = 'Mengunduh model deteksi objek COCO-SSD...';

        model = await cocoSsd.load({ base: 'lite_mobilenet_v2' });

        progressFill.style.width = '60%';
        loadingStatus.textContent = 'Memuat model pengenalan benda tambahan (MobileNet)...';

        // Muat MobileNet untuk memperkaya pengenalan Speaker, Lampu, Pintu, PC, dll.
        if (window.mobilenet) {
            mobilenet.load({ version: 2, alpha: 0.5 })
                .then(m => { mobilenetModel = m; })
                .catch(err => console.warn('MobileNet opsional gagal dimuat:', err));
        }

        progressFill.style.width = '80%';
        loadingStatus.textContent = 'Mengaktifkan kamera perangkat untuk AR...';

        await setupCamera();

        progressFill.style.width = '100%';
        loadingStatus.textContent = 'Siap! Arahkan kamera ke objek di sekitar Anda.';

        setTimeout(() => {
            loadingOverlay.classList.add('fade-out');
            detectFrame();
        }, 450);
    } catch (error) {
        console.error('Initialization error:', error);
        loadingStatus.innerHTML = `⚠️ Kamera tidak ditemukan atau izin ditolak.<br><small>Anda tetap dapat mencoba fitur AR menggunakan tombol <b>Uji Foto</b> di bawah.</small>`;
        progressFill.style.width = '100%';
        setTimeout(() => {
            loadingOverlay.classList.add('fade-out');
        }, 2200);
    }
}

initApp();
