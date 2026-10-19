/**
 * AR Furniture Vision - Deteksi Objek Lemari, Meja, dan Kursi di Browser
 * Menggunakan TensorFlow.js + COCO-SSD dengan AR HUD Overlay
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

// Counters & Cards
const countKursi = document.getElementById('count-kursi');
const countMeja = document.getElementById('count-meja');
const countLemari = document.getElementById('count-lemari');
const cardKursi = document.getElementById('card-kursi');
const cardMeja = document.getElementById('card-meja');
const cardLemari = document.getElementById('card-lemari');

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
let isTargetOnly = true;     // Default: Fokus ke Kursi, Meja, Lemari
let isVoiceEnabled = false;
let minConfidence = 0.40;
let currentFacingMode = 'environment'; // Gunakan kamera belakang secara default di HP
let isStaticImageMode = false;
let stream = null;
let lastSpokenObject = '';
let lastSpokenTime = 0;

// FPS Calculation
let lastFrameTime = performance.now();
let frameCount = 0;
let currentFps = 0;

/**
 * Pemetaan kelas COCO-SSD + Heuristik Bentuk ke Kategori Target:
 * 1. KURSI (Chair, Couch/Sofa, Bench)
 * 2. MEJA (Dining Table, Desk)
 * 3. LEMARI (Cabinet, Wardrobe, Refrigerator/Tall Storage, Oven/Lower Cabinet, Bookcase)
 */
const FURNITURE_MAP = {
    // Kategori KURSI
    'chair': {
        category: 'kursi',
        label: 'KURSI',
        subLabel: 'Kursi (Chair)',
        icon: '🪑',
        color: '#10b981',
        desc: 'Objek tempat duduk terdeteksi di area AR.'
    },
    'couch': {
        category: 'kursi',
        label: 'KURSI / SOFA',
        subLabel: 'Sofa / Kursi Panjang',
        icon: '🛋️',
        color: '#10b981',
        desc: 'Sofa atau tempat duduk ruang tamu terdeteksi.'
    },
    'bench': {
        category: 'kursi',
        label: 'KURSI BANGKU',
        subLabel: 'Bangku Panjang (Bench)',
        icon: '🪑',
        color: '#10b981',
        desc: 'Bangku tempat duduk terdeteksi.'
    },

    // Kategori MEJA
    'dining table': {
        category: 'meja',
        label: 'MEJA',
        subLabel: 'Meja (Table / Desk)',
        icon: '🪵',
        color: '#f59e0b',
        desc: 'Permukaan meja kerja atau meja makan terdeteksi.'
    },

    // Kategori LEMARI
    'refrigerator': {
        category: 'lemari',
        label: 'LEMARI',
        subLabel: 'Lemari / Kabinet Tinggi',
        icon: '🗄️',
        color: '#8b5cf6',
        desc: 'Struktur lemari tegak / penyimpanan besar terdeteksi.'
    },
    'oven': {
        category: 'lemari',
        label: 'LEMARI KABINET',
        subLabel: 'Kabinet / Lemari Bawah',
        icon: '🗄️',
        color: '#8b5cf6',
        desc: 'Unit kabinet atau lemari penyimpanan bawah terdeteksi.'
    },
    'microwave': {
        category: 'lemari',
        label: 'LEMARI RAK',
        subLabel: 'Kompartemen / Kabinet',
        icon: '🗄️',
        color: '#8b5cf6',
        desc: 'Unit kompartemen rak/lemari terdeteksi.'
    },
    'book': {
        // Sering terdeteksi berderet pada rak buku/lemari buku
        category: 'lemari',
        label: 'RAK / LEMARI BUKU',
        subLabel: 'Bagian Lemari Rak Buku',
        icon: '📚',
        color: '#8b5cf6',
        desc: 'Area rak atau lemari buku terdeteksi.'
    }
};

/**
 * Klasifikasi cerdas: Mengubah hasil prediksi mentah menjadi informasi AR Furnitur
 * Juga mendukung heuristik rasio ukuran (aspect ratio) untuk membantu mengenali lemari/meja
 */
function classifyObject(prediction, canvasWidth, canvasHeight) {
    const rawClass = prediction.class.toLowerCase();
    const [x, y, width, height] = prediction.bbox;
    const aspectRatio = height / Math.max(width, 1);
    const relativeArea = (width * height) / (canvasWidth * canvasHeight);

    // Jika langsung cocok di FURNITURE_MAP
    if (FURNITURE_MAP[rawClass]) {
        // Khusus 'dining table' yang sangat tinggi (vertikal), bisa jadi bagian kabinet/lemari
        if (rawClass === 'dining table' && aspectRatio > 1.55) {
            return {
                category: 'lemari',
                label: 'LEMARI',
                subLabel: 'Lemari / Rak Vertikal',
                icon: '🗄️',
                color: '#8b5cf6',
                desc: 'Furnitur penyimpanan vertikal (lemari) terdeteksi.',
                isTarget: true
            };
        }
        return {
            ...FURNITURE_MAP[rawClass],
            isTarget: true
        };
    }

    // Heuristik tambahan untuk objek furnitur besar di ruangan (misal bed/tv/suitcase besar yang menyerupai kabinet/meja)
    if (rawClass === 'bed' && aspectRatio < 0.85) {
        return {
            category: 'meja',
            label: 'MEJA / BIDANG DATAR',
            subLabel: 'Permukaan Furnitur Datar',
            icon: '🪵',
            color: '#f59e0b',
            desc: 'Permukaan furnitur datar terdeteksi.',
            isTarget: true
        };
    }

    if ((rawClass === 'suitcase' || rawClass === 'handbag' || rawClass === 'tv') && relativeArea > 0.12 && aspectRatio > 1.2) {
        return {
            category: 'lemari',
            label: 'LEMARI / KABINET',
            subLabel: 'Unit Lemari Penyimpanan',
            icon: '🗄️',
            color: '#8b5cf6',
            desc: 'Bentuk kotak penyimpanan vertikal (lemari) terdeteksi.',
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
function drawAROverlay(predictions, sourceWidth, sourceHeight) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Hitung skala object-fit: cover agar bounding box pas dengan layar
    const scale = Math.max(canvas.width / sourceWidth, canvas.height / sourceHeight);
    const offsetX = (canvas.width - sourceWidth * scale) / 2;
    const offsetY = (canvas.height - sourceHeight * scale) / 2;

    let tally = { kursi: 0, meja: 0, lemari: 0 };
    let centerLockedObject = null;
    let minCenterDist = Infinity;

    const screenCenterX = canvas.width / 2;
    const screenCenterY = canvas.height / 2;

    predictions.forEach(pred => {
        if (pred.score < minConfidence) return;

        const info = classifyObject(pred, sourceWidth, sourceHeight);
        if (isTargetOnly && !info.isTarget) return;

        // Tambah hitungan jika termasuk kursi, meja, atau lemari
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
        if (distToCenter < minCenterDist && distToCenter < Math.min(canvas.width, canvas.height) * 0.38) {
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
        const estDistanceMeters = Math.max(0.5, (1.4 / Math.sqrt((w * h) / (canvas.width * canvas.height))) * 0.45).toFixed(1);
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
 * Memperbarui angka counter Kursi, Meja, Lemari serta kartu fokus di tengah
 */
function updateDashboard(tally, centerLockedObject) {
    countKursi.textContent = tally.kursi;
    countMeja.textContent = tally.meja;
    countLemari.textContent = tally.lemari;

    cardKursi.classList.toggle('detected', tally.kursi > 0);
    cardMeja.classList.toggle('detected', tally.meja > 0);
    cardLemari.classList.toggle('detected', tally.lemari > 0);

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
    // Beri jeda 3.5 detik agar suara tidak tumpang tindih
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
        const predictions = await model.detect(video);
        drawAROverlay(predictions, video.videoWidth, video.videoHeight);

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

    // Gambar foto ke canvas dengan proporsi cover
    const scale = Math.max(canvas.width / imgW, canvas.height / imgH);
    const offsetX = (canvas.width - imgW * scale) / 2;
    const offsetY = (canvas.height - imgH * scale) / 2;

    const predictions = await model.detect(uploadedImage);
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
        ? '<span>🎯 Target: Furnitur</span>'
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
    link.download = `ar-furniture-${Date.now()}.png`;
    link.href = snapCanvas.toDataURL('image/png');
    link.click();
});

/**
 * Inisialisasi Aplikasi Utama
 */
async function initApp() {
    try {
        progressFill.style.width = '35%';
        loadingStatus.textContent = 'Mengunduh model pengenalan objek COCO-SSD...';

        model = await cocoSsd.load({ base: 'lite_mobilenet_v2' });

        progressFill.style.width = '75%';
        loadingStatus.textContent = 'Mengaktifkan kamera perangkat untuk AR...';

        await setupCamera();

        progressFill.style.width = '100%';
        loadingStatus.textContent = 'Siap! Arahkan kamera ke Kursi, Meja, atau Lemari.';

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

