// Mobile Vision Assistant - Main JavaScript
const API_BASE = '';

function getAccessToken() {
    return localStorage.getItem('access_token') || '';
}

async function apiFetch(url, options = {}) {
    if (sessionExpiredRedirecting) return null;
    const token = localStorage.getItem("access_token");
    const refreshToken = localStorage.getItem("refresh_token");

    const headers = {
        ...(options.headers || {})
    };

    if (token) {
        headers["Authorization"] = `Bearer ${token}`;
    }
    if (refreshToken) {
        headers["X-Refresh-Token"] = refreshToken;
    }

    if (options.body && !headers["Content-Type"]) {
        headers["Content-Type"] = "application/json";
    }

    const response = await fetch(url, {
        ...options,
        headers: {
            ...headers,
            'Authorization': token ? `Bearer ${token}` : ''
        }
    });

    const newAccessToken = response.headers.get("X-Access-Token");
    const newRefreshToken = response.headers.get("X-Refresh-Token");
    if (newAccessToken) {
        localStorage.setItem("access_token", newAccessToken);
    }
    if (newRefreshToken) {
        localStorage.setItem("refresh_token", newRefreshToken);
    }

    if (response.status === 401) {
        // Any 401 means we need to re-authenticate
        await handleSessionExpired();
        return null;
    }

    return response;
}

function resetNavigationUI() {
    const navBtn = document.getElementById('navigateBtn');
    const stopBtn = document.getElementById('stopNavBtn');
    if (navBtn) navBtn.style.display = 'block';
    if (stopBtn) stopBtn.style.display = 'none';
    document.body.classList.remove('navigation-active');
    navigationActive = false;
}

async function handleSessionExpired() {
    if (sessionExpiredRedirecting) return;
    sessionExpiredRedirecting = true;

    try {
        if (typeof stopFrameProcessing === 'function') stopFrameProcessing();
    } catch (e) { }
    try {
        if (typeof stopCamera === 'function') stopCamera();
    } catch (e) { }
    try {
        if (typeof stopVoice === 'function') stopVoice();
    } catch (e) { }
    try {
        stopNavigationLocationTracking();
    } catch (e) { }
    try {
        resetNavigationUI();
    } catch (e) { }

    localStorage.clear();

    if (!speechCancelSent) {
        speechCancelSent = true;
        fetch('/api/speech/cancel', { method: 'POST' }).catch(() => { });
    }

    if (!window.location.pathname.includes('/login')) {
        window.location.href = '/login';
    }
}

function reportApiError(message) {
    logToUI(message);
    if (typeof showToast === 'function') {
        showToast(message, 'error');
        return;
    }
    alert(message);
}

function logToUI(message) {
    if (!message) return;
    const logContainer = document.getElementById('logs');
    if (!logContainer) return;

    const entry = document.createElement('div');
    entry.textContent = message;
    logContainer.prepend(entry);

    while (logContainer.children.length > 20) {
        logContainer.removeChild(logContainer.lastChild);
    }
}

async function handleResponseSpeech(data, priority = 40, source = 'web') {
    if (!data || !data.speak) return;
    logToUI('Speaking response...');
    await speak(data.speak, priority, source);
}

async function apiFetchJson(url, options = {}, defaultMessage = 'Request failed') {
    const response = await apiFetch(url, options);
    if (!response) return null;

    if (!response.ok) {
        let message = defaultMessage;
        try {
            const err = await response.json();
            message = err.message || err.error || message;
        } catch (e) {
            // Ignore JSON parse errors
        }
        reportApiError(message);
        return null;
    }

    try {
        return await response.json();
    } catch (e) {
        reportApiError('Invalid server response');
        return null;
    }
}

function normalizeSessionId(sessionId) {
    if (sessionId === null || sessionId === undefined) return null;
    const normalized = String(sessionId).trim();
    if (!normalized || normalized === 'null' || normalized === 'undefined') {
        return null;
    }
    return normalized;
}

// Global state
let stream = null;
let videoElement = null;
let canvasElement = null;
let isProcessing = false;
let voiceRecognition = null;
let isListening = false;
let processingInterval = null;
let map = null;
let directionsService = null;
let directionsRenderer = null;
let navigationPolyline = null;
let navigationPolylineEncoded = null;
let navigationStartMarker = null;
let navigationEndMarker = null;
let historyPolyline = null;
let startMarker = null;
let endMarker = null;
let userMarker = null;
let lastKnownUserLocation = null;
let navigationRouteDestination = null;
let navigationRouteMode = null;
let navigationRouteLastUpdate = 0;
let navigationSessionId = null;
let navigationStatusIntervalId = null;
let locationWatchId = null;
let sendingFrame = false;
let sending = false;
let frameUploadInFlight = false;
let lastFrameSentAt = 0;
const FRAME_SEND_INTERVAL_MS = 600;
let frameLoopActive = false;
let lastDetectionLog = '';
let lastDetectionLogAt = 0;
const DETECTION_LOG_THROTTLE_MS = 3000;

// Global speech state tracking
let isSpeaking = false;
let isBackendSpeechActive = false;
let activeSpeechSource = null;
let locationSending = false;
let navigationActive = false;
let lastBackendSpeechActiveAt = 0;
let backendVoiceListening = false;
let sessionExpiredRedirecting = false;
let speechCancelSent = false;
let lastCommandId = null;
let lastSentCommandId = null;

const EMERGENCY_KEYWORDS = ['help', 'save me', 'emergency', 'bachao', 'madad'];
const NAVIGATION_CONTROL_PHRASES = [
    'stop navigation',
    'stop navigating',
    'cancel navigation',
    'cancel route',
    'end navigation',
    'stop route',
    'stop',
    'cancel'
];

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Check for HTTPS requirement
    if (location.protocol !== 'https:' &&
        location.hostname !== 'localhost' &&
        location.hostname !== '127.0.0.1') {
        console.warn('Camera access may require HTTPS on mobile devices');
        console.warn('For network access, use http://localhost:5000 or set up HTTPS');
    }
    initializeApp();
});

function initializeApp() {
    if (!getAccessToken()) {
        window.location.href = '/login';
        return;
    }
    videoElement = document.getElementById('videoElement');
    canvasElement = document.getElementById('canvasElement');

    // Setup event listeners
    setupEventListeners();

    // Check for Web Speech API support
    checkVoiceSupport();

    // Update status periodically (throttled)
    setInterval(fetchStatus, 3000);
    setInterval(fetchSpeechStatus, 2000);
    fetchStatus();

    // Load emergency contact if available
    loadEmergencyContact();

    // Set up volume button double-press detection (best-effort in browsers)
    setupEmergencyButtonDetection();

    // Map + analytics
    if (window.google && window.google.maps) {
        initMap();
    }

    // AUTO-START FOR BLIND USERS: Start camera and voice automatically
    setTimeout(() => {
        speak("Vision Assistant is ready. Starting camera and voice recognition.");
        startCamera();
        setTimeout(() => {
            if (!backendVoiceListening) {
                startVoice(); // Start browser voice only when backend STT is not active
            }
            speak("I am listening. You can say: describe scene, navigate, get location, or stop navigation.", 60, 'confirmation');
        }, 2000);
    }, 1000);

    // Poll for server-side voice prompts (navigation flow, etc.)
    startPendingSpeechPoll();

    // AUTO-SEND GPS so backend has location for "find nearest X" voice commands
    captureAndSendLocation({ announce: false }).catch(() => {});
}

// Background poll for server voice prompts (so phone hears them even without camera)
let _speechPollTimer = null;
let _lastKnownNavStatus = false;

async function pollPendingSpeech() {
    try {
        const data = await apiFetchJson(`${API_BASE}/api/speech/pending`, {}, '');
        if (data && data.pending_speech && data.pending_speech.length > 0) {
            for (const text of data.pending_speech) {
                if (text && String(text).trim()) {
                    logToUI('Voice: ' + text);
                    await speak(text, 60, 'voice_engine');
                }
            }
        }
        
        // Also sync navigation state from backend to frontend
        // If voice command started navigation on server, mobile UI needs to show it
        const statusData = await apiFetchJson(`${API_BASE}/api/status`, {}, '');
        if (statusData && statusData.success && statusData.status) {
            const isNavigatingNow = statusData.status.navigation_active;
            
            // Server started navigation but client UI doesn't know yet
            if (isNavigatingNow && !navigationActive) {
                logToUI('Backend navigation detected, syncing UI...');
                document.getElementById('navigateBtn').style.display = 'none';
                document.getElementById('stopNavBtn').style.display = 'block';
                document.body.classList.add('navigation-active');
                
                navigationActive = true;
                if (statusData.status.navigation_session_id) {
                    navigationSessionId = normalizeSessionId(statusData.status.navigation_session_id);
                }
                
                startNavigationLocationTracking();
                navigationRouteLastUpdate = 0;
                await waitForNavigationPolyline();
            } 
            // Server stopped navigation but client UI is still stuck
            else if (!isNavigatingNow && navigationActive) {
                logToUI('Backend navigation stopped, resetting UI...');
                document.getElementById('navigateBtn').style.display = 'block';
                document.getElementById('stopNavBtn').style.display = 'none';
                document.body.classList.remove('navigation-active');
                
                navigationActive = false;
                navigationSessionId = null;
                stopNavigationLocationTracking();
                if (directionsRenderer) {
                    directionsRenderer.set('directions', null);
                }
                clearNavigationRoute();
            }
        }
        
    } catch (e) {
        // Silent fail
    }
}
function startPendingSpeechPoll() {
    if (_speechPollTimer) return;
    _speechPollTimer = setInterval(pollPendingSpeech, 2000);
}

function setupEventListeners() {
    // Camera controls
    document.getElementById('startCameraBtn').addEventListener('click', startCamera);
    document.getElementById('stopCameraBtn').addEventListener('click', stopCamera);

    // Feature buttons
    document.getElementById('describeBtn').addEventListener('click', describeScene);
    document.getElementById('navigateBtn').addEventListener('click', openNavigationModal);
    document.getElementById('locationBtn').addEventListener('click', getLocation);
    document.getElementById('stopNavBtn').addEventListener('click', stopNavigation);

    // Personal objects button
    document.getElementById('addObjectsBtn').addEventListener('click', openAddObjectsModal);

    // Video resize buttons
    document.getElementById('resizeSmallBtn').addEventListener('click', () => resizeVideo('small'));
    document.getElementById('resizeMediumBtn').addEventListener('click', () => resizeVideo('medium'));
    document.getElementById('resizeLargeBtn').addEventListener('click', () => resizeVideo('large'));
    document.getElementById('resizeFullscreenBtn').addEventListener('click', () => resizeVideo('fullscreen'));

    // Navigation modal
    document.getElementById('closeNavModal').addEventListener('click', closeNavigationModal);
    document.getElementById('startNavigationBtn').addEventListener('click', startNavigation);

    // Add objects modal
    document.getElementById('closeObjectsModal').addEventListener('click', closeAddObjectsModal);
    document.getElementById('saveObjectBtn').addEventListener('click', () => {
        const name = document.getElementById('objectNameInput')?.value || '';
        const imageUrl = getCurrentObjectImageUrl();
        savePersonalObject(name, imageUrl);
    });
    document.getElementById('captureCurrentFrameBtn').addEventListener('click', captureCurrentFrameForObject);
    const objectNameInput = document.getElementById('objectNameInput');
    if (objectNameInput) {
        objectNameInput.addEventListener('input', updateSaveObjectButtonState);
    }
    const objectImageUpload = document.getElementById('objectImageUpload');
    if (objectImageUpload) {
        objectImageUpload.addEventListener('change', handleObjectImageUpload);
    }

    // Voice controls
    document.getElementById('voiceBtn').addEventListener('click', toggleVoice);

    // Feedback
    document.getElementById('submitFeedbackBtn').addEventListener('click', submitFeedback);

    // Emergency contact
    const saveEmergencyBtn = document.getElementById('saveEmergencyPhoneBtn');
    if (saveEmergencyBtn) {
        saveEmergencyBtn.addEventListener('click', saveEmergencyContact);
    }

    const sessionSelect = document.getElementById('sessionSelect');
    if (sessionSelect) {
        sessionSelect.addEventListener('change', () => {
            const sessionId = sessionSelect.value;
            if (sessionId) {
                loadLocationSummary(sessionId);
                loadLocationPolyline(sessionId);
            }
        });
    }

    // Logout
    document.getElementById('logoutBtn').addEventListener('click', logout);

    // Close modals on outside click
    document.getElementById('navigationModal').addEventListener('click', (e) => {
        if (e.target.id === 'navigationModal') {
            closeNavigationModal();
        }
    });

    document.getElementById('addObjectsModal').addEventListener('click', (e) => {
        if (e.target.id === 'addObjectsModal') {
            closeAddObjectsModal();
        }
    });
}

function initMap() {
    const mapEl = document.getElementById('routeMap');
    if (!mapEl || !window.google || !window.google.maps) return;
    map = new google.maps.Map(mapEl, {
        center: { lat: 20.5937, lng: 78.9629 },
        zoom: 4,
        mapTypeControl: false,
        streetViewControl: false,
        fullscreenControl: false
    });

    directionsService = new google.maps.DirectionsService();
    directionsRenderer = new google.maps.DirectionsRenderer({
        suppressMarkers: false,
        polylineOptions: {
            strokeColor: '#1A73E8',
            strokeWeight: 6
        }
    });
    directionsRenderer.setMap(map);

    historyPolyline = new google.maps.Polyline({
        path: [],
        strokeColor: '#1f8b4c',
        strokeWeight: 4,
        strokeOpacity: 0.9,
        map
    });

    loadLocationSessions();
}

async function loadLocationSessions() {
    const select = document.getElementById('sessionSelect');
    if (!select) return;
    select.innerHTML = '';
    try {
        const data = await apiFetchJson(`${API_BASE}/api/location_sessions?limit=20`, {}, 'Failed to load sessions');
        if (!data) return;
        if (!data.success || !data.sessions || data.sessions.length === 0) {
            const opt = document.createElement('option');
            opt.value = '';
            opt.textContent = 'No sessions yet';
            select.appendChild(opt);
            updateStats(null);
            return;
        }
        data.sessions.forEach((session, idx) => {
            const opt = document.createElement('option');
            opt.value = session.id;
            const started = session.started_at ? new Date(session.started_at).toLocaleString() : `Session ${idx + 1}`;
            opt.textContent = started;
            select.appendChild(opt);
        });
        const firstId = data.sessions[0].id;
        select.value = firstId;
        await loadLocationSummary(firstId);
        await loadLocationPolyline(firstId);
    } catch (e) {
        updateStats(null);
    }
}

async function loadLocationSummary(sessionId) {
    try {
        const data = await apiFetchJson(`${API_BASE}/api/location_summary?session_id=${encodeURIComponent(sessionId)}`, {}, 'Failed to load session summary');
        if (!data) return;
        if (data.success && data.session) {
            updateStats(data.session);
        } else {
            updateStats(null);
        }
    } catch (e) {
        updateStats(null);
    }
}

async function loadLocationPolyline(sessionId) {
    if (!map || !historyPolyline) return;
    const validSessionId = normalizeSessionId(sessionId);
    if (!validSessionId) {
        historyPolyline.setPath([]);
        if (startMarker) {
            startMarker.setMap(null);
            startMarker = null;
        }
        if (endMarker) {
            endMarker.setMap(null);
            endMarker = null;
        }
        return;
    }
    try {
        const data = await apiFetchJson(`${API_BASE}/api/location_polyline?session_id=${encodeURIComponent(validSessionId)}`, {}, 'Failed to load session route');
        if (!data) return;
        if (!data || !data.polyline) {
            historyPolyline.setPath([]);
            return;
        }
        const points = decodePolyline(data.polyline).map(([lat, lng]) => ({ lat, lng }));
        historyPolyline.setPath(points);
        if (points.length > 1) {
            if (startMarker) startMarker.setMap(null);
            if (endMarker) endMarker.setMap(null);
            startMarker = new google.maps.Marker({
                position: points[0],
                map,
                title: 'Start'
            });
            endMarker = new google.maps.Marker({
                position: points[points.length - 1],
                map,
                title: 'End'
            });
            const bounds = new google.maps.LatLngBounds();
            points.forEach(p => bounds.extend(p));
            map.fitBounds(bounds, { top: 20, right: 20, bottom: 20, left: 20 });
        }
    } catch (e) {
        historyPolyline.setPath([]);
    }
}

function updateStats(session) {
    const distanceEl = document.getElementById('statDistance');
    const timeEl = document.getElementById('statTime');
    const speedEl = document.getElementById('statSpeed');
    const stopsEl = document.getElementById('statStops');
    const startEndEl = document.getElementById('statStartEnd');

    if (!session) {
        if (distanceEl) distanceEl.textContent = '0 m';
        if (timeEl) timeEl.textContent = '0 min';
        if (speedEl) speedEl.textContent = '0 m/s';
        if (stopsEl) stopsEl.textContent = '0';
        if (startEndEl) startEndEl.textContent = '--';
        return;
    }
    const distance = session.total_distance_m || 0;
    const timeS = session.total_time_s || 0;
    const speed = session.avg_speed_mps || 0;
    const start = session.start_lat && session.start_lng ? `${session.start_lat.toFixed(4)}, ${session.start_lng.toFixed(4)}` : '--';
    const end = session.end_lat && session.end_lng ? `${session.end_lat.toFixed(4)}, ${session.end_lng.toFixed(4)}` : '--';

    if (distanceEl) distanceEl.textContent = distance >= 1000 ? `${(distance / 1000).toFixed(2)} km` : `${Math.round(distance)} m`;
    if (timeEl) timeEl.textContent = `${Math.round(timeS / 60)} min`;
    if (speedEl) speedEl.textContent = `${speed.toFixed(2)} m/s`;
    if (stopsEl) stopsEl.textContent = `${session.stops_count || 0}`;
    if (startEndEl) startEndEl.textContent = `${start} → ${end}`;
}

// Emergency contact handling
async function loadEmergencyContact() {
    try {
        const data = await apiFetchJson(`${API_BASE}/api/emergency_contact`, {}, 'Failed to load emergency contact');
        if (!data) return;
        if (data.success && data.emergency_phone) {
            const input = document.getElementById('emergencyPhoneInput');
            if (input) input.value = data.emergency_phone;
        }
    } catch (e) {
        // Silent fail
    }
}

async function saveEmergencyContact() {
    const input = document.getElementById('emergencyPhoneInput');
    if (!input) return;
    const emergency_phone = input.value.trim();
    if (!emergency_phone) {
        logToUI('Emergency contact is missing.');
        showToast('Please enter an emergency phone number', 'error');
        return;
    }
    try {
        logToUI('Saving emergency contact...');
        const data = await apiFetchJson(`${API_BASE}/api/emergency_contact`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ emergency_phone })
        }, 'Failed to save emergency contact');
        if (!data) return;
        if (data.success) {
            logToUI(`Emergency contact saved: ${data.emergency_phone || emergency_phone}`);
            await handleResponseSpeech(data, 60, 'emergency_contact');
            showToast('Emergency contact saved');
        } else {
            showToast(data.message || 'Failed to save emergency contact', 'error');
        }
    } catch (e) {
        showToast('Error saving emergency contact', 'error');
    }
}

// Emergency trigger (manual/keyboard)
let _emergencyInFlight = false;
let _emergencyLastFiredAt = 0;
const EMERGENCY_CLIENT_COOLDOWN_MS = 10000; // 10 seconds client-side guard

async function triggerEmergency(triggerType = 'button') {
    const now = Date.now();
    if (_emergencyInFlight || (now - _emergencyLastFiredAt) < EMERGENCY_CLIENT_COOLDOWN_MS) {
        console.warn('Emergency trigger ignored – already in flight or on cooldown');
        return;
    }
    _emergencyInFlight = true;
    _emergencyLastFiredAt = now;
    try {
        logToUI('Sending emergency alert...');
        const data = await apiFetchJson(`${API_BASE}/api/emergency_trigger`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ trigger_type: triggerType })
        }, 'Emergency alert failed');
        if (!data) return;
        if (data.success) {
            logToUI(data.message || 'Emergency alert sent');
            await handleResponseSpeech(data, 100, 'emergency');
            showToast('Emergency alert sent', 'success');
        } else {
            showToast(data.message || 'Emergency alert failed', 'error');
        }
    } catch (e) {
        showToast('Emergency alert failed', 'error');
    } finally {
        _emergencyInFlight = false;
    }
}

function setupEmergencyButtonDetection() {
    let lastPressTime = 0;
    let pressCount = 0;
    const maxInterval = 1500; // ms

    window.addEventListener('keydown', (e) => {
        if (e.code === 'AudioVolumeUp' || e.code === 'AudioVolumeDown') {
            const now = Date.now();
            if (now - lastPressTime <= maxInterval) {
                pressCount += 1;
            } else {
                pressCount = 1;
            }
            lastPressTime = now;

            if (pressCount >= 2) {
                pressCount = 0;
                triggerEmergency('button');
            }
        }
    });
}

function decodePolyline(str) {
    let index = 0, lat = 0, lng = 0, coordinates = [];
    while (index < str.length) {
        let b, shift = 0, result = 0;
        do {
            b = str.charCodeAt(index++) - 63;
            result |= (b & 0x1f) << shift;
            shift += 5;
        } while (b >= 0x20);
        const dlat = (result & 1) ? ~(result >> 1) : (result >> 1);
        lat += dlat;

        shift = 0;
        result = 0;
        do {
            b = str.charCodeAt(index++) - 63;
            result |= (b & 0x1f) << shift;
            shift += 5;
        } while (b >= 0x20);
        const dlng = (result & 1) ? ~(result >> 1) : (result >> 1);
        lng += dlng;

        coordinates.push([lat / 1e5, lng / 1e5]);
    }
    return coordinates;
}

// Camera Functions
async function startCamera() {
    try {
        logToUI('Starting camera...');
        // Check for mediaDevices support
        if (!navigator.mediaDevices) {
            // Try to polyfill for older browsers
            navigator.mediaDevices = {};
        }

        // Get getUserMedia with fallbacks
        let getUserMedia = navigator.mediaDevices.getUserMedia ||
            navigator.getUserMedia ||
            navigator.webkitGetUserMedia ||
            navigator.mozGetUserMedia ||
            navigator.msGetUserMedia;

        if (!getUserMedia) {
            // Check if HTTPS is required
            const isSecureContext = window.isSecureContext ||
                location.protocol === 'https:' ||
                location.hostname === 'localhost' ||
                location.hostname === '127.0.0.1';

            if (!isSecureContext) {
                throw new Error('Camera requires HTTPS. Please access via https:// or use localhost. Current: ' + location.protocol);
            } else {
                throw new Error('Camera API not supported in this browser. Please use Chrome, Safari, or Firefox.');
            }
        }

        // Wrap getUserMedia if it's not a promise-based API
        if (getUserMedia === navigator.getUserMedia ||
            getUserMedia === navigator.webkitGetUserMedia ||
            getUserMedia === navigator.mozGetUserMedia) {
            getUserMedia = function (constraints) {
                return new Promise((resolve, reject) => {
                    getUserMedia.call(navigator, constraints, resolve, reject);
                });
            };
        }

        // Request camera access with proper constraints
        const constraints = {
            video: {
                facingMode: 'environment', // Use back camera on mobile
                width: { ideal: 1280 },
                height: { ideal: 720 }
            },
            audio: false
        };

        stream = await getUserMedia.call(navigator.mediaDevices || navigator, constraints);

        videoElement.srcObject = stream;

        // Wait for video to be ready
        await new Promise((resolve, reject) => {
            videoElement.onloadedmetadata = () => {
                videoElement.play().then(resolve).catch(reject);
            };
            videoElement.onerror = reject;
            // Timeout after 5 seconds
            setTimeout(() => reject(new Error('Video loading timeout')), 5000);
        });

        // Hide overlay
        document.getElementById('cameraOverlay').style.display = 'none';

        // Update UI
        document.getElementById('startCameraBtn').disabled = true;
        document.getElementById('stopCameraBtn').disabled = false;

        // Start processing frames
        startFrameProcessing();

        logToUI('Camera started.');
        showToast('Camera started successfully');
        updateStatus();
    } catch (error) {
        console.error('Error accessing camera:', error);
        let errorMsg = 'Failed to access camera: ';
        if (error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError') {
            errorMsg += 'Permission denied. Please allow camera access in browser settings.';
        } else if (error.name === 'NotFoundError' || error.name === 'DevicesNotFoundError') {
            errorMsg += 'No camera found on this device.';
        } else if (error.name === 'NotReadableError' || error.name === 'TrackStartError') {
            errorMsg += 'Camera is already in use by another application.';
        } else if (error.message && error.message.includes('HTTPS')) {
            errorMsg = error.message;
        } else {
            errorMsg += error.message || 'Unknown error. Check browser console for details.';
        }
        logToUI(errorMsg);
        showToast(errorMsg, 'error');
        document.getElementById('cameraOverlay').style.display = 'flex';
        document.getElementById('cameraOverlay').innerHTML = `<p>${errorMsg}</p>`;
    }
}

function stopCamera() {
    logToUI('Stopping camera...');
    // Stop stream
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }

    videoElement.srcObject = null;

    // Show overlay
    document.getElementById('cameraOverlay').style.display = 'flex';

    // Update UI
    document.getElementById('startCameraBtn').disabled = false;
    document.getElementById('stopCameraBtn').disabled = true;

    // Stop processing
    stopFrameProcessing();

    logToUI('Camera stopped.');
    showToast('Camera stopped');
    updateStatus();
}

// Continuous scene description
let lastSceneDescriptionTime = 0;
const SCENE_DESCRIPTION_INTERVAL = 10000; // Describe scene every 10 seconds

function startFrameProcessing() {
    if (frameLoopActive) return;
    frameLoopActive = true;
    sendFrameLoop();
}

function stopFrameProcessing() {
    frameLoopActive = false;
    if (processingInterval) {
        clearInterval(processingInterval);
        processingInterval = null;
    }
}

async function isNavigationSpeechActive() {
    try {
        const status = await getSpeechStatus();
        const priority = status?.current_speech?.priority;
        return priority === 'NAVIGATION';
    } catch (e) {
        return false;
    }
}

async function sendFrame() {
    if (!videoElement || !stream) return;
    if (sendingFrame || frameUploadInFlight || isProcessing) return;
    if (videoElement.readyState !== videoElement.HAVE_ENOUGH_DATA) return;
    const now = Date.now();
    if (now - lastFrameSentAt < FRAME_SEND_INTERVAL_MS) return;
    if (await isNavigationSpeechActive()) return;
    sendingFrame = true;
    lastFrameSentAt = now;
    try {
        await processFrame();
        if (now - lastSceneDescriptionTime > SCENE_DESCRIPTION_INTERVAL) {
            lastSceneDescriptionTime = now;
            await describeSceneContinuous();
        }
    } finally {
        sendingFrame = false;
    }
}

async function sendFrameLoop() {
    if (!frameLoopActive) return;
    if (sending) return;
    sending = true;
    try {
        await sendFrame();
    } finally {
        sending = false;
        setTimeout(sendFrameLoop, FRAME_SEND_INTERVAL_MS);
    }
}

async function processFrame() {
    if (!videoElement || !stream) return;

    isProcessing = true;
    frameUploadInFlight = true;

    try {
        // Capture frame to canvas
        const context = canvasElement.getContext('2d');
        canvasElement.width = videoElement.videoWidth;
        canvasElement.height = videoElement.videoHeight;
        context.drawImage(videoElement, 0, 0);

        // Convert to base64
        const imageData = canvasElement.toDataURL('image/jpeg', 0.8);

        // Send to server for processing
        const data = await apiFetchJson(`${API_BASE}/api/process_frame`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                image: imageData,
                get_description: false
            })
        }, 'Failed to process frame');
        if (!data) return;

        if (data.success && data.detections) {
            displayDetections(data.detections);
            if (data.message && data.message !== lastDetectionLog) {
                const now = Date.now();
                if ((now - lastDetectionLogAt) >= DETECTION_LOG_THROTTLE_MS) {
                    logToUI(data.message);
                    lastDetectionLog = data.message;
                    lastDetectionLogAt = now;
                }
            }
            if (data.speak) {
                await handleResponseSpeech(data, 40, 'vision');
            }
        }
    } catch (error) {
        console.error('Error processing frame:', error);
    } finally {
        isProcessing = false;
        frameUploadInFlight = false;
    }
}

// Continuous scene description (non-blocking)
async function describeSceneContinuous() {
    if (!stream || isProcessing) return;
    if (frameUploadInFlight) return;

    try {
        frameUploadInFlight = true;
        // Skip scene description during navigation to avoid interference
        const inNavigationMode = document.body.classList.contains('navigation-active');
        if (inNavigationMode) {
            return;
        }
        if (await isNavigationSpeechActive()) {
            return;
        }

        const context = canvasElement.getContext('2d');
        canvasElement.width = videoElement.videoWidth;
        canvasElement.height = videoElement.videoHeight;
        context.drawImage(videoElement, 0, 0);
        const imageData = canvasElement.toDataURL('image/jpeg', 0.8);

        const data = await apiFetchJson(`${API_BASE}/api/process_frame`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                image: imageData,
                get_description: true
            })
        }, 'Failed to describe scene');
        if (!data) return;

        if (data.success && data.description) {
            await handleResponseSpeech(data, 20, 'scene');
        }
    } catch (error) {
        // Silently fail for continuous description
        console.error('Error in continuous scene description:', error);
    } finally {
        frameUploadInFlight = false;
    }
}

function displayDetections(detections) {
    const display = document.getElementById('detectionsDisplay');

    if (detections.length === 0) {
        display.textContent = 'No objects detected';
        return;
    }

    // During navigation, show more detailed information
    const isInNavigationMode = document.body.classList.contains('navigation-active');

    if (isInNavigationMode) {
        // Show high-confidence detections only during navigation
        const highConfidenceDetections = detections.filter(d => d.confidence > 0.7);
        if (highConfidenceDetections.length === 0) {
            display.textContent = 'No significant objects detected';
            return;
        }

        const items = highConfidenceDetections.map(d =>
            `${d.class_name} (${Math.round(d.confidence * 100)}%)`
        ).join(', ');

        display.innerHTML = `<strong>Detected:</strong> ${items}`;
        display.style.background = 'rgba(37, 99, 235, 0.1)';
        display.style.border = '1px solid var(--primary)';
    } else {
        // Normal mode - show all detections
        const items = detections.map(d =>
            `${d.class_name} (${Math.round(d.confidence * 100)}%)`
        ).join(', ');

        display.textContent = items;
        display.style.background = 'var(--light)';
        display.style.border = 'none';
    }
}

// Scene Description
async function describeScene() {
    if (!stream) {
        logToUI('Camera is not running.');
        showToast('Please start camera first', 'error');
        return;
    }

    try {
        logToUI('Analyzing camera frame...');
        showToast('Generating description...');

        // Capture current frame
        const context = canvasElement.getContext('2d');
        canvasElement.width = videoElement.videoWidth;
        canvasElement.height = videoElement.videoHeight;
        context.drawImage(videoElement, 0, 0);
        const imageData = canvasElement.toDataURL('image/jpeg', 0.8);

        const data = await apiFetchJson(`${API_BASE}/api/process_frame`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                image: imageData,
                get_description: true
            })
        }, 'Failed to describe scene');
        if (!data) return;

        if (data.success && data.description) {
            logToUI(`Scene description ready: ${data.description}`);
            await handleResponseSpeech(data, 20, 'scene');
            addAnnouncement(data.description);
            showToast('Description generated');
        } else {
            logToUI('Could not generate description.');
            showToast('Could not generate description', 'error');
        }
    } catch (error) {
        console.error('Error describing scene:', error);
        logToUI(`Describe failed: ${error.message}`);
        showToast('Error: ' + error.message, 'error');
    }
}

// Navigation (kept for button-based access, but voice is primary)
function openNavigationModal() {
    logToUI('Opening navigation options...');
    // For blind users, use voice instead
    logToUI('Speaking response...');
    speak('Say "navigate" to start navigation, or tell me where you want to go.', 60, 'confirmation');
    // Still open modal as fallback
    document.getElementById('navigationModal').classList.add('active');
}

function closeNavigationModal() {
    document.getElementById('navigationModal').classList.remove('active');
}

async function startNavigation() {
    const destination = document.getElementById('destinationInput').value.trim();
    const mode = document.getElementById('modeSelect').value;

    if (!destination) {
        logToUI('Navigation destination is missing.');
        logToUI('Speaking response...');
        speak('Please tell me the destination. Say "navigate" and then tell me where you want to go.', 60, 'confirmation');
        showToast('Please enter a destination', 'error');
        return;
    }

    await startNavigationVoice(destination, mode);
    closeNavigationModal();
}

async function stopNavigation() {
    try {
        logToUI('Stopping navigation...');
        stopNavigationLocationTracking(); // Stop location tracking

        const data = await apiFetchJson(`${API_BASE}/api/stop_navigation`, {
            method: 'POST'
        }, 'Failed to stop navigation');
        if (!data) return;

        if (data.success) {
            document.getElementById('navigateBtn').style.display = 'block';
            document.getElementById('stopNavBtn').style.display = 'none';
            // Remove navigation mode class from body
            document.body.classList.remove('navigation-active');
            showToast('Navigation stopped');
            logToUI('Navigation stopped.');
            await handleResponseSpeech(data, 80, 'navigation');
            navigationActive = false;
            navigationRouteDestination = null;
            navigationRouteMode = null;
            navigationRouteLastUpdate = 0;
            navigationSessionId = null;
            if (directionsRenderer) {
                directionsRenderer.set('directions', null);
            }
            clearNavigationRoute();

            // Reset navigation state
            navigationState = { waitingForDestination: false, waitingForMode: false, destination: null, mode: null };
        }
    } catch (error) {
        console.error('Error stopping navigation:', error);
        logToUI(`Stop navigation failed: ${error.message}`);
    }
}

// Location
async function postLocationToServer(payload, defaultMessage = 'Failed to set location') {
    return await apiFetchJson(`${API_BASE}/api/location`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    }, defaultMessage);
}

async function captureAndSendLocation({ announce = false } = {}) {
    if (locationSending) return null;
    locationSending = true;
    try {
        if (!navigator.geolocation) {
            showToast('GPS not supported by this browser', 'error');
            return null;
        }

        const position = await new Promise((resolve, reject) => {
            let watchId = null;
            let bestPosition = null;

            const timeoutId = setTimeout(() => {
                if (watchId) navigator.geolocation.clearWatch(watchId);
                if (bestPosition) {
                    resolve(bestPosition);
                } else {
                    reject(new Error('Location timeout'));
                }
            }, 10000); // 10s max wait for a good GPS lock

            watchId = navigator.geolocation.watchPosition(
                (pos) => {
                    // Save best accuracy coordinate
                    if (!bestPosition || pos.coords.accuracy < bestPosition.coords.accuracy) {
                        bestPosition = pos;
                    }
                    // If accuracy is 50 meters or better, we have a strong GPS lock
                    if (pos.coords.accuracy <= 50) {
                        clearTimeout(timeoutId);
                        navigator.geolocation.clearWatch(watchId);
                        resolve(bestPosition);
                    }
                },
                (err) => {
                    if (!bestPosition) {
                        clearTimeout(timeoutId);
                        if (watchId) navigator.geolocation.clearWatch(watchId);
                        reject(err);
                    }
                },
                { enableHighAccuracy: true, maximumAge: 0 }
            );
        });

        const lat = position.coords.latitude;
        const lng = position.coords.longitude;
        const currentLocation = { lat, lng };
        lastKnownUserLocation = currentLocation;
        updateUserMarker(currentLocation);

        if (navigationActive && navigationSessionId) {
            await refreshNavigationRoute();
        }

        const data = await postLocationToServer({ lat, lng, speak: announce }, 'Failed to send location to server');
        if (data && data.location_name) {
            lastKnownUserLocation.location_name = data.location_name;
            if (announce) {
                logToUI(`Location found: ${data.location_name}`);
                await handleResponseSpeech(data, 60, 'location');
            }
        }

        if (announce && lastKnownUserLocation) {
            const locationText = lastKnownUserLocation.location_name || `${lat.toFixed(4)}, ${lng.toFixed(4)}`;
            showToast(`Location: ${locationText}`);
        }

        return lastKnownUserLocation;
    } catch (error) {
        logToUI('Could not get GPS location: ' + (error.message || 'Permission denied or unavailable'));
        showToast('GPS location unavailable', 'error');
        return null;
    } finally {
        locationSending = false;
    }
}

async function getLocation() {
    try {
        logToUI('Finding your location...');
        showToast('Getting location...');
        const loc = await captureAndSendLocation({ fallbackToIP: true, announce: true });
        if (!loc) {
            logToUI('Could not determine location.');
            showToast('Could not determine location', 'error');
        }
    } catch (error) {
        console.error('Error getting location:', error);
        logToUI(`Location failed: ${error.message || 'Unknown error'}`);
        showToast('Could not determine location: ' + (error.message || 'Unknown error'), 'error');
    }
}

// Voice Recognition
function checkVoiceSupport() {
    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        voiceRecognition = new SpeechRecognition();
        voiceRecognition.continuous = true; // Continuous listening for blind users
        voiceRecognition.interimResults = false;
        voiceRecognition.lang = 'en-US';

        voiceRecognition.onresult = (event) => {
            // Handle continuous recognition results
            let finalTranscript = '';

            for (let i = event.resultIndex; i < event.results.length; i++) {
                const transcript = event.results[i][0].transcript;
                if (event.results[i].isFinal) {
                    finalTranscript += transcript + ' ';
                }
            }

            if (finalTranscript.trim()) {
                console.log('Final transcript:', finalTranscript.trim());
                document.getElementById('voiceStatus').textContent = `Heard: ${finalTranscript.trim()}`;
                handleVoiceCommand(finalTranscript.trim());
            }
        };

        voiceRecognition.onerror = (event) => {
            console.error('Speech recognition error:', event.error);
            let errorMsg = 'Voice recognition error: ' + event.error;

            // Handle specific errors
            if (event.error === 'no-speech') {
                errorMsg = 'No speech detected. Please try speaking clearly.';
            } else if (event.error === 'audio-capture') {
                errorMsg = 'Microphone access denied or not available.';
            } else if (event.error === 'not-allowed') {
                errorMsg = 'Microphone permission denied. Please allow microphone access.';
            }

            showToast(errorMsg, 'error');
            document.getElementById('voiceStatus').textContent = errorMsg;
        };

        voiceRecognition.onend = () => {
            if (isListening) {
                // Restart if still listening
                try {
                    voiceRecognition.start();
                } catch (e) {
                    console.error('Error restarting recognition:', e);
                }
            }
        };
    } else {
        document.getElementById('voiceBtn').disabled = true;
        document.getElementById('voiceBtn').textContent = 'Voice Not Supported';
    }
}

function toggleVoice() {
    if (!voiceRecognition) {
        logToUI('Voice recognition is not supported in this browser.');
        showToast('Voice recognition not supported', 'error');
        return;
    }

    if (isListening) {
        stopVoice();
    } else {
        startVoice();
    }
}

function startVoice() {
    if (backendVoiceListening) {
        logToUI('Backend voice listener is already active.');
        showToast('Backend voice input is active. Browser voice is disabled to avoid conflicts.', 'info');
        return;
    }
    try {
        logToUI('Starting voice recognition...');
        voiceRecognition.start();
        isListening = true;
        document.getElementById('voiceBtn').classList.add('listening');
        document.getElementById('voiceBtnText').textContent = 'Listening...';
        document.getElementById('voiceStatus').textContent = 'Listening for commands...';
        logToUI('Voice recognition started.');
        showToast('Voice recognition started');
    } catch (error) {
        console.error('Error starting voice recognition:', error);
        logToUI(`Voice start failed: ${error.message}`);
        showToast('Failed to start voice recognition', 'error');
    }
}

function stopVoice() {
    logToUI('Stopping voice recognition...');
    voiceRecognition.stop();
    isListening = false;
    document.getElementById('voiceBtn').classList.remove('listening');
    document.getElementById('voiceBtnText').textContent = 'Start Voice';
    document.getElementById('voiceStatus').textContent = '';
    logToUI('Voice recognition stopped.');
    showToast('Voice recognition stopped');
}

// Navigation state for voice-driven navigation
let navigationState = {
    waitingForDestination: false,
    waitingForMode: false,
    destination: null,
    mode: null
};

async function getBrowserLocation() {
    const loc = await captureAndSendLocation({ fallbackToIP: true, announce: false });
    return loc || null;
}

function updateUserMarker(location) {
    if (!map || !location || !window.google || !window.google.maps) return;
    if (!userMarker) {
        userMarker = new google.maps.Marker({
            position: location,
            map,
            title: 'You'
        });
    } else {
        userMarker.setPosition(location);
    }
}

function clearNavigationRoute() {
    if (navigationPolyline) {
        navigationPolyline.setMap(null);
        navigationPolyline = null;
    }
    if (navigationStartMarker) {
        navigationStartMarker.setMap(null);
        navigationStartMarker = null;
    }
    if (navigationEndMarker) {
        navigationEndMarker.setMap(null);
        navigationEndMarker = null;
    }
    navigationPolylineEncoded = null;
}

function drawNavigationRouteFromPolyline(polyline) {
    if (!map || !polyline) return;
    const points = decodePolyline(polyline).map(([lat, lng]) => ({ lat, lng }));
    if (points.length === 0) return;

    clearNavigationRoute();

    navigationPolyline = new google.maps.Polyline({
        path: points,
        strokeColor: '#1A73E8',
        strokeWeight: 6,
        strokeOpacity: 0.95,
        map
    });
    navigationPolylineEncoded = polyline;

    navigationStartMarker = new google.maps.Marker({
        position: points[0],
        map,
        title: 'Start',
        icon: 'http://maps.google.com/mapfiles/ms/icons/green-dot.png'
    });
    navigationEndMarker = new google.maps.Marker({
        position: points[points.length - 1],
        map,
        title: 'Destination',
        icon: 'http://maps.google.com/mapfiles/ms/icons/red-dot.png'
    });

    const bounds = new google.maps.LatLngBounds();
    points.forEach(p => bounds.extend(p));
    map.fitBounds(bounds, { top: 20, right: 20, bottom: 20, left: 20 });
}

async function refreshNavigationRoute(force = false) {
    const validSessionId = normalizeSessionId(navigationSessionId);
    if (!validSessionId) return;
    const now = Date.now();
    if (!force && (now - navigationRouteLastUpdate) < 10000) return;
    navigationRouteLastUpdate = now;
    const data = await apiFetchJson(`${API_BASE}/api/location_polyline?session_id=${encodeURIComponent(validSessionId)}`, {}, 'Failed to load route');
    if (!data || !data.polyline) return;
    if (data.polyline !== navigationPolylineEncoded) {
        drawNavigationRouteFromPolyline(data.polyline);
    }
}

async function waitForNavigationPolyline(maxAttempts = 10, delayMs = 1000) {
    const validSessionId = normalizeSessionId(navigationSessionId);
    if (!validSessionId) return false;
    for (let i = 0; i < maxAttempts; i++) {
        const data = await apiFetchJson(`${API_BASE}/api/location_polyline?session_id=${encodeURIComponent(validSessionId)}`, {}, 'Failed to load route');
        if (data && data.polyline) {
            drawNavigationRouteFromPolyline(data.polyline);
            return true;
        }
        await new Promise(resolve => setTimeout(resolve, delayMs));
    }
    return false;
}


// Voice-driven navigation
async function startNavigationVoice(destination, mode) {
    try {
        logToUI('Calculating route...');
        await captureAndSendLocation({ fallbackToIP: true, announce: false });
        const data = await apiFetchJson(`${API_BASE}/api/navigate`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                destination: destination,
                mode: mode
            })
        }, 'Failed to start navigation');
        if (!data) return;

        if (data.success) {
            document.getElementById('navigateBtn').style.display = 'none';
            document.getElementById('stopNavBtn').style.display = 'block';
            // Add navigation mode class to body
            document.body.classList.add('navigation-active');
            showToast('Navigation started');
            navigationActive = true;
            navigationRouteDestination = destination;
            navigationRouteMode = mode;
            navigationSessionId = normalizeSessionId(data.session_id);

            // Start continuous location tracking during navigation
            startNavigationLocationTracking();
            logToUI(`Navigation started to ${destination}.`);
            await handleResponseSpeech(data, 80, 'navigation');

            const origin = lastKnownUserLocation || await getBrowserLocation();
            if (origin) {
                lastKnownUserLocation = origin;
                updateUserMarker(origin);
            }
            navigationRouteLastUpdate = 0;
            await waitForNavigationPolyline();
        } else {
            logToUI(`Navigation failed: ${data.message || 'Unknown error'}`);
            logToUI('Speaking response...');
            speak(`Navigation failed: ${data.message || 'Unknown error'}`, 60, 'confirmation');
            showToast(data.message || 'Failed to start navigation', 'error');
        }
    } catch (error) {
        console.error('Error starting navigation:', error);
        logToUI(`Navigation error: ${error.message}`);
        logToUI('Speaking response...');
        speak(`Error starting navigation: ${error.message}`, 60, 'confirmation');
        showToast('Error: ' + error.message, 'error');
    }
}

function isEmergencyCommand(lowerCommand) {
    return EMERGENCY_KEYWORDS.some(k => lowerCommand.includes(k));
}

function isNavigationControlCommand(lowerCommand) {
    return NAVIGATION_CONTROL_PHRASES.some((phrase) => {
        return lowerCommand === phrase || lowerCommand.includes(phrase);
    });
}

// Enhanced voice command handler with personal object detection
async function handleVoiceCommand(command) {
    console.log('Voice command:', command);
    const lowerCommand = command.toLowerCase().trim();
    logToUI(`Voice command received: ${command}`);

    const voiceStatus = document.getElementById('voiceStatus');
    if (voiceStatus) {
        voiceStatus.textContent = `Heard: ${command}`;
        voiceStatus.className = 'voice-status processing';
    }
    updateStatusIndicator('Processing command...', 'processing');

    const isEmergency = isEmergencyCommand(lowerCommand);
    if (isEmergency) {
        logToUI('Speaking response...');
        speak('Emergency detected. Sending alert now.', 100, 'emergency');
        await triggerEmergency('voice');
        return;
    }

    const isNavControl = isNavigationControlCommand(lowerCommand);
    const recentlySpeaking = isBackendSpeechActive || (Date.now() - lastBackendSpeechActiveAt) < 1800;
    if (recentlySpeaking && !isNavControl) {
        console.log('Ignoring likely TTS echo:', command);
        updateStatusIndicator('Listening...', 'active');
        return;
    }

    if (navigationActive && !isNavControl) {
        console.log('Ignoring non-control voice during navigation:', command);
        updateStatusIndicator('Say "stop navigation" to stop route guidance.', 'waiting');
        return;
    }

    const personalObjectRegex = /(where|find|locate|see)\s+(my|your)\s+(\w+)/;
    const match = lowerCommand.match(personalObjectRegex);
    if (match) {
        const objectName = match[3];
        logToUI(`Looking for personal object: ${objectName}`);
        logToUI('Speaking response...');
        speak(`Looking for your ${objectName}...`, 40, 'personal');
        await searchForPersonalObject(objectName);
        return;
    }

    const commandId = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    lastSentCommandId = commandId;
    await apiFetchJson(`${API_BASE}/api/voice_command`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            command,
            command_id: commandId,
            source: 'web'
        })
    }, 'Failed to send voice command');
    logToUI('Voice command sent to backend.');
    updateStatusIndicator('Command sent to backend...', 'processing');
}

// Continuous location tracking during navigation
let locationIntervalId = null;
let lastLocationSentAt = 0;
let lastTrackedPosition = null;
const LOCATION_SEND_THROTTLE_MS = 5000;
const LOCATION_MIN_DISTANCE_M = 8;

function haversineDistanceMeters(a, b) {
    if (!a || !b) return Infinity;
    const toRad = (value) => (value * Math.PI) / 180;
    const earthRadiusM = 6371000;
    const dLat = toRad(b.lat - a.lat);
    const dLng = toRad(b.lng - a.lng);
    const lat1 = toRad(a.lat);
    const lat2 = toRad(b.lat);
    const x = Math.sin(dLat / 2) ** 2 + Math.cos(lat1) * Math.cos(lat2) * Math.sin(dLng / 2) ** 2;
    return 2 * earthRadiusM * Math.asin(Math.sqrt(x));
}

async function sendTrackedLocation(lat, lng, { force = false } = {}) {
    const nextLocation = { lat, lng };
    const now = Date.now();
    const movedDistance = lastTrackedPosition ? haversineDistanceMeters(lastTrackedPosition, nextLocation) : Infinity;
    if (!force) {
        if ((now - lastLocationSentAt) < LOCATION_SEND_THROTTLE_MS && movedDistance < LOCATION_MIN_DISTANCE_M) {
            return null;
        }
    }
    lastLocationSentAt = now;
    lastTrackedPosition = nextLocation;
    lastKnownUserLocation = { ...(lastKnownUserLocation || {}), ...nextLocation };
    updateUserMarker(lastKnownUserLocation);
    const data = await postLocationToServer({ lat, lng, speak: false }, 'Failed to send navigation location');
    if (data && data.location_name) {
        lastKnownUserLocation.location_name = data.location_name;
    }
    if (navigationActive && navigationSessionId) {
        await refreshNavigationRoute();
        await fetchNavigationStatus();
    }
    return data;
}

function startNavigationLocationTracking() {
    if (locationIntervalId !== null || locationWatchId !== null) return;
    if (!navigationActive) return;

    if (!navigator.geolocation) {
        return;
    }

    const sendLocationTick = async () => {
        const now = Date.now();
        if ((now - lastLocationSentAt) < LOCATION_SEND_THROTTLE_MS) return;
        await captureAndSendLocation({ fallbackToIP: false, announce: false });
    };

    try {
        locationWatchId = navigator.geolocation.watchPosition(async (position) => {
            const lat = position.coords.latitude;
            const lng = position.coords.longitude;
            await sendTrackedLocation(lat, lng);
        }, (error) => {
            console.warn('Navigation watchPosition error:', error);
        }, {
            enableHighAccuracy: true,
            timeout: 10000,
            maximumAge: 2000
        });
    } catch (error) {
        console.warn('Failed to start watchPosition:', error);
    }

    sendLocationTick();
    locationIntervalId = setInterval(sendLocationTick, LOCATION_SEND_THROTTLE_MS);
    startNavigationStatusPolling();
}

function stopNavigationLocationTracking() {
    if (locationIntervalId !== null) {
        clearInterval(locationIntervalId);
    }
    locationIntervalId = null;
    if (locationWatchId !== null && navigator.geolocation) {
        navigator.geolocation.clearWatch(locationWatchId);
    }
    locationWatchId = null;
    lastTrackedPosition = null;
    stopNavigationStatusPolling();
}

async function fetchNavigationStatus() {
    if (!navigationActive) return null;
    const data = await apiFetchJson(`${API_BASE}/api/navigation_status`, {}, 'Failed to load navigation status');
    if (!data || !data.success || !data.navigation) return null;
    const nav = data.navigation;
    const statusText = nav.next_instruction
        ? `${nav.distance_to_next_turn_text || ''} ${nav.next_instruction}`.trim()
        : 'Navigation active';
    updateStatusIndicator(statusText, 'active');
    const voiceStatus = document.getElementById('voiceStatus');
    if (voiceStatus && nav.next_instruction) {
        const remaining = nav.remaining_distance_text ? ` | Remaining ${nav.remaining_distance_text}` : '';
        voiceStatus.textContent = `Next: ${nav.next_instruction}${remaining}`;
        voiceStatus.className = 'voice-status processing';
    }
    return nav;
}

function startNavigationStatusPolling() {
    if (navigationStatusIntervalId !== null) return;
    navigationStatusIntervalId = setInterval(() => {
        fetchNavigationStatus().catch((error) => {
            console.warn('Navigation status polling failed:', error);
        });
    }, 3000);
}

function stopNavigationStatusPolling() {
    if (navigationStatusIntervalId !== null) {
        clearInterval(navigationStatusIntervalId);
    }
    navigationStatusIntervalId = null;
}

// Resize video container
function resizeVideo(size) {
    const videoContainer = document.getElementById('videoContainer');

    // Remove all size classes
    videoContainer.classList.remove('small', 'medium', 'large', 'fullscreen');

    if (size === 'fullscreen') {
        // Close any other modals when going fullscreen
        document.getElementById('navigationModal').classList.remove('active');
        document.getElementById('addObjectsModal').classList.remove('active');
    }

    // Add the selected size class
    videoContainer.classList.add(size);
}

// Open add objects modal
function openAddObjectsModal() {
    logToUI('Opening personal object dialog...');
    document.getElementById('addObjectsModal').classList.add('active');
    updateSaveObjectButtonState();
}

// Close add objects modal
function closeAddObjectsModal() {
    document.getElementById('addObjectsModal').classList.remove('active');
}

// Capture current frame for personal object
async function captureCurrentFrameForObject() {
    if (!videoElement || !stream) {
        logToUI('Camera not running.');
        showToast('Camera not running', 'error');
        return;
    }

    try {
        logToUI('Capturing current frame for personal object...');
        // Create a temporary canvas to capture the frame
        const canvas = document.createElement('canvas');
        canvas.width = videoElement.videoWidth;
        canvas.height = videoElement.videoHeight;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(videoElement, 0, 0);

        // Convert to data URL
        const imageData = canvas.toDataURL('image/jpeg', 0.8);

        // Store the captured image temporarily
        window.capturedObjectFrame = imageData;
        window.uploadedObjectImageUrl = null;
        updateSaveObjectButtonState();

        logToUI('Current frame captured.');
        showToast('Current frame captured');
    } catch (error) {
        console.error('Error capturing frame:', error);
        logToUI('Error capturing frame.');
        showToast('Error capturing frame', 'error');
    }
}

function handleObjectImageUpload(event) {
    const file = event.target?.files?.[0];
    if (!file) {
        window.uploadedObjectImageUrl = null;
        updateSaveObjectButtonState();
        return;
    }
    const reader = new FileReader();
    reader.onload = () => {
        window.uploadedObjectImageUrl = reader.result;
        window.capturedObjectFrame = null;
        updateSaveObjectButtonState();
    };
    reader.onerror = () => {
        console.error('Error reading uploaded image');
        window.uploadedObjectImageUrl = null;
        reportApiError('Could not read selected image');
        updateSaveObjectButtonState();
    };
    reader.readAsDataURL(file);
}

function getCurrentObjectImageUrl() {
    return window.uploadedObjectImageUrl || window.capturedObjectFrame || null;
}

function updateSaveObjectButtonState() {
    const btn = document.getElementById('saveObjectBtn');
    if (!btn) return;
    const name = document.getElementById('objectNameInput')?.value?.trim();
    const imageUrl = getCurrentObjectImageUrl();
    btn.disabled = !(name && imageUrl);
}

// Save personal object
async function savePersonalObject(objectName, imageUrl) {
    const trimmedName = (objectName || '').trim();
    if (!trimmedName || !imageUrl) {
        logToUI('Cannot save object: missing name or image.');
        reportApiError('Cannot save object: missing name or image.');
        return;
    }

    try {
        logToUI(`Saving personal object: ${trimmedName}`);
        const payload = {
            object_name: trimmedName,
            image_url: imageUrl
        };

        console.log('Saving object payload:', payload);

        const data = await apiFetchJson(`${API_BASE}/api/add_personal_object`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        }, 'Failed to save object');
        if (!data) return;

        if (data.success) {
            logToUI(`Personal object saved: ${trimmedName}`);
            await handleResponseSpeech(data, 60, 'personal_object');
            showToast('Object saved successfully', 'success');
            closeAddObjectsModal();
            // Clear the input
            document.getElementById('objectNameInput').value = '';
            window.capturedObjectFrame = null;
            window.uploadedObjectImageUrl = null;
            const uploadInput = document.getElementById('objectImageUpload');
            if (uploadInput) uploadInput.value = '';
            updateSaveObjectButtonState();
        } else {
            showToast(data.message || 'Failed to save object', 'error');
        }
    } catch (error) {
        console.error('Error saving personal object:', error);
        logToUI('Error saving personal object.');
        showToast('Error saving object', 'error');
    }
}

// Search for a personal object
async function searchForPersonalObject(objectName) {
    try {
        logToUI(`Searching for personal object: ${objectName}`);
        const data = await apiFetchJson(`${API_BASE}/api/search_personal_object`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ object_name: objectName })
        }, 'Failed to search for object');
        if (!data) return 'Error searching for object';

        if (data.success) {
            logToUI(data.message);
            await handleResponseSpeech(data, 40, 'personal_search');
            addAnnouncement(data.message);
            showToast(data.message);
            return data.message;
        } else {
            const errorMessage = data.message || 'Error searching for object';
            logToUI(errorMessage);
            logToUI('Speaking response...');
            speak(errorMessage);
            showToast(errorMessage, 'error');
            return errorMessage;
        }
    } catch (error) {
        console.error('Error searching for personal object:', error);
        const errorMessage = 'Error searching for object';
        logToUI(errorMessage);
        logToUI('Speaking response...');
        speak(errorMessage);
        showToast(errorMessage, 'error');
        return errorMessage;
    }
}

// Logout function
async function logout() {
    if (confirm('Are you sure you want to logout?')) {
        try {
            logToUI('Logging out...');
            // Stop camera and voice if running
            if (stream) {
                stopCamera();
            }
            if (isListening) {
                stopVoice();
            }

            // Stop navigation if active
            if (document.getElementById('stopNavBtn').style.display !== 'none') {
                stopNavigation();
            }

            // Call logout API
            const data = await apiFetchJson(`${API_BASE}/api/logout`, {
                method: 'POST'
            }, 'Logout failed');
            if (!data) return;

            if (data.success) {
                logToUI('Logged out successfully.');
                showToast('Logged out successfully');
                localStorage.removeItem('access_token');
                localStorage.removeItem('refresh_token');
                // Redirect to login page
                window.location.href = '/login';
            } else {
                showToast('Logout failed: ' + (data.message || 'Unknown error'), 'error');
            }
        } catch (error) {
            console.error('Error during logout:', error);
            logToUI('Logout failed.');
            // Still redirect to login page even if API call fails
            window.location.href = '/login';
        }
    }
}

// Text-to-Speech
let speechQueue = [];
const USE_BACKEND_SPEECH = (location.hostname === 'localhost' || location.hostname === '127.0.0.1');
let forceBackendSpeech = USE_BACKEND_SPEECH;
let lastBackendSpokenId = 0;

async function speak(text, priority = 40, source = 'web') {
    if (!text || !String(text).trim()) return;
    if (forceBackendSpeech) {
        await speakViaBackend(text, priority, source);
        return;
    }
    speechQueue.push({ text, priority, source });
    if (!isSpeaking) {
        await processSpeechQueue();
    }
}

async function speakViaBackend(text, priority = 40, source = 'web') {
    if (!text || !String(text).trim()) return;
    try {
        const response = await apiFetch('/api/speak', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text, priority, source })
        });
        if (!response) return;
        if (!response.ok) {
            let message = 'Speech request failed';
            try { message = (await response.json()).message || message; } catch (e) {}
            reportApiError(message);
        }
    } catch (error) {
        console.warn('Speech backend error (no fallback):', error);
    }
}

async function processSpeechQueue() {
    if (speechQueue.length === 0) {
        isSpeaking = false;
        return;
    }

    isSpeaking = true;
    speechQueue.sort((a, b) => b.priority - a.priority);
    const speechItem = speechQueue.shift();

    if (forceBackendSpeech) {
        try {
            const response = await apiFetch('/api/speak', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(speechItem)
            });
            if (response && response.ok) {
                let data = null;
                try { data = await response.json(); } catch(e) {}
                if (data && data.success) {
                    addAnnouncement(speechItem.text);
                } else {
                    await speakLocally(speechItem.text);
                }
            } else {
                await speakLocally(speechItem.text);
            }
        } catch (error) {
            await speakLocally(speechItem.text);
        }
    } else {
        await speakLocally(speechItem.text);
    }

    setTimeout(processSpeechQueue, 100);
}

function speakLocally(text) {
    return new Promise((resolve) => {
        if ('speechSynthesis' in window) {
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.rate = 1.0;
            utterance.pitch = 1.0;
            utterance.volume = 1.0;
            
            utterance.onend = () => { addAnnouncement(text); resolve(); };
            utterance.onerror = () => { console.log('TTS Error:', text); addAnnouncement(text); resolve(); };
            
            window.speechSynthesis.speak(utterance);
        } else {
            console.log('TTS:', text);
            addAnnouncement(text);
            resolve();
        }
    });
}

function clearSpeechQueue() {
    speechQueue = [];
    isSpeaking = false;
    if ('speechSynthesis' in window) window.speechSynthesis.cancel();
    apiFetch('/api/speech/cancel', { method: 'POST' }).catch(() => {});
}

async function getSpeechStatus() {
    try {
        const data = await apiFetchJson('/api/speech/status', {}, 'Failed to load speech status');
        if (!data) return { is_speaking: isSpeaking, queue_size: speechQueue.length };
        return data;
    } catch (error) {
        return { is_speaking: isSpeaking, queue_size: speechQueue.length };
    }
}

async function fetchSpeechStatus() {
    const status = await getSpeechStatus();
    if (!status) return;
    
    isBackendSpeechActive = !!status.is_speaking;
    if (isBackendSpeechActive) lastBackendSpeechActiveAt = Date.now();
}

// Feedback
async function submitFeedback() {
    const feedback = document.getElementById('feedbackInput').value.trim();

    if (!feedback) {
        logToUI('Feedback is empty.');
        showToast('Please enter feedback', 'error');
        return;
    }

    try {
        logToUI('Submitting feedback...');
        const data = await apiFetchJson(`${API_BASE}/api/feedback`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ feedback: feedback })
        }, 'Failed to submit feedback');
        if (!data) return;

        if (data.success) {
            document.getElementById('feedbackInput').value = '';
            logToUI('Feedback submitted.');
            await handleResponseSpeech(data, 40, 'feedback');
            showToast('Feedback submitted successfully');
        } else {
            showToast('Failed to submit feedback', 'error');
        }
    } catch (error) {
        console.error('Error submitting feedback:', error);
        logToUI(`Feedback failed: ${error.message}`);
        showToast('Error: ' + error.message, 'error');
    }
}

// Announcements
function addAnnouncement(text) {
    const list = document.getElementById('announcementsList');
    const item = document.createElement('div');
    item.className = 'announcement-item';
    item.textContent = text;
    list.insertBefore(item, list.firstChild);

    // Keep only last 10 announcements
    while (list.children.length > 10) {
        list.removeChild(list.lastChild);
    }
}

// Status Updates
async function fetchStatus() {
    await updateStatus();
}

async function updateStatus() {
    try {
        const data = await apiFetchJson(`${API_BASE}/api/status`, {}, 'Failed to load status');
        if (!data) return;

        if (data.success) {
            const status = data.status;
            backendVoiceListening = !!status.voice_active;
            const statusText = document.getElementById('statusText');
            const statusDot = document.querySelector('.status-dot');

            if (backendVoiceListening && isListening) {
                stopVoice();
                const voiceStatus = document.getElementById('voiceStatus');
                if (voiceStatus) {
                    voiceStatus.textContent = 'Using backend voice listener';
                }
            }

            const navigating = !!status.navigation_active;
            if (navigating !== navigationActive) {
                navigationActive = navigating;
                if (navigationActive) {
                    startNavigationLocationTracking();
                    document.body.classList.add('navigation-active');
                    const navBtn = document.getElementById('navigateBtn');
                    const stopBtn = document.getElementById('stopNavBtn');
                    if (navBtn) navBtn.style.display = 'none';
                    if (stopBtn) stopBtn.style.display = 'block';
                    if (!normalizeSessionId(navigationSessionId) && status.navigation_session_id) {
                        navigationSessionId = normalizeSessionId(status.navigation_session_id);
                        navigationRouteLastUpdate = 0;
                        waitForNavigationPolyline();
                    }
                } else {
                    stopNavigationLocationTracking();
                    document.body.classList.remove('navigation-active');
                    const navBtn = document.getElementById('navigateBtn');
                    const stopBtn = document.getElementById('stopNavBtn');
                    if (navBtn) navBtn.style.display = 'block';
                    if (stopBtn) stopBtn.style.display = 'none';
                    navigationSessionId = null;
                    clearNavigationRoute();
                }
            }
            if (navigationActive && !normalizeSessionId(navigationSessionId) && status.navigation_session_id) {
                navigationSessionId = normalizeSessionId(status.navigation_session_id);
                navigationRouteLastUpdate = 0;
                waitForNavigationPolyline();
            }

            const lastCmd = status.last_command || null;
            if (lastCmd && lastCmd.id && lastCmd.id !== lastCommandId) {
                lastCommandId = lastCmd.id;
                if (!lastSentCommandId || lastSentCommandId !== lastCmd.id) {
                    const voiceStatus = document.getElementById('voiceStatus');
                    if (voiceStatus && lastCmd.command) {
                        voiceStatus.textContent = `Command: ${lastCmd.command}`;
                    }
                    addAnnouncement(lastCmd.command || 'Command received');
                }
            }

            if (status.navigation_active) {
                statusText.textContent = 'Navigating';
                statusDot.style.background = 'var(--warning-color)';
            } else if (status.camera_running) {
                statusText.textContent = 'Camera Active';
                statusDot.style.background = 'var(--secondary-color)';
            } else {
                statusText.textContent = 'Ready';
                statusDot.style.background = 'var(--secondary-color)';
            }
        }
    } catch (error) {
        console.error('Error updating status:', error);
    }
}

// Toast Notifications
function showToast(message, type = 'info') {
    const toast = document.getElementById('toast');
    toast.textContent = message;
    toast.className = `toast show ${type}`;

    setTimeout(() => {
        toast.classList.remove('show');
    }, 3000);
}

// Handle page visibility (pause/resume processing)
document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
        stopFrameProcessing();
    } else if (stream) {
        startFrameProcessing();
    }
});

// Status indicator update function
function updateStatusIndicator(text, status) {
    const statusIndicator = document.getElementById('statusIndicator');
    const statusText = document.getElementById('statusText');
    const statusDot = statusIndicator.querySelector('.status-dot');

    statusText.textContent = text;

    // Remove all status classes
    statusDot.className = 'status-dot';

    // Add appropriate status class
    switch (status) {
        case 'processing':
            statusDot.classList.add('processing');
            break;
        case 'waiting':
            statusDot.classList.add('waiting');
            break;
        case 'active':
            statusDot.classList.add('active');
            break;
        case 'error':
            statusDot.classList.add('error');
            break;
        default:
            statusDot.classList.add('ready');
    }

    // Reset voice status after processing
    if (status === 'ready' || status === 'active') {
        setTimeout(() => {
            const voiceStatus = document.getElementById('voiceStatus');
            if (voiceStatus) {
                voiceStatus.className = 'voice-status';
                voiceStatus.textContent = 'Voice ready';
            }
        }, 2000);
    }
}

// Prevent zoom on double tap (mobile)
let lastTouchEnd = 0;
document.addEventListener('touchend', (event) => {
    const now = Date.now();
    if (now - lastTouchEnd <= 300) {
        event.preventDefault();
    }
    lastTouchEnd = now;
}, false);

