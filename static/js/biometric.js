// Biometric System JavaScript Functions

// Global variables
let isRecording = false;
let cameraStream = null;
let audioContext = null;
let mediaRecorder = null;

// Utility Functions
function showNotification(message, type = 'info', duration = 5000) {
    const notification = document.createElement('div');
    notification.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
    notification.style.cssText = `
        top: 20px;
        right: 20px;
        z-index: 1050;
        min-width: 300px;
        animation: slideInRight 0.3s ease-out;
    `;
    
    notification.innerHTML = `
        <i class="fas fa-${getIconForType(type)} me-2"></i>
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    document.body.appendChild(notification);
    
    // Auto remove
    setTimeout(() => {
        if (notification.parentNode) {
            notification.remove();
        }
    }, duration);
}

function getIconForType(type) {
    const icons = {
        'success': 'check-circle',
        'danger': 'exclamation-circle',
        'warning': 'exclamation-triangle',
        'info': 'info-circle',
        'primary': 'bell'
    };
    return icons[type] || 'info-circle';
}

// Camera Functions
class CameraManager {
    constructor(videoElement) {
        this.videoElement = videoElement;
        this.stream = null;
        this.isActive = false;
    }
    
    async initialize() {
        try {
            this.stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 1280 },
                    height: { ideal: 720 },
                    facingMode: 'user'
                }
            });
            
            if (this.videoElement) {
                this.videoElement.srcObject = this.stream;
                this.isActive = true;
                return true;
            }
        } catch (error) {
            console.error('Camera initialization failed:', error);
            showNotification('Camera access denied. Please allow camera permissions.', 'danger');
            return false;
        }
    }
    
    stop() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
            this.isActive = false;
        }
    }
    
    captureFrame() {
        if (!this.videoElement || !this.isActive) return null;
        
        const canvas = document.createElement('canvas');
        canvas.width = this.videoElement.videoWidth;
        canvas.height = this.videoElement.videoHeight;
        
        const ctx = canvas.getContext('2d');
        ctx.drawImage(this.videoElement, 0, 0);
        
        return canvas.toDataURL('image/jpeg', 0.8);
    }
}

// Audio Functions
class AudioManager {
    constructor() {
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.isRecording = false;
    }
    
    async initialize() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ 
                audio: {
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true
                }
            });
            
            this.mediaRecorder = new MediaRecorder(stream, {
                mimeType: 'audio/webm'
            });
            
            this.mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    this.audioChunks.push(event.data);
                }
            };
            
            return true;
        } catch (error) {
            console.error('Audio initialization failed:', error);
            showNotification('Microphone access denied. Please allow microphone permissions.', 'danger');
            return false;
        }
    }
    
    startRecording() {
        if (this.mediaRecorder && !this.isRecording) {
            this.audioChunks = [];
            this.mediaRecorder.start();
            this.isRecording = true;
            return true;
        }
        return false;
    }
    
    stopRecording() {
        return new Promise((resolve) => {
            if (this.mediaRecorder && this.isRecording) {
                this.mediaRecorder.onstop = () => {
                    const audioBlob = new Blob(this.audioChunks, { type: 'audio/webm' });
                    this.isRecording = false;
                    resolve(audioBlob);
                };
                this.mediaRecorder.stop();
            } else {
                resolve(null);
            }
        });
    }
    
    cleanup() {
        if (this.mediaRecorder && this.mediaRecorder.stream) {
            this.mediaRecorder.stream.getTracks().forEach(track => track.stop());
        }
    }
}

// Biometric Authentication Manager
class BiometricAuth {
    constructor() {
        this.camera = null;
        this.audio = null;
        this.currentStep = 1;
        this.maxSteps = 3;
        this.authData = {};
    }
    
    async initialize() {
        try {
            // Initialize camera
            const videoElement = document.getElementById('cameraFeed');
            if (videoElement) {
                this.camera = new CameraManager(videoElement);
                await this.camera.initialize();
            }
            
            // Initialize audio
            this.audio = new AudioManager();
            await this.audio.initialize();
            
            return true;
        } catch (error) {
            console.error('Biometric initialization failed:', error);
            return false;
        }
    }
    
    async authenticateUser() {
        try {
            this.updateProgress(1, 'Capturing face...');
            
            // Step 1: Face Authentication
            const faceResult = await this.authenticateFace();
            if (!faceResult.success) {
                throw new Error(faceResult.error);
            }
            
            this.authData.face = faceResult.data;
            this.updateProgress(2, 'Face verified! Preparing voice verification...');
            
            // Step 2: Voice Authentication
            const voiceResult = await this.authenticateVoice();
            if (!voiceResult.success) {
                throw new Error(voiceResult.error);
            }
            
            this.authData.voice = voiceResult.data;
            this.updateProgress(3, 'Authentication complete!');
            
            // Step 3: Complete Authentication
            return await this.completeAuthentication();
            
        } catch (error) {
            console.error('Authentication failed:', error);
            showNotification(error.message, 'danger');
            return { success: false, error: error.message };
        }
    }
    
    async authenticateFace() {
        try {
            const response = await fetch('/api/authenticate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });
            
            const data = await response.json();
            
            if (response.ok) {
                return { success: true, data };
            } else {
                return { success: false, error: data.error || 'Face authentication failed' };
            }
        } catch (error) {
            return { success: false, error: 'Network error during face authentication' };
        }
    }
    
    async authenticateVoice() {
        try {
            // Start recording
            if (!this.audio.startRecording()) {
                throw new Error('Failed to start voice recording');
            }
            
            // Record for 5 seconds
            await this.recordVoiceWithProgress(5000);
            
            // Stop recording and get audio data
            const audioBlob = await this.audio.stopRecording();
            if (!audioBlob) {
                throw new Error('Failed to capture voice data');
            }
            
            // Send to server for verification
            const response = await fetch('/api/voice_verify', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });
            
            const data = await response.json();
            
            if (response.ok) {
                return { success: true, data };
            } else {
                return { success: false, error: data.error || 'Voice verification failed' };
            }
        } catch (error) {
            return { success: false, error: error.message };
        }
    }
    
    async recordVoiceWithProgress(duration) {
        return new Promise((resolve) => {
            const progressBar = document.getElementById('voiceProgressBar');
            const startTime = Date.now();
            
            const updateProgress = () => {
                const elapsed = Date.now() - startTime;
                const progress = Math.min((elapsed / duration) * 100, 100);
                
                if (progressBar) {
                    progressBar.style.width = progress + '%';
                }
                
                if (progress < 100) {
                    requestAnimationFrame(updateProgress);
                } else {
                    resolve();
                }
            };
            
            updateProgress();
        });
    }
    
    async completeAuthentication() {
        // This would typically send final authentication data to server
        return {
            success: true,
            data: {
                action: this.authData.face.action || 'LOGIN',
                user: this.authData.face.name,
                faceConfidence: this.authData.face.confidence,
                voiceConfidence: this.authData.voice.confidence,
                timestamp: new Date().toISOString()
            }
        };
    }
    
    updateProgress(step, message) {
        this.currentStep = step;
        
        // Update step indicators
        for (let i = 1; i <= this.maxSteps; i++) {
            const stepElement = document.getElementById(`step${i}`);
            if (stepElement) {
                const circle = stepElement.querySelector('.step-circle');
                if (i < step) {
                    circle.className = 'step-circle completed';
                } else if (i === step) {
                    circle.className = 'step-circle active';
                } else {
                    circle.className = 'step-circle';
                }
            }
        }
        
        // Update status message
        const statusElement = document.getElementById('statusMessage');
        if (statusElement) {
            statusElement.innerHTML = `<i class="fas fa-spinner fa-spin me-2"></i>${message}`;
        }
    }
    
    cleanup() {
        if (this.camera) {
            this.camera.stop();
        }
        if (this.audio) {
            this.audio.cleanup();
        }
    }
}

// Registration Manager
class RegistrationManager {
    constructor() {
        this.currentStep = 1;
        this.maxSteps = 4;
        this.registrationData = {};
        this.camera = null;
        this.audio = null;
    }
    
    async initialize() {
        // Similar to BiometricAuth initialization
        return await new BiometricAuth().initialize();
    }
    
    async registerUser(userData) {
        try {
            this.registrationData = userData;
            
            // Step 1: Capture face photos
            this.updateStep(2, 'Capturing face photos...');
            const faceResult = await this.captureFacePhotos();
            
            if (!faceResult.success) {
                throw new Error(faceResult.error);
            }
            
            // Step 2: Record voice clips
            this.updateStep(3, 'Recording voice clips...');
            const voiceResult = await this.recordVoiceClips();
            
            if (!voiceResult.success) {
                throw new Error(voiceResult.error);
            }
            
            // Step 3: Save to database
            this.updateStep(4, 'Saving profile...');
            const saveResult = await this.saveUserProfile();
            
            if (saveResult.success) {
                showNotification('Registration completed successfully!', 'success');
                return saveResult;
            } else {
                throw new Error(saveResult.error);
            }
            
        } catch (error) {
            console.error('Registration failed:', error);
            showNotification(error.message, 'danger');
            return { success: false, error: error.message };
        }
    }
    
    async captureFacePhotos() {
        try {
            const response = await fetch('/api/capture_face_photos', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });
            
            const data = await response.json();
            return response.ok ? { success: true, data } : { success: false, error: data.error };
        } catch (error) {
            return { success: false, error: 'Failed to capture face photos' };
        }
    }
    
    async recordVoiceClips() {
        try {
            const response = await fetch('/api/capture_voice_clips', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });
            
            const data = await response.json();
            return response.ok ? { success: true, data } : { success: false, error: data.error };
        } catch (error) {
            return { success: false, error: 'Failed to record voice clips' };
        }
    }
    
    async saveUserProfile() {
        // Profile is saved automatically by the API endpoints
        return { success: true };
    }
    
    updateStep(step, message) {
        this.currentStep = step;
        
        // Update UI elements
        const statusElement = document.getElementById('registrationStatus');
        if (statusElement) {
            statusElement.innerHTML = `<i class="fas fa-spinner fa-spin me-2"></i>${message}`;
        }
    }
}

// System Health Monitor
class SystemMonitor {
    constructor() {
        this.isMonitoring = false;
        this.healthStatus = {
            camera: false,
            microphone: false,
            database: false,
            models: false
        };
    }
    
    async startMonitoring() {
        if (this.isMonitoring) return;
        
        this.isMonitoring = true;
        this.checkSystemHealth();
        
        // Check every 30 seconds
        setInterval(() => {
            if (this.isMonitoring) {
                this.checkSystemHealth();
            }
        }, 30000);
    }
    
    async checkSystemHealth() {
        try {
            // Check camera
            this.healthStatus.camera = await this.checkCamera();
            
            // Check microphone
            this.healthStatus.microphone = await this.checkMicrophone();
            
            // Check database (via API)
            this.healthStatus.database = await this.checkDatabase();
            
            // Update UI
            this.updateHealthDisplay();
            
        } catch (error) {
            console.error('Health check failed:', error);
        }
    }
    
    async checkCamera() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            stream.getTracks().forEach(track => track.stop());
            return true;
        } catch {
            return false;
        }
    }
    
    async checkMicrophone() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            stream.getTracks().forEach(track => track.stop());
            return true;
        } catch {
            return false;
        }
    }
    
    async checkDatabase() {
        try {
            const response = await fetch('/api/health/database');
            return response.ok;
        } catch {
            return false;
        }
    }
    
    updateHealthDisplay() {
        const statusElements = {
            camera: document.getElementById('cameraStatus'),
            microphone: document.getElementById('microphoneStatus'),
            database: document.getElementById('databaseStatus'),
            models: document.getElementById('modelsStatus')
        };
        
        Object.keys(this.healthStatus).forEach(component => {
            const element = statusElements[component];
            if (element) {
                const isHealthy = this.healthStatus[component];
                element.className = `status-indicator ${isHealthy ? 'status-success' : 'status-danger'}`;
                element.innerHTML = `
                    <i class="fas fa-${isHealthy ? 'check-circle' : 'exclamation-circle'} me-2"></i>
                    ${component.charAt(0).toUpperCase() + component.slice(1)} ${isHealthy ? 'Ready' : 'Error'}
                `;
            }
        });
    }
    
    stopMonitoring() {
        this.isMonitoring = false;
    }
}

// Global instances
let biometricAuth = null;
let registrationManager = null;
let systemMonitor = null;

// Initialize on DOM load
document.addEventListener('DOMContentLoaded', function() {
    // Initialize system monitor
    systemMonitor = new SystemMonitor();
    systemMonitor.startMonitoring();
    
    // Add smooth scrolling
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
    
    // Add loading states to forms
    document.querySelectorAll('form').forEach(form => {
        form.addEventListener('submit', function() {
            const submitBtn = this.querySelector('button[type="submit"]');
            if (submitBtn && !submitBtn.disabled) {
                showLoading(submitBtn);
            }
        });
    });
});

// Cleanup on page unload
window.addEventListener('beforeunload', function() {
    if (biometricAuth) {
        biometricAuth.cleanup();
    }
    if (systemMonitor) {
        systemMonitor.stopMonitoring();
    }
});

// Export for global use
window.BiometricSystem = {
    BiometricAuth,
    RegistrationManager,
    SystemMonitor,
    CameraManager,
    AudioManager,
    showNotification
};