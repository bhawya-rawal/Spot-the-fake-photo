document.addEventListener('DOMContentLoaded', () => {
  // Elements
  const tabUpload = document.getElementById('tab-upload');
  const tabCamera = document.getElementById('tab-camera');
  const panelUpload = document.getElementById('panel-upload');
  const panelCamera = document.getElementById('panel-camera');

  const dropZone = document.getElementById('drop-zone');
  const fileInput = document.getElementById('file-input');
  const browseBtn = document.getElementById('browse-btn');

  const webcam = document.getElementById('webcam');
  const cameraCanvas = document.getElementById('camera-canvas');
  const captureBtn = document.getElementById('capture-btn');
  const cameraSelect = document.getElementById('camera-select');
  const cameraSelectContainer = document.getElementById('camera-select-container');

  const previewContainer = document.getElementById('preview-container');
  const previewPlaceholder = document.getElementById('preview-placeholder');
  const previewImage = document.getElementById('preview-image');
  const scanOverlay = document.getElementById('scan-overlay');

  const resultsPlaceholder = document.getElementById('results-placeholder');
  const resultsContent = document.getElementById('results-content');
  const verdictBanner = document.getElementById('verdict-banner');
  const verdictIcon = document.getElementById('verdict-icon');
  const verdictTitle = document.getElementById('verdict-title');
  const verdictSubtitle = document.getElementById('verdict-subtitle');
  const probabilityValue = document.getElementById('probability-value');
  const probabilityBar = document.getElementById('probability-bar');
  const detailVerdict = document.getElementById('detail-verdict');
  const detailConfidence = document.getElementById('detail-confidence');
  const resetBtn = document.getElementById('reset-btn');

  // State
  let activeStream = null;
  let currentFile = null;

  // Set this to your external backend API URL if deploying frontend & backend separately
  // e.g. 'https://username-spot-the-fake-photo.hf.space'
  const BACKEND_URL = 'https://bhawyarawal-spot-the-fake-photo.hf.space';



  // -------------------------------------------------------------
  // TAB NAVIGATION
  // -------------------------------------------------------------
  tabUpload.addEventListener('click', () => {
    switchTab('upload');
  });

  tabCamera.addEventListener('click', () => {
    switchTab('camera');
  });

  function switchTab(mode) {
    if (mode === 'upload') {
      tabUpload.classList.add('active');
      tabCamera.classList.remove('active');
      panelUpload.classList.add('active');
      panelCamera.classList.remove('active');
      stopCamera();
    } else {
      tabUpload.classList.remove('active');
      tabCamera.classList.add('active');
      panelUpload.classList.remove('active');
      panelCamera.classList.add('active');
      startCamera();
    }
  }

  // -------------------------------------------------------------
  // DRAG & DROP FILE UPLOAD
  // -------------------------------------------------------------
  browseBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    fileInput.click();
  });

  dropZone.addEventListener('click', () => {
    fileInput.click();
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
      handleImageSelected(e.target.files[0]);
    }
  });

  ['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.add('dragover');
    }, false);
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.remove('dragover');
    }, false);
  });

  dropZone.addEventListener('drop', (e) => {
    const dt = e.dataTransfer;
    const files = dt.files;
    if (files.length > 0) {
      handleImageSelected(files[0]);
    }
  });

  function handleImageSelected(file) {
    if (!file.type.startsWith('image/')) {
      alert('Please select an image file.');
      return;
    }
    currentFile = file;
    
    // Read and preview the image locally
    const reader = new FileReader();
    reader.onload = (e) => {
      showPreview(e.target.result);
      // Run the model check!
      analyzeImage(file);
    };
    reader.readAsDataURL(file);
  }

  // -------------------------------------------------------------
  // LIVE CAMERA API
  // -------------------------------------------------------------
  async function startCamera() {
    stopCamera();
    
    try {
      // Prompt for camera access
      const initialStream = await navigator.mediaDevices.getUserMedia({ video: true });
      initialStream.getTracks().forEach(track => track.stop()); // close prompt stream
      
      // Enumerate cameras
      const devices = await navigator.mediaDevices.enumerateDevices();
      const videoDevices = devices.filter(device => device.kind === 'videoinput');
      
      cameraSelect.innerHTML = '';
      if (videoDevices.length === 0) {
        cameraSelect.innerHTML = '<option value="">No camera found</option>';
        return;
      }
      
      videoDevices.forEach((device, index) => {
        const option = document.createElement('option');
        option.value = device.deviceId;
        option.text = device.label || `Camera ${index + 1}`;
        cameraSelect.appendChild(option);
      });
      
      if (videoDevices.length > 1) {
        cameraSelectContainer.style.display = 'flex';
      } else {
        cameraSelectContainer.style.display = 'none';
      }
      
      // Select the first device
      const selectedDeviceId = videoDevices[0].deviceId;
      await openCameraStream(selectedDeviceId);
      
    } catch (err) {
      console.error('Camera access error:', err);
      alert('Camera access denied or unavailable. Please use the upload feature.');
      switchTab('upload');
    }
  }

  async function openCameraStream(deviceId) {
    stopCamera();
    
    const constraints = {
      video: deviceId ? { deviceId: { exact: deviceId } } : true
    };
    
    try {
      activeStream = await navigator.mediaDevices.getUserMedia(constraints);
      webcam.srcObject = activeStream;
    } catch (err) {
      console.error('Error opening camera stream:', err);
    }
  }

  cameraSelect.addEventListener('change', () => {
    if (cameraSelect.value) {
      openCameraStream(cameraSelect.value);
    }
  });

  function stopCamera() {
    if (activeStream) {
      activeStream.getTracks().forEach(track => track.stop());
      activeStream = null;
    }
    webcam.srcObject = null;
  }

  captureBtn.addEventListener('click', () => {
    if (!activeStream) return;
    
    const width = webcam.videoWidth;
    const height = webcam.videoHeight;
    cameraCanvas.width = width;
    cameraCanvas.height = height;
    
    const ctx = cameraCanvas.getContext('2d');
    ctx.drawImage(webcam, 0, 0, width, height);
    
    // Convert canvas to blob
    cameraCanvas.toBlob((blob) => {
      const file = new File([blob], 'snapshot.jpg', { type: 'image/jpeg' });
      currentFile = file;
      
      // Convert to local URL for preview
      const dataUrl = cameraCanvas.toDataURL('image/jpeg');
      showPreview(dataUrl);
      
      // Run the model check!
      analyzeImage(file);
    }, 'image/jpeg');
  });

  // -------------------------------------------------------------
  // PREVIEW AND SCANNING STATES
  // -------------------------------------------------------------
  function showPreview(src) {
    previewPlaceholder.classList.add('hidden');
    previewImage.src = src;
    previewImage.classList.remove('hidden');
  }

  function clearPreview() {
    previewImage.src = '';
    previewImage.classList.add('hidden');
    previewPlaceholder.classList.remove('hidden');
    scanOverlay.style.display = 'none';
  }

  // -------------------------------------------------------------
  // API PREDICTION PIPELINE
  // -------------------------------------------------------------
  async function analyzeImage(file) {
    // Show scanning lines and prepare results container
    scanOverlay.style.display = 'block';
    resultsPlaceholder.classList.add('hidden');
    resultsContent.classList.add('hidden');
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
      const apiEndpoint = BACKEND_URL ? `${BACKEND_URL}/api/predict` : '/api/predict';
      const response = await fetch(apiEndpoint, {
        method: 'POST',
        body: formData
      });
      
      let data;
      try {
        data = await response.json();
      } catch (e) {
        throw new Error(`HTTP Error: ${response.status}`);
      }
      
      if (!response.ok) {
        throw new Error(data.error || `HTTP Error: ${response.status}`);
      }
      
      displayResults(data);
      
    } catch (err) {
      console.error('Prediction failed:', err);
      displayResults({
        success: false,
        error: err.message || 'An error occurred during analysis.'
      });
    } finally {
      // Turn off scanning overlay
      scanOverlay.style.display = 'none';
    }
  }

  function displayResults(data) {
    resultsPlaceholder.classList.add('hidden');
    resultsContent.classList.remove('hidden');
    
    if (data.error || !data.success) {
      verdictBanner.className = 'verdict-banner verdict-fake';
      verdictIcon.innerHTML = '<i class="fa-solid fa-triangle-exclamation"></i>';
      verdictTitle.innerText = 'Analysis Error';
      verdictSubtitle.innerText = data.error || 'Failed to complete analysis';
      
      probabilityValue.innerText = '0%';
      probabilityBar.style.width = '0%';
      probabilityBar.style.backgroundColor = 'var(--fake-color)';
      
      detailVerdict.innerText = 'Error';
      detailConfidence.innerText = '0.00%';
      return;
    }
    
    const probPercent = Math.round(data.fake_probability * 100);
    probabilityValue.innerText = `${probPercent}%`;
    probabilityBar.style.width = `${probPercent}%`;
    
    if (data.is_fake) {
      verdictBanner.className = 'verdict-banner verdict-fake';
      verdictIcon.innerHTML = '<i class="fa-solid fa-triangle-exclamation"></i>';
      verdictTitle.innerText = 'Verdict: SPOOF DETECTED';
      verdictSubtitle.innerText = 'High probability of photo recapture or editing.';
      probabilityBar.style.backgroundColor = 'var(--fake-color)';
      
      detailVerdict.innerText = 'FAKE (Recapture)';
      detailConfidence.innerText = `${(data.confidence * 100).toFixed(2)}%`;
    } else {
      verdictBanner.className = 'verdict-banner verdict-real';
      verdictIcon.innerHTML = '<i class="fa-solid fa-circle-check"></i>';
      verdictTitle.innerText = 'Verdict: GENUINE';
      verdictSubtitle.innerText = 'Image presents natural pixel-level profiles.';
      probabilityBar.style.backgroundColor = 'var(--real-color)';
      
      detailVerdict.innerText = 'REAL (Genuine)';
      detailConfidence.innerText = `${(data.confidence * 100).toFixed(2)}%`;
    }
  }

  // -------------------------------------------------------------
  // RESET
  // -------------------------------------------------------------
  resetBtn.addEventListener('click', () => {
    currentFile = null;
    clearPreview();
    resultsContent.classList.add('hidden');
    resultsPlaceholder.classList.remove('hidden');
    
    // If we're on the camera tab, make sure the feed is working
    if (tabCamera.classList.contains('active')) {
      startCamera();
    }
  });
});
