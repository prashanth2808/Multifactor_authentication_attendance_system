const $ = (id) => document.getElementById(id);

let videoStream = null;
let audioStream = null;
let audioContext = null;

let faceBlob = null;
let voiceBlob = null;

async function startCameraAndMic() {
  videoStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
  $('video').srcObject = videoStream;

  // For speaker verification, disable browser DSP features as they can distort speaker characteristics.
  audioStream = await navigator.mediaDevices.getUserMedia({
    audio: {
      channelCount: 1,
      echoCancellation: false,
      noiseSuppression: false,
      autoGainControl: false,
    },
    video: false,
  });

  audioContext = new (window.AudioContext || window.webkitAudioContext)();
  try { await audioContext.resume(); } catch (_) {}
}

function captureFaceFrame() {
  const video = $('video');
  const canvas = $('canvas');
  const ctx = canvas.getContext('2d');
  canvas.width = video.videoWidth || 640;
  canvas.height = video.videoHeight || 480;
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (!blob) return reject(new Error('Failed to capture'));
      resolve(blob);
    }, 'image/jpeg', 0.9);
  });
}

// Record raw PCM and encode as WAV 16kHz mono
async function recordWav(seconds = 5) {
  if (!audioStream || !audioContext) throw new Error('Mic not started');

  const source = audioContext.createMediaStreamSource(audioStream);
  const processor = audioContext.createScriptProcessor(4096, 1, 1);
  const nullGain = audioContext.createGain();
  nullGain.gain.value = 0;

  const chunks = [];
  processor.onaudioprocess = (e) => {
    const input = e.inputBuffer.getChannelData(0);
    chunks.push(new Float32Array(input));
  };

  source.connect(processor);
  processor.connect(nullGain);
  nullGain.connect(audioContext.destination);

  await new Promise((r) => setTimeout(r, seconds * 1000));

  source.disconnect();
  processor.disconnect();

  return encodeWav(chunks, audioContext.sampleRate, 16000);
}

function encodeWav(float32Chunks, inputSampleRate, targetSampleRate) {
  // Flatten
  let length = 0;
  for (const c of float32Chunks) length += c.length;
  const data = new Float32Array(length);
  let offset = 0;
  for (const c of float32Chunks) { data.set(c, offset); offset += c.length; }

  // Resample (linear)
  const ratio = inputSampleRate / targetSampleRate;
  const newLen = Math.floor(data.length / ratio);
  const resampled = new Float32Array(newLen);
  for (let i = 0; i < newLen; i++) {
    const x = i * ratio;
    const x0 = Math.floor(x);
    const x1 = Math.min(x0 + 1, data.length - 1);
    const t = x - x0;
    resampled[i] = data[x0] * (1 - t) + data[x1] * t;
  }

  // Normalize a bit (helps VAD)
  let sumSq = 0;
  for (let i = 0; i < resampled.length; i++) sumSq += resampled[i] * resampled[i];
  const rms = Math.sqrt(sumSq / Math.max(1, resampled.length));
  const targetRms = 0.05;
  let gain = (rms > 1e-8) ? (targetRms / rms) : 1.0;
  gain = Math.min(gain, 10.0);

  const pcm16 = new Int16Array(resampled.length);
  for (let i = 0; i < resampled.length; i++) {
    const s0 = resampled[i] * gain;
    const s = Math.max(-1, Math.min(1, s0));
    pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
  }

  const header = new ArrayBuffer(44);
  const view = new DataView(header);
  const writeStr = (off, str) => { for (let i = 0; i < str.length; i++) view.setUint8(off+i, str.charCodeAt(i)); };
  writeStr(0, 'RIFF');
  view.setUint32(4, 36 + pcm16.byteLength, true);
  writeStr(8, 'WAVE');
  writeStr(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, targetSampleRate, true);
  view.setUint32(28, targetSampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeStr(36, 'data');
  view.setUint32(40, pcm16.byteLength, true);

  return new Blob([header, pcm16.buffer], { type: 'audio/wav' });
}

async function verify() {
  if (!faceBlob) throw new Error('Capture face first');
  if (!voiceBlob) throw new Error('Record voice first');

  const fd = new FormData();
  fd.append('face', faceBlob, 'face.jpg');
  fd.append('voice', voiceBlob, 'voice.wav');

  const res = await fetch('/api/session/auto', { method: 'POST', body: fd });
  const text = await res.text();
  let json = null;
  try {
    json = JSON.parse(text);
  } catch (_) {
    json = { ok: false, error: 'Non-JSON response from server', raw: text.slice(0, 200) };
  }
  return { res, json };
}

$('btnStart').addEventListener('click', async () => {
  try {
    $('camStatus').textContent = 'Camera: requesting permission…';
    await startCameraAndMic();
    $('camStatus').textContent = 'Camera: ready';
    $('btnCapture').disabled = false;
    $('btnRecord').disabled = false;
    $('btnVerify').disabled = false;
  } catch (e) {
    $('camStatus').textContent = 'Camera/Mic error: ' + String(e);
  }
});

$('btnCapture').addEventListener('click', async () => {
  try {
    faceBlob = await captureFaceFrame();
    $('faceStatus').textContent = `Face: captured (${Math.round(faceBlob.size/1024)} KB)`;
  } catch (e) {
    $('faceStatus').textContent = 'Face error: ' + String(e);
  }
});

$('btnRecord').addEventListener('click', async () => {
  try {
    $('voiceStatus').textContent = 'Voice: recording…';
    voiceBlob = await recordWav(8);
    $('voiceStatus').textContent = `Voice: recorded (${Math.round(voiceBlob.size/1024)} KB)`;
    const a = $('playback');
    a.style.display = 'block';
    a.src = URL.createObjectURL(voiceBlob);
  } catch (e) {
    $('voiceStatus').textContent = 'Voice error: ' + String(e);
  }
});

$('btnVerify').addEventListener('click', async () => {
  try {
    $('result').textContent = 'Result: verifying…';
    const { res, json } = await verify();

    if (!res.ok || json.ok === false) {
      $('result').textContent = `Result: server error (${res.status}) ${json.error || ''}`;
      return;
    }

    if (json.verified) {
      const action = json.session?.action || 'OK';
      const msg = json.session?.message || '';
      $('result').textContent = `Result: VERIFIED — ${action} — ${msg}`;
    } else {
      $('result').textContent = `Result: DENIED — ${json.reason || 'failed'}`;
    }
  } catch (e) {
    $('result').textContent = 'Result error: ' + String(e);
  }
});
