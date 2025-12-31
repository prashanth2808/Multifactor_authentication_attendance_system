# Face + Voice Attendance / Login-Logout System (CLI)

This project is a **biometric login/logout + attendance system** that verifies a user using:

- **Face recognition** (ArcFace embedding + cosine similarity)
- **Voice verification** (SpeechBrain ECAPA-TDNN speaker embedding + cosine similarity)

It is implemented as a **command-line application (CLI)** using **Typer** and stores data in **MongoDB**.

---

## Features

- Register a user with **3 face photos** + **3 voice clips**
- Verify user using **Face + Voice**
- Track **login/logout sessions** with a strict rule:
  - Login → no logout for **< 9 hours** → still Present (grace)
  - Login → no logout for **≥ 9 hours** → Absent (user fault)
  - If already marked today → Malpractice protection
- Reports:
  - View daily login/logout report
  - Admin dashboard commands (users/today/logs/export)

---

## Project structure (important folders)

- `main.py` – CLI entry point
- `cli/` – CLI commands (register/session/report/admin)
- `services/` – Face/voice embedding + verification logic
- `db/` – MongoDB repositories (users, sessions)
- `config/` – `.env` configuration loader
- `captured_voices/` – 1 best voice WAV saved per registration
- `voice_backups/` – 3 VAD-cleaned voice clips per user (WAV) saved for future re-training/re-embedding

---

## Requirements

### System

- Python 3.10+ (recommended)
- Webcam
- Microphone
- MongoDB (local or Atlas)

### Python packages

Installed from `requirements.txt`.

---

## Setup (step-by-step)

### 1) Create and activate a virtual environment

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2) Install dependencies

```powershell
pip install -r requirements.txt
```

> Note: `torch/torchaudio` installs can vary by system.
> If you face installation errors, install PyTorch from the official site and then run `pip install -r requirements.txt` again.

### 3) Start MongoDB

If using local MongoDB:

- Make sure MongoDB service is running
- Default URI used by this project: `mongodb://localhost:27017`

### 4) Configure environment variables (`.env`)

Edit the file `.env` in the project root:

```env
MONGODB_URI=mongodb://localhost:27017
DB_NAME=face_attendance
SIMILARITY_THRESHOLD=0.62
MIN_PHOTOS=3
LIVENESS_REQUIRED=false
LOG_LEVEL=INFO
```

### 5) Voice model files (ECAPA)

Voice verification loads SpeechBrain ECAPA model from:

```
pretrained_models/spkrec-ecapa-voxceleb/
```

If those files are missing, voice initialization will fail with a message like:

> "Run download_ecapa_manual.py first!"

Make sure the ECAPA folder exists and contains the checkpoint/config files.

---

## How it works (end-to-end)

### A) Registration (enroll a user)

Command:

```powershell
python main.py register --name "Your Name" --email "you@example.com"
```

What it does:

1. Captures **3 face photos** via webcam
2. Generates **3 face embeddings** and stores them in MongoDB (`users.face_embeddings`)
3. Records **3 voice clips**
4. Runs VAD (Silero) to keep only speech
5. Generates **3 voice embeddings** (ECAPA) and stores the **average embedding** in MongoDB (`users.voice_embedding`)
6. Saves audio files:
   - One best clip to `captured_voices/<name>_<timestamp>_voice.wav`
   - All 3 cleaned clips to `voice_backups/user_<readable_id>/...`

#### Voice backup folder naming

For easy identification, the backup folder id is now created like:

```
<SafeName>_<8-char-email-hash>
```

Example:

```
voice_backups/user_Prashanth_S_d4c74594/
  Prashanth_S_d4c74594_clip_1_20251229_174510.wav
  Prashanth_S_d4c74594_clip_2_20251229_174510.wav
  Prashanth_S_d4c74594_clip_3_20251229_174510.wav
```

---

### B) Login/Logout Session (daily attendance logic)

Command:

```powershell
python main.py session
```

What it does:

1. Continuously reads frames from webcam
2. Detects + crops face, generates a face embedding
3. Finds best match in DB using cosine similarity
4. If face matches, asks user to proceed with voice verification
5. Records voice, computes live embedding, compares with stored embedding
6. If both pass, it updates today’s session:
   - First time today → LOGIN
   - Second time (within 9 hours) → LOGOUT
   - If already completed today → MALPRACTICE
   - If user forgot logout for 9+ hours → AUTO ABSENT

---

### C) View report

Today:

```powershell
python main.py report --today
```

Specific date:

```powershell
python main.py report --date 2025-12-29
```

---

### D) Admin commands

List users:

```powershell
python main.py admin users
```

Today summary:

```powershell
python main.py admin today
```

Logs for a date:

```powershell
python main.py admin logs --date 2025-12-29
```

Export CSV:

```powershell
python main.py admin export --date 2025-12-29 --file attendance_report.csv
```

---

## Data stored in MongoDB

### `users` collection

Key fields (typical):

- `name`, `email`
- `face_embeddings`: list of 3 embeddings (each embedding is a list of floats)
- `voice_embedding`: single embedding vector (list of floats)
- `voice_audio_path`: path to the best saved wav in `captured_voices/`
- `voice_backup_paths`: list of 3 backup wav paths in `voice_backups/`
- `backup_user_id`: the readable folder id used under `voice_backups/`

### `sessions` collection

- `user_id`, `name`, `email`
- `login_time`, `logout_time`
- `duration_minutes`
- `status` (`active`, `present`, `absent_fault`)
- `date` (`YYYY-MM-DD`)

---

## Troubleshooting

### 1) MongoDB connection failed

- Check MongoDB is running
- Verify `.env` → `MONGODB_URI`

### 2) Webcam not opening

- Close other apps using the camera
- Try changing camera index in code (0 → 1)

### 3) No speech detected (VAD)

- Speak louder / closer to mic
- Reduce background noise

### 4) ECAPA model missing

- Ensure `pretrained_models/spkrec-ecapa-voxceleb/` exists and contains:
  - `hyperparams.yaml`
  - `embedding_model.ckpt`
  - `classifier.ckpt`
  - etc.

### 5) Torch hub download issues (Silero VAD)

- It loads via `torch.hub.load('snakers4/silero-vad', ...)`
- Ensure internet access on first run

---

## Notes / Known limitations

- `cli/attendance.py` exists but it references `db.attendance_repo`, which is not present in this v2.2 folder. If you want attendance logging separate from sessions, we can add that repo + wire an `attendance` command into `main.py`.

python main.py register --name "Your Name" --email "you@example.com"
python main.py session
python main.py report --today
python main.py admin today
```
