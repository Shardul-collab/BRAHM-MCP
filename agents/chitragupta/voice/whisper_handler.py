# voice/whisper_handler.py

"""
Voice Input Layer — microphone capture + local Whisper transcription.

Two recording modes:
    capture_and_transcribe() → long clip for log entries
    capture_command()        → short clip for menu / confirmation responses

Jarvis additions (v0.3)
-----------------------
- _is_hallucination(): rejects transcripts that contain non-Latin characters
  above a 15% ratio, known Whisper training-data phrases, or implausible
  word repetition.  Raises TranscriptionEmptyError so callers retry rather
  than storing garbage in a Notion field.
- Beep tone presets: play_listen_beep() / play_confirm_beep() / play_error_beep()
  give each context a distinct audio signature (high/mid/low pitch).
- _record() now calls play_listen_beep() instead of the raw _play_beep() so
  all voice captures get the standard listening tone.

Earlier fixes (v0.2) preserved:
- Model warmup on first load.
- Empty transcript guard + retry (×2).
- TranscriptionEmptyError separate from SilenceError.
"""

import logging
import re
import threading
import numpy as np

import config.settings as cfg

logger = logging.getLogger("chitragupta.whisper_handler")


# ── Whisper model singleton ───────────────────────────────────────────────────

_whisper_model = None
_model_lock    = threading.Lock()


def _get_model():
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model
    with _model_lock:
        if _whisper_model is not None:
            return _whisper_model
        try:
            import whisper
            logger.info("Loading Whisper model '%s' ...", cfg.WHISPER_MODEL)
            _whisper_model = whisper.load_model(cfg.WHISPER_MODEL)
            logger.info("Whisper model '%s' loaded.", cfg.WHISPER_MODEL)
            _warmup_model(_whisper_model)
        except Exception as exc:
            raise WhisperLoadError(
                f"Failed to load Whisper model '{cfg.WHISPER_MODEL}': {exc}"
            ) from exc
    return _whisper_model


# ── Model warmup ──────────────────────────────────────────────────────────────

def _warmup_model(model) -> None:
    try:
        dummy = np.zeros(cfg.AUDIO_SAMPLE_RATE, dtype="float32")
        model.transcribe(dummy, fp16=False, language=None, verbose=False)
        logger.info("Whisper model warmed up successfully.")
    except Exception as exc:
        logger.warning("Whisper warmup skipped (non-fatal): %s", exc)


# ── Hallucination filter ──────────────────────────────────────────────────────

# Characters outside Latin / Extended-Latin blocks
_NON_LATIN_RE = re.compile(r"[^\x00-\x7F\u00C0-\u024F\u1E00-\u1EFF\s]")

# Phrases Whisper hallucinates from its subtitle training data
_HALLUCINATION_PHRASES: frozenset[str] = frozenset({
    "thank you for watching",
    "thanks for watching",
    "please subscribe",
    "subtitles by",
    "transcribed by",
    "www.",
    "http",
    "copyright",
    "all rights reserved",
    "♪", "♫",
})


def _is_hallucination(text: str) -> bool:
    """
    Return True if the transcript shows hallucination signals.

    Three checks (any one triggers rejection):
    1. Non-Latin character ratio > 15 % — catches Chinese, Arabic, etc.
       that Whisper produces on near-silence with a foreign-language bias.
    2. Known hallucination phrases from Whisper's subtitle training data.
    3. Implausible word repetition — same word (len > 2) appears > 4 times.
    """
    if not text:
        return False

    # 1. Non-Latin ratio
    non_latin = len(_NON_LATIN_RE.findall(text))
    if non_latin / max(len(text), 1) > 0.15:
        logger.warning("Hallucination detected: non-Latin ratio=%.2f in '%s'", non_latin / len(text), text[:60])
        return True

    # 2. Known phrases
    lowered = text.lower()
    if any(phrase in lowered for phrase in _HALLUCINATION_PHRASES):
        logger.warning("Hallucination detected: known phrase in '%s'", text[:60])
        return True

    # 3. Repetition
    words = re.findall(r"\b\w+\b", lowered)
    for word in set(words):
        if len(word) > 2 and words.count(word) > 4:
            logger.warning("Hallucination detected: '%s' repeated %d times", word, words.count(word))
            return True

    return False


# ── Beep tones ────────────────────────────────────────────────────────────────

def _play_beep(frequency: int = 880, duration: float = 0.18) -> None:
    """Play a short sine-wave beep. Visual fallback on audio failure."""
    try:
        import sounddevice as sd
        sr  = cfg.AUDIO_SAMPLE_RATE
        t   = np.linspace(0, duration, int(sr * duration), endpoint=False)
        env = np.linspace(1.0, 0.0, len(t))
        beep = (0.35 * np.sin(2 * np.pi * frequency * t) * env).astype("float32")
        sd.play(beep, samplerate=sr)
        sd.wait()
    except Exception as exc:
        logger.debug("_play_beep failed (non-fatal): %s", exc)
        print("  ▶  ", end="", flush=True)


def play_listen_beep() -> None:
    """High-pitch beep — mic is now open, please speak."""
    _play_beep(frequency=880, duration=0.18)


def play_confirm_beep() -> None:
    """Mid-pitch beep — confirmation / success moment."""
    _play_beep(frequency=660, duration=0.15)


def play_error_beep() -> None:
    """Low-pitch beep — error or retry needed."""
    _play_beep(frequency=440, duration=0.12)


# ── Silence check ─────────────────────────────────────────────────────────────

def _check_silence(audio: np.ndarray, threshold: float = 0.01) -> None:
    rms = float(np.sqrt(np.mean(audio ** 2)))
    logger.debug("Audio RMS: %.5f (threshold: %.3f)", rms, threshold)
    if rms < threshold:
        raise SilenceError(
            f"No speech detected (RMS={rms:.5f}). Please speak clearly and try again."
        )


# ── Core recording ────────────────────────────────────────────────────────────

def _record(duration: int, label: str = "Recording") -> np.ndarray:
    """Record `duration` seconds. Plays listen beep before opening mic."""
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise MicrophoneError(
            "sounddevice is not installed. Run: pip install sounddevice"
        ) from exc

    sr = cfg.AUDIO_SAMPLE_RATE
    print(f"\n  🎙  {label} ({duration}s) — speak after the beep …")
    play_listen_beep()

    try:
        raw: np.ndarray = sd.rec(
            frames=int(sr * duration),
            samplerate=sr,
            channels=cfg.AUDIO_CHANNELS,
            dtype="float32",
            blocking=True,
        )
    except Exception as exc:
        raise MicrophoneError(
            f"Microphone error: {exc}. Check that a microphone is connected."
        ) from exc

    print("  ✓  Done.\n")
    return raw.flatten()


def _transcribe(audio: np.ndarray) -> str:
    """
    Transcribe audio array via Whisper.

    Raises TranscriptionEmptyError on blank or hallucinated transcripts
    so callers retry rather than storing garbage.
    """
    model = _get_model()
    try:
        result: dict = model.transcribe(
            audio, fp16=False, language=None, verbose=False,
        )
    except Exception as exc:
        raise TranscriptionError(f"Whisper transcription failed: {exc}") from exc

    transcript = result.get("text", "").strip()
    logger.info("Transcript: '%s'", transcript)

    if not transcript:
        raise TranscriptionEmptyError(
            "Whisper returned an empty transcript. Please speak more clearly."
        )

    if _is_hallucination(transcript):
        raise TranscriptionEmptyError(
            f"Transcript appears to be hallucinated: '{transcript[:60]}'. "
            "Please try again."
        )

    return transcript


# ── Public: long entry recording ──────────────────────────────────────────────

def record_audio() -> np.ndarray:
    audio = _record(cfg.AUDIO_RECORD_SECONDS, label="Recording entry")
    _check_silence(audio, threshold=0.01)
    return audio


def transcribe_audio(audio: np.ndarray) -> str:
    return _transcribe(audio)


def capture_and_transcribe() -> str:
    """Record → silence check → transcribe. Retries once on empty."""
    for attempt in range(1, 3):
        try:
            audio      = record_audio()
            transcript = _transcribe(audio)
            return transcript
        except TranscriptionEmptyError:
            if attempt == 1:
                print("  ⚠  Didn't catch that — please try again.\n")
                logger.warning("capture_and_transcribe: empty/hallucinated, retrying.")
            else:
                raise
    return ""


# ── Public: short command recording ──────────────────────────────────────────

def capture_command(prompt: str = "", timeout: int = 5) -> str:
    """
    Record a short voice command. Retries once on empty/hallucinated result.
    Returns lowercased transcript or "" on failure.
    """
    if prompt:
        print(f"\n  🎙  {prompt}")

    for attempt in range(1, 3):
        try:
            audio      = _record(timeout, label="Listening")
            _check_silence(audio, threshold=0.005)
            transcript = _transcribe(audio).lower().strip()
            print(f"  ✓  Heard: '{transcript}'")
            return transcript

        except TranscriptionEmptyError:
            if attempt == 1:
                print("  ⚠  Didn't catch that — please try again.\n")
                logger.warning("capture_command: empty/hallucinated, retrying (attempt %d).", attempt)
            else:
                return ""
        except SilenceError:
            return ""
        except (MicrophoneError, WhisperLoadError, TranscriptionError) as exc:
            logger.warning("capture_command failed: %s", exc)
            return ""
        except Exception as exc:
            logger.warning("capture_command unexpected error: %s", exc)
            return ""

    return ""


# ── Fallback: manual text entry ───────────────────────────────────────────────

def manual_input(prompt: str = "Type your input: ") -> str:
    try:
        text = input(f"\n  ⌨  {prompt}").strip()
        logger.info("Manual input: '%s'", text)
        return text
    except (EOFError, KeyboardInterrupt):
        return ""


# ── Custom exceptions ─────────────────────────────────────────────────────────

class WhisperLoadError(Exception):
    """Raised when the Whisper model cannot be loaded."""

class MicrophoneError(Exception):
    """Raised when audio cannot be captured from the microphone."""

class SilenceError(Exception):
    """Raised when recorded audio contains no detectable speech."""

class TranscriptionError(Exception):
    """Raised when Whisper fails during inference."""

class TranscriptionEmptyError(Exception):
    """Raised when Whisper returns a blank or hallucinated transcript."""
