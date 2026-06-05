"""
YouTube Summary Telegram Bot
- 자막 있으면: youtube-transcript-api v1.x → Gemini 텍스트 요약
- 자막 없으면: Gemini 2.0 Flash multimodal (YouTube URL 직접)
- GitHub Actions polling 방식 (5분마다 실행)

검증된 버전:
  google-genai >= 0.8
  youtube-transcript-api == 1.2.4
  python-telegram-bot 없음 (requests로 직접 Telegram Bot API 호출)
"""

import os
import re
import time
import logging
import requests
from google import genai
from google.genai import types
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import CouldNotRetrieveTranscript

# ── 로깅 ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── 환경변수 ──────────────────────────────────────────────
TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
GEMINI_API_KEY  = os.environ["GEMINI_API_KEY"]
OFFSET_FILE     = "offset.txt"
POLL_TIMEOUT    = 30          # long-polling 초
RUN_DURATION    = 240         # GitHub Actions 1회 실행 최대 4분

TELEGRAM_API    = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"
MODEL_ID        = "gemini-2.0-flash"

# ── Gemini 클라이언트 ──────────────────────────────────────
gemini = genai.Client(api_key=GEMINI_API_KEY)

# ── YouTube URL 파싱 ──────────────────────────────────────
_YT_RE = re.compile(r"(?:v=|youtu\.be/|shorts/)([A-Za-z0-9_-]{11})")

def extract_video_id(text: str) -> str | None:
    m = _YT_RE.search(text)
    return m.group(1) if m else None

def normalize_url(text: str, video_id: str) -> str:
    """메시지에서 온전한 YouTube URL을 추출하거나 표준 URL 반환"""
    url_match = re.search(r"https?://(?:www\.)?(?:youtube\.com|youtu\.be)\S+", text)
    if url_match:
        return url_match.group(0)
    return f"https://www.youtube.com/watch?v={video_id}"

# ── 자막 추출 (youtube-transcript-api v1.x) ───────────────
PREFERRED_LANGS = ["ko", "en", "ja", "zh-Hans", "zh-Hant"]
_ytt = YouTubeTranscriptApi()

def get_transcript(video_id: str) -> str | None:
    """
    1순위: 선호 언어 수동 자막
    2순위: 선호 언어 자동 생성 자막
    3순위: 사용 가능한 첫 번째 자막
    """
    try:
        fetched = _ytt.fetch(video_id, languages=PREFERRED_LANGS)
        return " ".join(s.text for s in fetched)
    except CouldNotRetrieveTranscript:
        pass
    except Exception as e:
        log.warning(f"fetch 1차 실패: {e}")

    # fallback: TranscriptList에서 직접 탐색
    try:
        tl = _ytt.list(video_id)
        # 자동 생성 자막 시도
        try:
            t = tl.find_generated_transcript(PREFERRED_LANGS)
            fetched = t.fetch()
            return " ".join(s.text for s in fetched)
        except Exception:
            pass
        # 수동 자막 시도
        try:
            t = tl.find_manually_created_transcript(PREFERRED_LANGS)
            fetched = t.fetch()
            return " ".join(s.text for s in fetched)
        except Exception:
            pass
        # 언어 무관 첫 번째 자막
        try:
            t = tl.find_transcript(["ko", "en", "ja", "zh", "fr", "de", "es", "pt"])
            fetched = t.fetch()
            return " ".join(s.text for s in fetched)
        except Exception:
            pass
    except Exception as e:
        log.warning(f"TranscriptList 탐색 실패: {e}")

    return None

# ── Gemini 요약: 텍스트 기반 ──────────────────────────────
_PROMPT_TEXT = """\
아래는 YouTube 영상의 자막 텍스트입니다.
한국어로 다음 형식에 맞게 요약해주세요:

📌 **핵심 주제** (1줄)
📝 **주요 내용** (3~5개 불릿)
💡 **핵심 인사이트** (1~2줄)

자막:
{transcript}
"""

def summarize_text(transcript: str) -> str:
    trimmed = transcript[:12000]  # 토큰 절약
    response = gemini.models.generate_content(
        model=MODEL_ID,
        contents=_PROMPT_TEXT.format(transcript=trimmed),
    )
    return response.text

# ── Gemini 요약: 멀티모달 (YouTube URL 직접) ──────────────
_PROMPT_MULTIMODAL = """\
이 YouTube 영상을 보고 한국어로 다음 형식에 맞게 요약해주세요:

📌 **핵심 주제** (1줄)
📝 **주요 내용** (3~5개 불릿)
💡 **핵심 인사이트** (1~2줄)
"""

def summarize_multimodal(video_url: str) -> str:
    contents = [
        types.Part.from_uri(file_uri=video_url, mime_type="video/*"),
        types.Part(text=_PROMPT_MULTIMODAL),
    ]
    response = gemini.models.generate_content(
        model=MODEL_ID,
        contents=contents,
    )
    return response.text

# ── 통합 요약 ─────────────────────────────────────────────
def summarize(raw_text: str, video_id: str) -> str:
    transcript = get_transcript(video_id)
    if transcript:
        log.info(f"[{video_id}] 자막 {len(transcript)}자 → 텍스트 요약")
        return summarize_text(transcript)
    else:
        url = normalize_url(raw_text, video_id)
        log.info(f"[{video_id}] 자막 없음 → 멀티모달 요약 ({url})")
        return summarize_multimodal(url)

# ── Telegram API 헬퍼 ─────────────────────────────────────
def tg_get(method: str, **params) -> dict:
    r = requests.get(f"{TELEGRAM_API}/{method}", params=params, timeout=POLL_TIMEOUT + 5)
    r.raise_for_status()
    return r.json()

def tg_send(chat_id: int, text: str) -> None:
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "Markdown",
    }
    try:
        r = requests.post(f"{TELEGRAM_API}/sendMessage", json=payload, timeout=15)
        if not r.ok:
            # Markdown 파싱 실패 시 plain text로 재시도
            payload["parse_mode"] = None
            requests.post(f"{TELEGRAM_API}/sendMessage", json=payload, timeout=15)
    except Exception as e:
        log.error(f"sendMessage 실패: {e}")

# ── offset 관리 ───────────────────────────────────────────
def load_offset() -> int:
    if os.path.exists(OFFSET_FILE):
        with open(OFFSET_FILE) as f:
            val = f.read().strip()
            return int(val) if val.isdigit() else 0
    return 0

def save_offset(offset: int) -> None:
    with open(OFFSET_FILE, "w") as f:
        f.write(str(offset))

# ── 메인 polling 루프 ─────────────────────────────────────
def run() -> None:
    offset = load_offset()
    log.info(f"봇 시작 (offset={offset}, 실행제한={RUN_DURATION}s)")

    deadline = time.time() + RUN_DURATION

    while time.time() < deadline:
        try:
            data = tg_get("getUpdates", offset=offset, timeout=POLL_TIMEOUT)
        except requests.exceptions.Timeout:
            continue
        except Exception as e:
            log.error(f"getUpdates 오류: {e}")
            time.sleep(5)
            continue

        updates = data.get("result", [])
        for update in updates:
            offset = update["update_id"] + 1
            save_offset(offset)

            msg  = update.get("message", {})
            chat_id = msg.get("chat", {}).get("id")
            text = msg.get("text", "")

            if not chat_id or not text:
                continue

            # /start 명령
            if text.strip().startswith("/start"):
                tg_send(chat_id,
                    "👋 *YouTube 요약 봇*에 오신 걸 환영합니다!\n\n"
                    "YouTube URL을 보내주시면 한국어로 요약해드립니다.\n"
                    "자막이 없는 영상도 AI가 직접 분석합니다. 🎬"
                )
                continue

            video_id = extract_video_id(text)
            if not video_id:
                continue  # 유튜브 URL 아닌 메시지는 무시

            tg_send(chat_id, "⏳ 분석 중입니다. 잠시만 기다려주세요...")

            try:
                summary = summarize(text, video_id)
                tg_send(chat_id, f"🎬 *YouTube 요약*\n\n{summary}")
                log.info(f"[{video_id}] 요약 전송 완료")
            except Exception as e:
                log.error(f"[{video_id}] 요약 실패: {e}")
                tg_send(chat_id,
                    f"❌ 요약에 실패했습니다.\n"
                    f"원인: {str(e)[:200]}\n\n"
                    "비공개 영상이거나 지역 제한이 있을 수 있습니다."
                )

    save_offset(offset)
    log.info(f"봇 종료 (offset={offset})")

if __name__ == "__main__":
    run()
