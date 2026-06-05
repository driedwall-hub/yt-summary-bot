"""
YouTube 자동 요약 봇
- 특정 채널 새 영상 감지 (YouTube RSS, API 키 불필요)
- Gemini 2.0 Flash로 요약 (자막 있으면 텍스트, 없으면 멀티모달)
- 텔레그램으로 내 Chat ID에 자동 DM
- GitHub Actions 5분마다 실행
"""

import os
import re
import json
import time
import logging
import requests
import xml.etree.ElementTree as ET
from google import genai
from google.genai import types
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import CouldNotRetrieveTranscript

# ── 수정 1: logging format 오류 수정 ──────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── 환경변수 ──────────────────────────────────────────────
TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
CHAT_ID        = os.environ["TELEGRAM_CHAT_ID"]
SEEN_FILE      = "seen_videos.json"

TELEGRAM_API   = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"
MODEL_ID       = "gemini-2.0-flash"

# ── 모니터링 채널 ─────────────────────────────────────────
CHANNELS = [
    "@sosumonkey",
]

# ── Gemini / Transcript 클라이언트 ────────────────────────
gemini = genai.Client(api_key=GEMINI_API_KEY)
_ytt   = YouTubeTranscriptApi()

# ── seen_videos 관리 ──────────────────────────────────────
def load_seen() -> set:
    if os.path.exists(SEEN_FILE):
        with open(SEEN_FILE) as f:
            return set(json.load(f))
    return set()

def save_seen(seen: set):
    with open(SEEN_FILE, "w") as f:
        json.dump(list(seen), f)

# ── channel_id 조회 ───────────────────────────────────────
def resolve_channel_id(channel: str) -> str | None:
    if channel.startswith("UC"):
        return channel
    handle = channel.lstrip("@")
    url = f"https://www.youtube.com/@{handle}"
    try:
        r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        for pat in [
            r'youtube\.com/channel/(UC[a-zA-Z0-9_-]+)',
            r'"channelId":"(UC[a-zA-Z0-9_-]+)"',
            r'"externalId":"(UC[a-zA-Z0-9_-]+)"',
            r'href="/channel/(UC[a-zA-Z0-9_-]+)"',
        ]:
            m = re.search(pat, r.text)
            if m:
                return m.group(1)
    except Exception as e:
        log.error(f"channel_id 조회 실패 ({channel}): {e}")
    return None

# ── YouTube RSS 파싱 ──────────────────────────────────────
NS = {
    "atom":  "http://www.w3.org/2005/Atom",
    "yt":    "http://www.youtube.com/xml/schemas/2015",
    "media": "http://search.yahoo.com/mrss/",
}

def fetch_latest_videos(channel_id: str, max_count: int = 5) -> list[dict]:
    url = f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"
    try:
        r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        root = ET.fromstring(r.content)
        videos = []
        for entry in root.findall("atom:entry", NS)[:max_count]:
            vid_el   = entry.find("yt:videoId", NS)
            title_el = entry.find("atom:title", NS)
            pub_el   = entry.find("atom:published", NS)
            if vid_el is None or title_el is None:
                continue
            videos.append({
                "id":        vid_el.text,
                "title":     title_el.text,
                "published": pub_el.text if pub_el is not None else "",
                "url":       f"https://www.youtube.com/watch?v={vid_el.text}",
            })
        return videos
    except Exception as e:
        log.error(f"RSS 파싱 실패 ({channel_id}): {e}")
        return []

# ── 자막 추출 ─────────────────────────────────────────────
LANGS = ["ko", "en", "ja", "zh-Hans", "zh-Hant"]

def get_transcript(video_id: str) -> str | None:
    try:
        fetched = _ytt.fetch(video_id, languages=LANGS)
        return " ".join(s.text for s in fetched)
    except CouldNotRetrieveTranscript:
        pass
    except Exception:
        pass
    try:
        tl = _ytt.list(video_id)
        for finder in [
            lambda: tl.find_generated_transcript(LANGS),
            lambda: tl.find_manually_created_transcript(LANGS),
        ]:
            try:
                return " ".join(s.text for s in finder().fetch())
            except Exception:
                pass
    except Exception:
        pass
    return None

# ── Gemini 요약 ───────────────────────────────────────────
_PROMPT_TEXT = """\
아래는 YouTube 영상의 자막입니다. 한국어로 요약해주세요.

📌 **핵심 주제** (1줄)
📝 **주요 내용** (3~5개 불릿)
💡 **핵심 인사이트** (1~2줄)

자막:
{transcript}
"""

_PROMPT_MULTI = """\
이 YouTube 영상을 보고 한국어로 요약해주세요.

📌 **핵심 주제** (1줄)
📝 **주요 내용** (3~5개 불릿)
💡 **핵심 인사이트** (1~2줄)
"""

def summarize(video: dict) -> str:
    transcript = get_transcript(video["id"])
    if transcript:
        log.info(f"자막 {len(transcript)}자 → 텍스트 요약")
        resp = gemini.models.generate_content(
            model=MODEL_ID,
            contents=_PROMPT_TEXT.format(transcript=transcript[:12000]),
        )
    else:
        log.info("자막 없음 → 멀티모달 요약")
        resp = gemini.models.generate_content(
            model=MODEL_ID,
            contents=[
                types.Part.from_uri(file_uri=video["url"], mime_type="video/*"),
                types.Part(text=_PROMPT_MULTI),
            ],
        )
    return resp.text

# ── 수정 2: tg_send - parse_mode None 키 제거 ─────────────
def tg_send(text: str, parse_mode: str = "Markdown"):
    payload = {"chat_id": CHAT_ID, "text": text}
    if parse_mode:
        payload["parse_mode"] = parse_mode
    try:
        r = requests.post(f"{TELEGRAM_API}/sendMessage", json=payload, timeout=15)
        if not r.ok:
            # Markdown 실패 시 plain text 재시도
            requests.post(
                f"{TELEGRAM_API}/sendMessage",
                json={"chat_id": CHAT_ID, "text": text},
                timeout=15,
            )
    except Exception as e:
        log.error(f"sendMessage 실패: {e}")

# ── 메인 ──────────────────────────────────────────────────
def run():
    seen = load_seen()
    log.info(f"시작. 기존 처리 영상 수: {len(seen)}")

    for channel in CHANNELS:
        log.info(f"채널 확인: {channel}")
        channel_id = resolve_channel_id(channel)
        if not channel_id:
            log.error(f"channel_id 조회 실패: {channel}")
            continue
        log.info(f"channel_id: {channel_id}")

        videos = fetch_latest_videos(channel_id, max_count=5)
        log.info(f"RSS에서 {len(videos)}개 영상 확인")

        if not videos:
            log.warning("영상 목록 비어있음 - RSS 접근 실패 가능")
            continue

        # 첫 실행: 최신 1개만 seen 등록 (스팸 방지)
        if not seen:
            log.info("첫 실행: 최신 영상 seen 등록만 함 (요약 안 보냄)")
            seen.add(videos[0]["id"])
            save_seen(seen)
            tg_send("✅ 봇 활성화됨! 다음 새 영상부터 자동 요약합니다.")
            continue

        new_videos = [v for v in reversed(videos) if v["id"] not in seen]
        if not new_videos:
            log.info("새 영상 없음")
            continue

        for video in new_videos:
            log.info(f"새 영상 발견: {video['title']} ({video['id']})")
            tg_send(f"🎬 *새 영상 알림*\n\n*{video['title']}*\n{video['url']}\n\n⏳ 요약 중...")

            try:
                summary = summarize(video)
                tg_send(f"📋 *요약*\n\n{summary}")
                log.info(f"요약 전송 완료: {video['id']}")
            except Exception as e:
                log.error(f"요약 실패: {e}")
                tg_send(f"❌ 요약 실패\n{str(e)[:200]}", parse_mode=None)

            seen.add(video["id"])
            save_seen(seen)
            time.sleep(2)

    log.info("완료")

if __name__ == "__main__":
    run()
