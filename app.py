import base64
from typing import List, Dict
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

API_KEY = os.getenv("DG_API_KEY")
DEEPGRAM_URL = 'https://api.deepgram.com/v1/listen?model=nova-2&diarize=true&filler_words=true'

class AudioRequest(BaseModel):
    audio_base64: str

def process_transcript(data: Dict, threshold: float = 0.5) -> List[Dict]:
    speakers = []
    current_speaker = None
    current_sentence = ""
    start_time = None
    end_time = None

    for word in data['words']:
        if current_speaker is None or word['speaker'] != current_speaker:
            if current_speaker is not None:
                add_sentence(speakers, current_speaker, current_sentence, start_time, end_time)
            current_speaker = word['speaker']
            current_sentence = word['word']
            start_time = word['start']
            end_time = word['end']
        elif word['start'] - end_time > threshold:
            add_sentence(speakers, current_speaker, current_sentence, start_time, end_time)
            current_sentence = word['word']
            start_time = word['start']
            end_time = word['end']
        else:
            current_sentence += f" {word['word']}"
            end_time = word['end']

    if current_sentence:
        add_sentence(speakers, current_speaker, current_sentence, start_time, end_time)

    return speakers

def add_sentence(speakers: List[Dict], speaker: int, sentence: str, start: float, end: float):
    for s in speakers:
        if s['speaker'] == speaker:
            s['transcript'].append({
                "content": sentence.strip(),
                "time_stamp": f"{start:.2f} - {end:.2f}"
            })
            return

    speakers.append({
        "speaker": speaker,
        "transcript": [{
            "content": sentence.strip(),
            "time_stamp": f"{start:.2f} - {end:.2f}"
        }]
    })

@app.post("/transcribe")
async def transcribe_audio(audio_request: AudioRequest):
    try:

        audio_bytes = base64.b64decode(audio_request.audio_base64)

        headers = {
            'Authorization': f'Token {API_KEY}',
            'Content-Type': 'audio/wav',
        }

        response = requests.post(
            url=DEEPGRAM_URL,
            headers=headers,
            data=audio_bytes
        )

        if response.status_code == 200:
            response_data = response.json()
            transcript_data = response_data["results"]["channels"][0]["alternatives"][0]
            result = process_transcript(transcript_data)
            return result
        else:
            raise HTTPException(status_code=response.status_code, detail=response.text)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
