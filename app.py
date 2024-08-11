import base64
import os
from typing import List, Dict
import requests
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from textblob import TextBlob

load_dotenv()

app = FastAPI()

origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = os.getenv("DG_API_KEY")
DEEPGRAM_URL = 'https://api.deepgram.com/v1/listen?model=nova-2&diarize=true&filler_words=true'

FILLER_WORDS = ['uh', 'um', 'mhmm', 'mm-mm', 'uh-uh', 'uh-huh', 'nuh-uh']

class AudioRequest(BaseModel):
    audio_base64: str

def count_filler_words(sentence: str) -> int:
    return sum(sentence.lower().split().count(word) for word in FILLER_WORDS)

def get_sentiment(sentence: str) -> float:
    return TextBlob(sentence).sentiment.polarity

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

    for speaker in speakers:
        speaker['total_filler_words'] = sum(sentence['filler_word_count'] for sentence in speaker['transcript'])
        speaker['total_time'] = sum(sentence['duration'] for sentence in speaker['transcript'])
        speaker['average_sentiment'] = sum(sentence['sentiment'] for sentence in speaker['transcript']) / len(speaker['transcript'])

    return speakers

def add_sentence(speakers: List[Dict], speaker: int, sentence: str, start: float, end: float):
    filler_word_count = count_filler_words(sentence)
    duration = end - start
    sentiment = get_sentiment(sentence)

    for s in speakers:
        if s['speaker'] == speaker:
            s['transcript'].append({
                "content": sentence.strip(),
                "time_stamp": f"{start:.2f} - {end:.2f}",
                "filler_word_count": filler_word_count,
                "duration": duration,
                "sentiment": sentiment
            })
            return

    speakers.append({
        "speaker": speaker,
        "transcript": [{
            "content": sentence.strip(),
            "time_stamp": f"{start:.2f} - {end:.2f}",
            "filler_word_count": filler_word_count,
            "duration": duration,
            "sentiment": sentiment
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
