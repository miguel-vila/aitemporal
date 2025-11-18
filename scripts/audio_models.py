from pydantic import BaseModel
import json
from typing import Any

class AudioLine(BaseModel):
    speaker: str
    text: str
    start: float
    end: float

    def toJSON(self) -> str:
        return json.dumps({
            'speaker': self.speaker,
            'text': self.text,
            'start': self.start,
            'end': self.end
        }, ensure_ascii=False)
        
    @staticmethod
    def fromDict(data: dict[str,Any]) -> 'AudioLine':
        return AudioLine(
            speaker=data['speaker'],
            text=data['text'],
            start=data.get('start', 0.0),
            end=data.get('end', 0.0)
        )
    @staticmethod
    def from_db(data: str) -> list['AudioLine']:
        segments_json = json.loads(data)['segments']
        if type(segments_json) is list:
            return [AudioLine.fromDict(line) for line in segments_json]
        else:
            raise Exception(f"Invalid cached diarized transcript format. Expected list, got {type(segments_json)}")

class AudioLineEncoder(json.JSONEncoder):
    def default(self, o: Any):
        if isinstance(o, AudioLine):
            return o.model_dump()
        return super().default(o)
