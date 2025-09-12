import aiosqlite
import asyncio
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
from pydantic import BaseModel

class VideoRecord(BaseModel):
    id: str
    title: str
    url: str
    description: str
    transcript: Optional[str] = None
    processed: bool = False
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

class TranscriptDB:
    def __init__(self, db_path: str = "videos_cache.db"):
        self.db_path = Path(db_path)
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize the database and create tables if they don't exist"""
        if self._initialized:
            return
            
        async with aiosqlite.connect(self.db_path) as db:
            # Create videos table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS videos (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    url TEXT NOT NULL,
                    description TEXT,
                    transcript TEXT,
                    processed BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create transcript_chunks table for storing processed chunks
            await db.execute("""
                CREATE TABLE IF NOT EXISTS transcript_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    chunk_text TEXT NOT NULL,
                    embedding_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (video_id) REFERENCES videos (id),
                    UNIQUE(video_id, chunk_index)
                )
            """)
            
            # Create index for faster lookups
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_videos_processed 
                ON videos (processed)
            """)
            
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunks_video_id 
                ON transcript_chunks (video_id)
            """)
            
            await db.commit()
        
        self._initialized = True
        print(f"Database initialized at {self.db_path}")
    
    async def get_video(self, video_id: str) -> Optional[VideoRecord]:
        """Get a video record by ID"""
        await self.initialize()
        
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT id, title, url, description, transcript, processed, created_at, updated_at FROM videos WHERE id = ?",
                (video_id,)
            )
            row = await cursor.fetchone()
            
            if row:
                return VideoRecord(
                    id=row[0],
                    title=row[1],
                    url=row[2],
                    description=row[3],
                    transcript=row[4],
                    processed=bool(row[5]),
                    created_at=row[6],
                    updated_at=row[7]
                )
            return None
    
    async def upsert_video(self, video_record: VideoRecord) -> None:
        """Insert or update a video record"""
        await self.initialize()
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT OR REPLACE INTO videos 
                (id, title, url, description, transcript, processed, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (
                video_record.id,
                video_record.title,
                video_record.url,
                video_record.description,
                video_record.transcript,
                video_record.processed
            ))
            await db.commit()
    
    async def get_description(self, video_id: str) -> Optional[str]:
        """Get cached video description"""
        video = await self.get_video(video_id)
        return video.description if video else None
    
    async def cache_description(self, video_id: str, description: str) -> None:
        """Cache video description"""
        await self.initialize()
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "UPDATE videos SET description = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (description, video_id)
            )
            await db.commit()
    
    async def cache_transcript(self, video_id: str, transcript: str) -> None:
        """Cache transcript for a video"""
        await self.initialize()
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "UPDATE videos SET transcript = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (transcript, video_id)
            )
            await db.commit()
    
    async def mark_processed(self, video_id: str) -> None:
        """Mark a video as processed"""
        await self.initialize()
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "UPDATE videos SET processed = TRUE, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (video_id,)
            )
            await db.commit()
    
    async def get_unprocessed_videos(self) -> List[VideoRecord]:
        """Get all videos that haven't been processed yet"""
        await self.initialize()
        
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT id, title, url, description, transcript, processed, created_at, updated_at FROM videos WHERE processed = FALSE AND transcript IS NOT NULL"
            )
            rows = await cursor.fetchall()
            
            return [
                VideoRecord(
                    id=row[0],
                    title=row[1],
                    url=row[2],
                    description=row[3],
                    transcript=row[4],
                    processed=bool(row[5]),
                    created_at=row[6],
                    updated_at=row[7]
                )
                for row in rows
            ]
    
    async def save_transcript_chunks(self, video_id: str, chunks: List[str], embeddings: List[List[float]]) -> None:
        """Save transcript chunks and their embeddings"""
        await self.initialize()
        
        async with aiosqlite.connect(self.db_path) as db:
            # Clear existing chunks for this video
            await db.execute("DELETE FROM transcript_chunks WHERE video_id = ?", (video_id,))
            
            # Insert new chunks
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                embedding_json = json.dumps(embedding)
                await db.execute(
                    "INSERT INTO transcript_chunks (video_id, chunk_index, chunk_text, embedding_json) VALUES (?, ?, ?, ?)",
                    (video_id, i, chunk, embedding_json)
                )
            
            await db.commit()
    
    async def get_transcript_chunks(self, video_id: str) -> List[Tuple[str, List[float]]]:
        """Get cached transcript chunks and embeddings for a video"""
        await self.initialize()
        
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT chunk_text, embedding_json FROM transcript_chunks WHERE video_id = ? ORDER BY chunk_index",
                (video_id,)
            )
            rows = await cursor.fetchall()
            
            return [
                (row[0], json.loads(row[1]) if row[1] else [])
                for row in rows
            ]
    
    async def get_stats(self) -> Dict[str, int]:
        """Get database statistics"""
        await self.initialize()
        
        async with aiosqlite.connect(self.db_path) as db:
            # Count total videos
            cursor = await db.execute("SELECT COUNT(*) FROM videos")
            total_videos = (await cursor.fetchone())[0]
            
            # Count processed videos
            cursor = await db.execute("SELECT COUNT(*) FROM videos WHERE processed = TRUE")
            processed_videos = (await cursor.fetchone())[0]
            
            # Count videos with transcripts
            cursor = await db.execute("SELECT COUNT(*) FROM videos WHERE transcript IS NOT NULL")
            videos_with_transcripts = (await cursor.fetchone())[0]
            
            # Count total chunks
            cursor = await db.execute("SELECT COUNT(*) FROM transcript_chunks")
            total_chunks = (await cursor.fetchone())[0]
            
            return {
                "total_videos": total_videos,
                "processed_videos": processed_videos,
                "videos_with_transcripts": videos_with_transcripts,
                "total_chunks": total_chunks
            }
