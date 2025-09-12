import asyncio
from transcript_db import TranscriptDB, VideoRecord

async def test_database():
    """Test basic database functionality"""
    db = TranscriptDB("test_transcripts.db")
    
    # Initialize database
    await db.initialize()
    print("✓ Database initialized")
    
    # Test caching a video description
    await db.cache_description(
        video_id="test123",
        title="Test Video",
        url="https://youtube.com/watch?v=test123",
        description="This is a test video description"
    )
    print("✓ Cached video description")
    
    # Test retrieving cached description
    cached_desc = await db.get_description("test123")
    assert cached_desc == "This is a test video description", f"Expected test description, got: {cached_desc}"
    print("✓ Retrieved cached description")
    
    # Test caching transcript
    await db.cache_transcript("test123", "This is a test transcript with multiple sentences. Here is another sentence.")
    print("✓ Cached transcript")
    
    # Test saving chunks and embeddings
    test_chunks = ["This is chunk 1", "This is chunk 2"]
    test_embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    await db.save_transcript_chunks("test123", test_chunks, test_embeddings)
    print("✓ Saved transcript chunks")
    
    # Test retrieving chunks
    cached_chunks = await db.get_transcript_chunks("test123")
    assert len(cached_chunks) == 2, f"Expected 2 chunks, got {len(cached_chunks)}"
    print("✓ Retrieved transcript chunks")
    
    # Test marking as processed
    await db.mark_processed("test123")
    print("✓ Marked video as processed")
    
    # Test getting video record
    video = await db.get_video("test123")
    assert video is not None, "Video should exist"
    assert video.processed == True, "Video should be marked as processed"
    print("✓ Retrieved video record")
    
    # Test database stats
    stats = await db.get_stats()
    print(f"✓ Database stats: {stats}")
    
    print("\n🎉 All database tests passed!")

if __name__ == "__main__":
    asyncio.run(test_database())
