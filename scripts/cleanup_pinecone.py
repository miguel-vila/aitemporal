#!/usr/bin/env python3
"""
Script to clean up Pinecone data.
This will delete all vectors from the Pinecone index.
"""

import os
import asyncio
from pinecone import PineconeAsyncio
from pinecone.db_data import IndexAsyncio

async def cleanup_pinecone():
    """Delete all vectors from the Pinecone index"""

    # Check for API key
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        print("Error: PINECONE_API_KEY environment variable not set")
        return

    index_host = 'atemporal-transcripts-f09myss.svc.aped-4627-b74a.pinecone.io'

    print("Connecting to Pinecone...")

    async with PineconeAsyncio(api_key=api_key) as pc:
        index: IndexAsyncio = pc.IndexAsyncio(host=index_host)

        # Get index stats before cleanup
        stats = await index.describe_index_stats()
        total_vectors = stats.get('total_vector_count', 0)

        print(f"Found {total_vectors} vectors in the index")

        if total_vectors == 0:
            print("Index is already empty")
            return

        # Confirm deletion
        response = input(f"Are you sure you want to delete ALL {total_vectors} vectors? (y/N): ")
        if response.lower() != 'y':
            print("Cleanup cancelled")
            return

        print("Deleting all vectors...")

        # Delete all vectors (delete_all is the most efficient way)
        await index.delete(delete_all=True)

        print("All vectors deleted successfully!")

        # Verify cleanup
        stats_after = await index.describe_index_stats()
        remaining_vectors = stats_after.get('total_vector_count', 0)
        print(f"Remaining vectors: {remaining_vectors}")

if __name__ == "__main__":
    asyncio.run(cleanup_pinecone())