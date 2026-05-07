#!/usr/bin/env python3
"""
RAG Migration CLI: Re-embed collections when switching embedding models.

Usage:
    python tools/rag_migrate.py --source-model all-MiniLM-L6-v2 --target-model BAAI/bge-base-en-v1.5
    python tools/rag_migrate.py --source-collection old_collection_name --schema-version 1

Examples:
    # Migrate from old model to new model (same schema version)
    python tools/rag_migrate.py --source-model all-MiniLM-L6-v2 --target-model BAAI/bge-base-en-v1.5

    # Migrate from old schema version to new one
    python tools/rag_migrate.py --source-collection dexter_personal_rag_idxv1_default --schema-version 1
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.brain.rag import PersonalRAGIndex, INDEX_SCHEMA_VERSION
from utils.logger import get_logger

logger = get_logger("rag_migrate")


def migrate_from_model(
    user_id: str,
    persist_directory: str,
    source_model: str,
    target_model: str = "BAAI/bge-base-en-v1.5",
    schema_version: int | None = None,
) -> int:
    """
    Migrate from one embedding model to another.
    
    Args:
        user_id: User identifier
        persist_directory: RAG persist directory
        source_model: Source embedding model name
        target_model: Target embedding model name
        schema_version: Schema version (if None, uses current INDEX_SCHEMA_VERSION)
    
    Returns:
        Number of chunks migrated
    """
    schema_v = schema_version or INDEX_SCHEMA_VERSION
    print(f"\n{'='*60}")
    print(f"RAG Migration: {source_model} → {target_model}")
    print(f"{'='*60}")
    print(f"User ID: {user_id}")
    print(f"Schema version: {schema_v}")
    print(f"Persist directory: {persist_directory}")
    
    # Create target index with new model
    target_idx = PersonalRAGIndex(
        persist_directory=persist_directory,
        user_id=user_id,
        embedding_model=target_model,
        index_schema_version=schema_v,
    )
    
    print(f"\nTarget collection: {target_idx._collection_name}")
    
    # Perform migration
    t0 = time.perf_counter()
    migrated = target_idx.migrate_from_model(source_model, source_index_schema_version=schema_v)
    elapsed = time.perf_counter() - t0
    
    print(f"\n✓ Migration completed!")
    print(f"  Chunks re-embedded: {migrated}")
    print(f"  Elapsed time: {elapsed/60:.1f} minutes")
    print(f"  Target collection: {target_idx._collection_name}")
    print(f"  Source model: {source_model}")
    print(f"  Target model: {target_model}")
    
    return migrated


def migrate_from_collection(
    user_id: str,
    persist_directory: str,
    source_collection: str,
    target_model: str = "BAAI/bge-base-en-v1.5",
    target_schema_version: int | None = None,
) -> int:
    """
    Migrate from a named source collection to a new target collection.
    
    Args:
        user_id: User identifier
        persist_directory: RAG persist directory
        source_collection: Source collection name
        target_model: Target embedding model
        target_schema_version: Target schema version
    
    Returns:
        Number of chunks migrated
    """
    target_v = target_schema_version or INDEX_SCHEMA_VERSION
    print(f"\n{'='*60}")
    print(f"RAG Collection Migration")
    print(f"{'='*60}")
    print(f"User ID: {user_id}")
    print(f"Source collection: {source_collection}")
    print(f"Target model: {target_model}")
    print(f"Target schema version: {target_v}")
    print(f"Persist directory: {persist_directory}")
    
    # Create target index
    target_idx = PersonalRAGIndex(
        persist_directory=persist_directory,
        user_id=user_id,
        embedding_model=target_model,
        index_schema_version=target_v,
    )
    
    print(f"\nTarget collection: {target_idx._collection_name}")
    
    # Perform migration
    t0 = time.perf_counter()
    migrated = target_idx.migrate_from_collection(source_collection)
    elapsed = time.perf_counter() - t0
    
    print(f"\n✓ Migration completed!")
    print(f"  Chunks re-embedded: {migrated}")
    print(f"  Elapsed time: {elapsed/60:.1f} minutes")
    print(f"  Target collection: {target_idx._collection_name}")
    
    return migrated


def main():
    parser = argparse.ArgumentParser(
        description="Re-embed RAG collections when switching embedding models."
    )
    parser.add_argument(
        "--user-id",
        default=None,
        help="User ID (default: current user)",
    )
    parser.add_argument(
        "--persist-dir",
        default="./memory_db",
        help="RAG persist directory (default: ./memory_db)",
    )
    parser.add_argument(
        "--source-model",
        help="Source embedding model name (e.g., all-MiniLM-L6-v2)",
    )
    parser.add_argument(
        "--source-collection",
        help="Source collection name (alternative to --source-model)",
    )
    parser.add_argument(
        "--target-model",
        default="BAAI/bge-base-en-v1.5",
        help="Target embedding model (default: BAAI/bge-base-en-v1.5)",
    )
    parser.add_argument(
        "--schema-version",
        type=int,
        help="Source schema version (for --source-model mode)",
    )
    parser.add_argument(
        "--target-schema-version",
        type=int,
        help="Target schema version (default: current)",
    )
    
    args = parser.parse_args()
    
    # Determine user ID
    user_id = args.user_id
    if not user_id:
        import getpass
        user_id = getpass.getuser()
    
    # Validate persist directory
    persist_dir = os.path.abspath(args.persist_dir)
    if not os.path.isdir(persist_dir):
        print(f"Error: Persist directory not found: {persist_dir}")
        sys.exit(1)
    
    # Perform migration
    if args.source_model:
        # Model-to-model migration
        migrated = migrate_from_model(
            user_id=user_id,
            persist_directory=persist_dir,
            source_model=args.source_model,
            target_model=args.target_model,
            schema_version=args.schema_version,
        )
    elif args.source_collection:
        # Collection-to-collection migration
        migrated = migrate_from_collection(
            user_id=user_id,
            persist_directory=persist_dir,
            source_collection=args.source_collection,
            target_model=args.target_model,
            target_schema_version=args.target_schema_version,
        )
    else:
        print("Error: Specify either --source-model or --source-collection")
        parser.print_help()
        sys.exit(1)
    
    if migrated == 0:
        print("\n⚠ No chunks were migrated. Check source collection/model name.")
        sys.exit(1)
    
    print(f"\n✓ Done! {migrated} chunks now available in the new collection.")


if __name__ == "__main__":
    main()
