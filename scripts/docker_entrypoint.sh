#!/bin/sh
set -e

if [ ! -d "chroma_db" ] || [ -z "$(ls -A chroma_db 2>/dev/null)" ]; then
  echo "向量库为空，正在构建知识库..."
  python scripts/ingest_knowledge.py --rebuild
fi

exec "$@"
