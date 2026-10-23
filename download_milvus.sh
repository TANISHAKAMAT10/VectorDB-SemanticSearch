#!/usr/bin/env bash
set -euo pipefail
# Pinned to the official Milvus standalone Compose release used in this Review-2 setup.
# If a newer course-approved release is required, replace the version in this command.
curl -L "https://github.com/milvus-io/milvus/releases/download/v3.0.2/milvus-standalone-docker-compose.yml" -o docker-compose.yml
printf 'Downloaded official Milvus standalone docker-compose.yml\n'
