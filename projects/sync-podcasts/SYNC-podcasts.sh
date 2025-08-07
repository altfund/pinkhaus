#!/bin/bash

sync-podcasts --since 2000-01-01 --daemon --verbose --round-robin --model tiny --db ../../resources/fc/fc.db --feeds ../../resources/fc/feeds-small.toml --vector-store ../../resources/fc/small_grant_chroma_db --embedding-model nomic-embed-text:v1.5
