#!/bin/bash

sync-podcasts --since 2000-01-01 --daemon --verbose --round-robin --model large --db ../../resources/fc/fc-text.db --feeds ../../resources/fc/feeds-text.toml --vector-store ../../resources/fc/text_grant_chroma_db --embedding-model nomic-embed-text:v1.5
