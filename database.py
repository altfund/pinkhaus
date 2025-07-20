#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 19 17:29:04 2025

@author: ess
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

DB_NAME = "sport_odds.db"

DB_URL = f"sqlite:///./{DB_NAME}"  # or full path
engine = create_engine(DB_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
