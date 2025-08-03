#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  2 18:08:08 2025

@author: ess
"""

from sqlalchemy.orm import Session
from sqlalchemy.exc import OperationalError
from sqlalchemy import text

# assume SessionLocal = sessionmaker(bind=engine, ...)
from database import SessionLocal

sess: Session = SessionLocal()

# 1) Check if you’re already mid-transaction
if sess.in_transaction():
    print("⚠️  Found an open/failed transaction—rolling back now.")
    sess.rollback()
else:
    print("✔️  No active transaction.")

# 2) (Optionally) test a dummy query to see if the DB is locked
try:
    sess.execute(text("SELECT 1"))
    print("✔️  DB is responding.")
except OperationalError as e:
    if "database is locked" in str(e).lower():
        print("🔒  Database is locked right now.")
        # if somehow a txn was left open, rollback again
        if sess.in_transaction():
            sess.rollback()
            print("🔄  Rolled back pending transaction.")
    else:
        raise  # some other problem

# … now you can safely start your real work …

# finally, when you’re done
sess.close()
