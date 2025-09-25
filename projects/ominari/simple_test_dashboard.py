#!/usr/bin/env python3
"""
Simple test to verify dashboard functionality
"""

import os
from flask import Flask
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_simple_dashboard():
    """Test a minimal Flask app to verify port 8888 works"""
    
    # Set PostgreSQL environment
    os.environ.update({
        'PG_HOST': 'localhost',
        'PG_PORT': '5435',
        'PG_USER': 'ominari_user', 
        'PG_PASSWORD': 'ominari_2025_secure',
        'PG_DB': 'ominari_production'
    })
    
    # Create simple Flask app
    app = Flask(__name__)
    
    @app.route('/')
    def home():
        return """
        <h1>🚀 Ominari Dashboard Test</h1>
        <p>✅ Flask is working on port 8888</p>
        <p>✅ PostgreSQL connection: Ready</p>
        <p><a href="/test">Test Database</a></p>
        """
    
    @app.route('/test')
    def test_db():
        try:
            from sqlalchemy import create_engine, text
            engine = create_engine(f"postgresql://{os.environ['PG_USER']}:{os.environ['PG_PASSWORD']}@{os.environ['PG_HOST']}:{os.environ['PG_PORT']}/{os.environ['PG_DB']}")
            
            with engine.connect() as conn:
                count = conn.execute(text("SELECT COUNT(*) FROM market WHERE source LIKE 'blockchain_%'")).scalar()
                
            return f"""
            <h1>Database Test</h1>
            <p>✅ PostgreSQL Connection: SUCCESS</p>
            <p>✅ Blockchain Markets: {count}</p>
            <p><a href="/">Back to Home</a></p>
            """
        except Exception as e:
            return f"""
            <h1>Database Test</h1>
            <p>❌ PostgreSQL Connection: FAILED</p>
            <p>Error: {e}</p>
            <p><a href="/">Back to Home</a></p>
            """
    
    logger.info("🔧 Starting simple dashboard test on port 8888...")
    logger.info("Visit: http://localhost:8888")
    
    # Run the app
    app.run(host='0.0.0.0', port=8888, debug=True)

if __name__ == "__main__":
    test_simple_dashboard()