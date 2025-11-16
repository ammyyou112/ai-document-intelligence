"""
Run script for the AI Document OCR application
This script provides better output visibility
"""
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == '__main__':
    print("=" * 60)
    print("Starting AI Document OCR Application...")
    print("=" * 60)
    print("\nServer will be available at:")
    print("  🌐 http://localhost:5000")
    print("  🌐 http://127.0.0.1:5000")
    print("\nPress CTRL+C to stop the server")
    print("=" * 60)
    print()
    
    try:
        from app import app
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n\nServer stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        print("\nMake sure all dependencies are installed:")
        print("  pip install -r requirements.txt")
        sys.exit(1)


