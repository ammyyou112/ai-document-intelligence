"""
Run script for the AI Document OCR application
This script provides better output visibility
"""
import sys
import os

# Add parent directory to path (so we can import from root)
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

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
        # Change to parent directory and import app
        os.chdir(parent_dir)
        import app
        app.app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n\nServer stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        print("\nMake sure all dependencies are installed:")
        print("  pip install -r requirements.txt")
        sys.exit(1)


