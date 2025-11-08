#!/usr/bin/env python3
import http.server
import socketserver
import os
import sys

# Change to script directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

PORT = 3000
BIND_ADDRESS = '127.0.0.1'

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        return super().end_headers()

try:
    # Create server with explicit IPv4
    socketserver.TCPServer.allow_reuse_address = True
    handler = MyHTTPRequestHandler
    
    with socketserver.TCPServer((BIND_ADDRESS, PORT), handler) as httpd:
        print(f"✓ Server running at http://{BIND_ADDRESS}:{PORT}")
        print(f"✓ Press Ctrl+C to stop")
        print(f"✓ Files being served from: {os.getcwd()}")
        httpd.serve_forever()
        
except OSError as e:
    print(f"✗ Error: {e}")
    print(f"✗ Port {PORT} might be in use. Try a different port.")
    sys.exit(1)
except KeyboardInterrupt:
    print("\n✓ Server stopped")
    sys.exit(0)