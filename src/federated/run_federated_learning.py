#!/usr/bin/env python3
"""
Simple Runner for Federated Learning System
==========================================

This script provides an easy way to run the federated learning system
with different configurations.
"""

import os
import sys
import argparse
import subprocess
import time

def run_tests():
    """Run system tests"""
    print("🧪 Running system tests...")
    result = subprocess.run([sys.executable, "test_system.py"], 
                          cwd=os.path.dirname(os.path.abspath(__file__)))
    return result.returncode == 0

def run_demo(clients=3, rounds=5):
    """Run the federated learning demo"""
    print(f"🚀 Starting federated learning demo with {clients} clients for {rounds} rounds...")
    result = subprocess.run([sys.executable, "demo.py", 
                           "--clients", str(clients),
                           "--rounds", str(rounds)],
                          cwd=os.path.dirname(os.path.abspath(__file__)))
    return result.returncode == 0

def run_server():
    """Run the central server"""
    print("🖥️ Starting central server...")
    result = subprocess.run([sys.executable, "server/central.py"],
                          cwd=os.path.dirname(os.path.abspath(__file__)))
    return result.returncode == 0

def run_client(client_id):
    """Run a client node"""
    print(f"👤 Starting client {client_id}...")
    result = subprocess.run([sys.executable, "client/node.py", client_id],
                          cwd=os.path.dirname(os.path.abspath(__file__)))
    return result.returncode == 0

def main():
    parser = argparse.ArgumentParser(description='Federated Learning System Runner')
    parser.add_argument('--mode', choices=['test', 'demo', 'server', 'client'], 
                       default='demo', help='Mode to run')
    parser.add_argument('--clients', type=int, default=3, 
                       help='Number of clients (for demo mode)')
    parser.add_argument('--rounds', type=int, default=5, 
                       help='Number of rounds (for demo mode)')
    parser.add_argument('--client-id', type=str, default='client_1', 
                       help='Client ID (for client mode)')
    
    args = parser.parse_args()
    
    print("🎯 Federated Learning System Runner")
    print("=" * 40)
    
    if args.mode == 'test':
        success = run_tests()
        if success:
            print("✅ All tests passed!")
        else:
            print("❌ Some tests failed!")
            sys.exit(1)
    
    elif args.mode == 'demo':
        success = run_demo(args.clients, args.rounds)
        if success:
            print("✅ Demo completed successfully!")
        else:
            print("❌ Demo failed!")
            sys.exit(1)
    
    elif args.mode == 'server':
        run_server()
    
    elif args.mode == 'client':
        run_client(args.client_id)

if __name__ == '__main__':
    main() 