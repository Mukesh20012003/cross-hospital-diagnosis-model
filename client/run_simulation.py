# client/run_simulation.py

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
import time
from client.hospital_client import HospitalClient


SERVER_URL = "http://127.0.0.1:8000"


def check_server():
    """Check if the server is running"""
    try:
        response = requests.get(f"{SERVER_URL}/")
        if response.status_code == 200:
            print("✅ Server is running!")
            return True
    except requests.exceptions.ConnectionError:
        print("❌ Server is NOT running!")
        print("   Start the server first with:")
        print("   python -m uvicorn server.main:app --reload --port 8000")
        return False


def run_federated_simulation(num_rounds=3):
    """
    Run a complete federated learning simulation with 3 hospitals.
    """
    print("\n" + "=" * 60)
    print("  🌐 CROSS-HOSPITAL FEDERATED LEARNING SIMULATION")
    print("=" * 60)
    
    # Check server
    if not check_server():
        return
    
    # Create hospital clients
    hospitals = [
        HospitalClient("hospital_1", SERVER_URL),
        HospitalClient("hospital_2", SERVER_URL),
        HospitalClient("hospital_3", SERVER_URL),
    ]
    
    # Step 1: Register all hospitals
    print("\n" + "=" * 60)
    print("  PHASE 1: REGISTERING HOSPITALS")
    print("=" * 60)
    
    for hospital in hospitals:
        success = hospital.register()
        if not success:
            print(f"❌ Failed to register {hospital.hospital_name}")
            return
        time.sleep(1)
    
    # Step 2: Run training rounds
    for round_num in range(1, num_rounds + 1):
        print("\n" + "=" * 60)
        print(f"  ROUND {round_num} of {num_rounds}")
        print("=" * 60)
        
        # Start a new round on the server
        print("\n🚀 Starting new training round on server...")
        response = requests.post(f"{SERVER_URL}/start_round", json={"target_participants": 3})
        
        if response.status_code == 200:
            round_info = response.json()
            print(f"   ✅ Round {round_info['round_number']} started!")
        else:
            print(f"   ❌ Failed to start round: {response.text}")
            return
        
        time.sleep(1)
        
        # Each hospital trains and submits
        for hospital in hospitals:
            success = hospital.run_one_round()
            if not success:
                print(f"❌ {hospital.hospital_name} failed this round")
            time.sleep(1)
        
        # Check round status
        time.sleep(2)
        response = requests.get(f"{SERVER_URL}/round_status")
        if response.status_code == 200:
            status = response.json()
            print(f"\n📊 Round Status: {status['status']}")
            if 'global_accuracy' in status and status['global_accuracy']:
                print(f"   Global Accuracy: {status['global_accuracy']:.4f}")
        
        print(f"\n✅ Round {round_num} complete!")
        time.sleep(2)
    
    # Final summary
    print("\n" + "=" * 60)
    print("  📊 TRAINING SUMMARY")
    print("=" * 60)
    
    response = requests.get(f"{SERVER_URL}/rounds")
    if response.status_code == 200:
        rounds = response.json()
        for r in rounds:
            print(f"   Round {r['round_number']}: "
                  f"Status={r['status']}, "
                  f"Participants={r['num_participants']}, "
                  f"Accuracy={r.get('global_accuracy', 'N/A')}")
    
    response = requests.get(f"{SERVER_URL}/api/dashboard_stats")
    if response.status_code == 200:
        stats = response.json()
        print(f"\n   Total Hospitals: {stats['total_hospitals']}")
        print(f"   Total Rounds: {stats['total_rounds']}")
        print(f"   Latest Accuracy: {stats['latest_accuracy']}")
        print(f"   Total Updates: {stats['total_updates']}")
    
    print(f"\n🎉 Federated Learning Simulation Complete!")
    print("=" * 60)


if __name__ == "__main__":
    # First check if data exists
    if not os.path.exists("data/hospital_1"):
        print("⚠️ Data not found! Preparing data first...")
        from client.prepare_data import main as prepare_main
        prepare_main()
    
    # Run simulation with 3 rounds
    run_federated_simulation(num_rounds=3)