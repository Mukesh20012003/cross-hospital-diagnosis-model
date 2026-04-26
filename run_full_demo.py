# run_full_demo.py

import sys
import os
import time
import subprocess
import requests
import json
import threading

SERVER_URL = "http://127.0.0.1:8000"
SERVER_PROCESS = None


def print_banner(text):
    print("\n" + "=" * 65)
    print(f"  {text}")
    print("=" * 65)


def print_step(step_num, text):
    print(f"\n  📌 Step {step_num}: {text}")
    print("  " + "-" * 50)


def start_server():
    """Start the FastAPI server in background"""
    global SERVER_PROCESS
    print_step(1, "Starting Aggregator Server")
    
    SERVER_PROCESS = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "server.main:app", "--port", "8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait for server to start
    print("  ⏳ Waiting for server to start...")
    for i in range(15):
        try:
            response = requests.get(f"{SERVER_URL}/")
            if response.status_code == 200:
                print("  ✅ Server is running!")
                return True
        except:
            pass
        time.sleep(1)
    
    print("  ❌ Server failed to start!")
    return False


def stop_server():
    """Stop the server"""
    global SERVER_PROCESS
    if SERVER_PROCESS:
        SERVER_PROCESS.terminate()
        SERVER_PROCESS.wait()
        print("\n  🛑 Server stopped.")


def clean_previous_data():
    """Remove old database and model files for fresh start"""
    print_step(0, "Cleaning Previous Data")
    
    files_to_remove = ["federated_learning.db"]
    dirs_to_clean = [
        "models/updates", "models/global",
        "data/hospital_1/api_key.txt",
        "data/hospital_2/api_key.txt", 
        "data/hospital_3/api_key.txt"
    ]
    
    for f in files_to_remove:
        if os.path.exists(f):
            os.remove(f)
            print(f"  🗑️ Removed {f}")
    
    for d in dirs_to_clean:
        if os.path.isfile(d):
            os.remove(d)
            print(f"  🗑️ Removed {d}")
        elif os.path.isdir(d):
            import shutil
            shutil.rmtree(d)
            print(f"  🗑️ Removed {d}/")
    
    print("  ✅ Clean slate ready!")


def prepare_data():
    """Prepare hospital data partitions"""
    print_step(2, "Preparing Hospital Data Partitions")
    
    from client.prepare_data import main as prepare_main
    prepare_main()


def register_hospitals():
    """Register 3 hospitals with the server"""
    print_step(3, "Registering Hospitals")
    
    hospitals = [
        {"name": "hospital_1", "location": "New York, USA", "data_size": 300},
        {"name": "hospital_2", "location": "London, UK", "data_size": 300},
        {"name": "hospital_3", "location": "Tokyo, Japan", "data_size": 300},
    ]
    
    api_keys = {}
    
    for h in hospitals:
        response = requests.post(f"{SERVER_URL}/register_hospital", json=h)
        if response.status_code == 200:
            result = response.json()
            api_keys[h["name"]] = result["api_key"]
            print(f"  ✅ {h['name']} registered ({h['location']})")
            
            # Save API key
            key_dir = os.path.join("data", h["name"])
            os.makedirs(key_dir, exist_ok=True)
            with open(os.path.join(key_dir, "api_key.txt"), "w") as f:
                f.write(result["api_key"])
        else:
            print(f"  ❌ Failed to register {h['name']}: {response.text}")
    
    return api_keys


def run_training_rounds(num_rounds=3):
    """Run federated learning training rounds"""
    print_step(4, f"Running {num_rounds} Federated Learning Rounds")
    
    from client.hospital_client import HospitalClient
    
    hospitals = [
        HospitalClient("hospital_1", SERVER_URL),
        HospitalClient("hospital_2", SERVER_URL),
        HospitalClient("hospital_3", SERVER_URL),
    ]
    
    # Load API keys
    for h in hospitals:
        key_file = os.path.join("data", h.hospital_name, "api_key.txt")
        if os.path.exists(key_file):
            with open(key_file, "r") as f:
                h.api_key = f.read().strip()
    
    round_results = []
    
    for round_num in range(1, num_rounds + 1):
        print(f"\n  {'─' * 50}")
        print(f"  🔄 ROUND {round_num} of {num_rounds}")
        print(f"  {'─' * 50}")
        
        # Start round
        response = requests.post(
            f"{SERVER_URL}/start_round",
            json={"target_participants": 3}
        )
        
        if response.status_code != 200:
            print(f"  ❌ Failed to start round: {response.text}")
            continue
        
        round_info = response.json()
        print(f"  ✅ Round {round_info['round_number']} started")
        
        # Each hospital trains and submits
        for hospital in hospitals:
            hospital.run_one_round()
            time.sleep(0.5)
        
        # Check results
        time.sleep(1)
        response = requests.get(f"{SERVER_URL}/round_status")
        if response.status_code == 200:
            status = response.json()
            if status.get("global_accuracy"):
                round_results.append({
                    "round": round_num,
                    "accuracy": status["global_accuracy"],
                    "loss": status.get("global_loss", 0)
                })
                print(f"\n  📊 Round {round_num} Results:")
                print(f"     Accuracy: {status['global_accuracy']*100:.2f}%")
                print(f"     Loss: {status.get('global_loss', 'N/A')}")
        
        time.sleep(1)
    
    return round_results


def generate_reports():
    """Generate compliance reports"""
    print_step(5, "Generating Compliance Reports")
    
    # PDF Report
    try:
        response = requests.get(f"{SERVER_URL}/export/pdf")
        if response.status_code == 200:
            pdf_path = "reports/compliance_report.pdf"
            os.makedirs("reports", exist_ok=True)
            with open(pdf_path, "wb") as f:
                f.write(response.content)
            print(f"  ✅ PDF Report: {pdf_path}")
    except Exception as e:
        print(f"  ⚠️ PDF generation: {e}")
    
    # CSV Report
    try:
        response = requests.get(f"{SERVER_URL}/export/csv")
        if response.status_code == 200:
            csv_path = "reports/training_report.csv"
            with open(csv_path, "wb") as f:
                f.write(response.content)
            print(f"  ✅ CSV Report: {csv_path}")
    except Exception as e:
        print(f"  ⚠️ CSV generation: {e}")
    
    # Audit Log
    try:
        response = requests.get(f"{SERVER_URL}/export/audit_csv")
        if response.status_code == 200:
            audit_path = "reports/audit_log.csv"
            with open(audit_path, "wb") as f:
                f.write(response.content)
            print(f"  ✅ Audit Log: {audit_path}")
    except Exception as e:
        print(f"  ⚠️ Audit log: {e}")


def print_final_summary():
    """Print final summary"""
    print_step(6, "Final Summary")
    
    try:
        # Dashboard stats
        response = requests.get(f"{SERVER_URL}/api/dashboard_stats")
        if response.status_code == 200:
            stats = response.json()
            print(f"\n  ┌────────────────────────────────────────┐")
            print(f"  │    FEDERATED LEARNING RESULTS          │")
            print(f"  ├────────────────────────────────────────┤")
            print(f"  │  Total Hospitals:  {stats['total_hospitals']:>18}  │")
            print(f"  │  Active Hospitals: {stats['active_hospitals']:>18}  │")
            print(f"  │  Total Rounds:     {stats['total_rounds']:>18}  │")
            print(f"  │  Total Updates:    {stats['total_updates']:>18}  │")
            print(f"  │  Model Downloads:  {stats['model_downloads']:>18}  │")
            accuracy_str = f"{stats['latest_accuracy']*100:.2f}%" if stats['latest_accuracy'] else "N/A"
            print(f"  │  Latest Accuracy:  {accuracy_str:>18}  │")
            print(f"  └────────────────────────────────────────┘")
        
        # Round history
        response = requests.get(f"{SERVER_URL}/rounds")
        if response.status_code == 200:
            rounds = response.json()
            print(f"\n  📊 Round-by-Round Accuracy:")
            for r in reversed(rounds):
                acc = f"{r['global_accuracy']*100:.2f}%" if r.get('global_accuracy') else "N/A"
                bar_len = int(r['global_accuracy'] * 30) if r.get('global_accuracy') else 0
                bar = "█" * bar_len + "░" * (30 - bar_len)
                print(f"     Round {r['round_number']}: [{bar}] {acc}")
        
        # Privacy info
        response = requests.get(f"{SERVER_URL}/privacy_info")
        if response.status_code == 200:
            privacy = response.json()
            print(f"\n  🔒 Privacy Status:")
            print(f"     Raw data shared: {privacy['data_handling']['raw_data_shared']}")
            print(f"     Data location: {privacy['data_handling']['data_location']}")
            print(f"     HIPAA compatible: {privacy['compliance']['hipaa_compatible']}")
            print(f"     GDPR compatible: {privacy['compliance']['gdpr_compatible']}")
    
    except Exception as e:
        print(f"  Error getting summary: {e}")
    
    print(f"\n  🌐 Dashboard: http://127.0.0.1:8000/dashboard")
    print(f"  📄 API Docs:  http://127.0.0.1:8000/docs")


def main():
    """Run complete demo"""
    print_banner("🏥 CROSS-HOSPITAL DIAGNOSIS MODEL — FULL DEMO")
    print("  Federated Learning System for Collaborative Medical Diagnosis")
    print("  No raw patient data is shared between hospitals!")
    
    try:
        # Step 0: Clean
        clean_previous_data()
        
        # Step 1: Start server
        if not start_server():
            print("❌ Cannot start server. Exiting.")
            return
        
        time.sleep(2)
        
        # Step 2: Prepare data
        prepare_data()
        
        time.sleep(1)
        
        # Step 3: Register hospitals
        register_hospitals()
        
        time.sleep(1)
        
        # Step 4: Run training
        results = run_training_rounds(num_rounds=3)
        
        # Step 5: Generate reports
        generate_reports()
        
        # Step 6: Summary
        print_final_summary()
        
        print_banner("🎉 DEMO COMPLETE!")
        print("\n  The server is still running. You can:")
        print("  1. Open dashboard: http://127.0.0.1:8000/dashboard")
        print("  2. View API docs:  http://127.0.0.1:8000/docs")
        print("  3. Press Ctrl+C to stop\n")
        
        # Keep server running for dashboard viewing
        while True:
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("\n\n  👋 Shutting down...")
    finally:
        stop_server()
        print("  ✅ Demo ended. Goodbye!")


if __name__ == "__main__":
    main()