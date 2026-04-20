#!/usr/bin/env python3
"""
Automation Script for TF Drift Data Collection - IMPROVED VERSION
=========================================================

Robust automation with proper process management, Gazebo validation, and cleanup.
Based on the proven multipath ghost automation system.

Optimizations:
- Robust process management with tree killing
- Gazebo startup validation with timeout
- Proper system cleanup between runs
- Shutdown signal coordination
- File collection with validation
"""

import subprocess
import time
import os
import yaml
from datetime import datetime
import signal
import random
import argparse
import numpy as np
import glob
import psutil

def parse_args():
    parser = argparse.ArgumentParser(description="Automate data collection for TF drift experiments.")
    parser.add_argument("--ros_workspace", type=str, default="/home/talib/catkin_ws", help="Path to the ROS workspace.")
    parser.add_argument("--launch_file_path", type=str, help="Path to the roslaunch file.")
    parser.add_argument("--save_base_dir", type=str, default="/home/talib/collected_datasets_tf_drift", help="Base directory to save collected datasets.")
    parser.add_argument("--injector_save_dir", type=str, help="Directory where dataset_collector saves its CSVs.")
    parser.add_argument("--scenarios_file", type=str, default="/home/talib/catkin_ws/src/stretch_ros/stretch_navigation/config/scenarios.yaml", help="Path to scenarios YAML configuration file.")
    parser.add_argument("--gazebo_timeout", type=int, default=120, help="Timeout for Gazebo startup (seconds)")
    parser.add_argument("--skip_gazebo_check", action="store_true", help="Skip Gazebo ready check (use if Gazebo is slow)")

    return parser.parse_args()

def load_scenarios(scenarios_file):
    """Load scenarios from YAML file"""
    try:
        with open(scenarios_file, 'r') as f:
            scenarios = yaml.safe_load(f)
        print(f"Loaded {len(scenarios)} scenarios from {scenarios_file}")
        return scenarios
    except FileNotFoundError:
        print(f"Error: Scenarios file '{scenarios_file}' not found.")
        exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file '{scenarios_file}': {e}")
        exit(1)

def kill_process_tree(pid, timeout=10):
    """Kill process tree starting from PID"""
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        
        # Kill children first
        for child in children:
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass
        
        # Wait for children to terminate
        gone, alive = psutil.wait_procs(children, timeout=timeout)
        
        # Kill any remaining children
        for child in alive:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        
        # Kill parent
        try:
            parent.terminate()
            parent.wait(timeout=5)
        except (psutil.NoSuchProcess, psutil.TimeoutExpired):
            try:
                parent.kill()
            except psutil.NoSuchProcess:
                pass
                
    except psutil.NoSuchProcess:
        pass

def is_ros_master_running():
    """Check if ROS master is running"""
    try:
        result = subprocess.run(['pgrep', '-f', 'rosmaster'], 
                              capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            return True
    except Exception:
        pass
    
    try:
        result = subprocess.run(['pgrep', '-f', 'roscore'], 
                              capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            return True
    except Exception:
        pass
    
    return False

def force_kill_ros_gazebo():
    """Force kill all ROS and Gazebo processes"""
    print("🔨 Force killing all ROS/Gazebo processes...")
    
    kill_patterns = [
        'gzclient', 'gzserver', 'gazebo',
        'move_base', 'amcl', 'map_server', 'rviz', 
        'stretch_navigation', 'dataset_collector', 'tf_drift_injector', 'launch_terminator',
        'rosmaster', 'roscore', 'roslaunch'
    ]
    
    for pattern in kill_patterns:
        try:
            subprocess.run(['pkill', '-f', pattern], 
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(0.5)
            subprocess.run(['pkill', '-9', '-f', pattern], 
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass
    
    try:
        subprocess.run(['rosnode', 'kill', '--all'], 
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
    except Exception:
        pass
    
    try:
        subprocess.run(['killall', '-9', 'rosmaster', 'roscore'], 
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass
    
    # Clean up Gazebo lock files
    gazebo_lock_files = [
        '~/.gazebo/server.pid',
        '~/.gazebo/client.pid'
    ]
    
    for lock_file in gazebo_lock_files:
        lock_path = os.path.expanduser(lock_file)
        if os.path.exists(lock_path):
            try:
                os.remove(lock_path)
                print(f"Removed lock file: {lock_path}")
            except Exception:
                pass
    
    time.sleep(3)

def is_system_clean():
    """Check if system is completely clean of ROS/Gazebo processes"""
    if is_ros_master_running():
        print("ROS master is still running")
        return False
    
    check_patterns = [
        'gzserver', 'gzclient', 'gazebo', 'roslaunch'
    ]
    
    for pattern in check_patterns:
        try:
            result = subprocess.run(['pgrep', '-f', pattern], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                pids = result.stdout.strip().split('\n')
                if any(pid.strip() for pid in pids):
                    print(f"Found running processes for pattern: {pattern}")
                    return False
        except Exception:
            pass
    
    return True

def wait_for_clean_system(timeout=60):
    """Wait for completely clean system state"""
    print("🔄 Waiting for clean system state...")
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        if is_system_clean():
            print("✅ System is clean - ready for next run")
            return True
        
        force_kill_ros_gazebo()
        
        elapsed = time.time() - start_time
        remaining = timeout - elapsed
        print(f"Waiting... {elapsed:.1f}s/{timeout}s (remaining: {remaining:.1f}s)")
        time.sleep(5)
    
    print(f"❌ Timeout waiting for clean system after {timeout}s")
    return False

def wait_for_gazebo_ready(timeout=120, skip_check=False):
    """Wait for Gazebo to be ready with multiple detection methods"""
    if skip_check:
        print("⏩ Skipping Gazebo ready check - waiting fixed time instead")
        time.sleep(30)  # Fixed wait time
        return True
    
    print("🔄 Waiting for Gazebo to be ready...")
    start_time = time.time()
    
    methods_tried = 0
    methods = [
        # Method 1: Check /scan topic
        lambda: check_topic_ready("/scan", "LaserScan"),
        # Method 2: Check /odom topic  
        lambda: check_topic_ready("/odom", "Odometry"),
        # Method 3: Check /gazebo/link_states
        lambda: check_topic_ready("/gazebo/link_states", "Any"),
        # Method 4: Check if Gazebo process is running and stable
        lambda: check_gazebo_process_stable(),
    ]
    
    while time.time() - start_time < timeout:
        for i, method in enumerate(methods):
            try:
                if method():
                    print(f"✅ Gazebo ready (method {i+1})")
                    return True
            except Exception as e:
                print(f"Method {i+1} failed: {e}")
        
        elapsed = time.time() - start_time
        if elapsed < 30:  # First 30 seconds - check less frequently
            time.sleep(5)
        else:  # After 30 seconds - check more frequently
            time.sleep(2)
            
        print(f"Waiting for Gazebo... {elapsed:.1f}s/{timeout}s")
    
    print(f"❌ Gazebo startup timeout after {timeout}s")
    return False

def check_topic_ready(topic_name, topic_type="Any"):
    """Check if a specific topic is publishing data"""
    try:
        result = subprocess.run([
            "rostopic", "hz", topic_name, "-n", "3"
        ], capture_output=True, timeout=10, text=True)
        
        if "average rate" in result.stdout:
            print(f"   Topic {topic_name} is active")
            return True
        return False
    except subprocess.TimeoutExpired:
        print(f"   Topic {topic_name} check timeout")
        return False
    except Exception as e:
        print(f"   Topic {topic_name} check failed: {e}")
        return False

def check_gazebo_process_stable():
    """Check if Gazebo process is running and not crashing"""
    try:
        result = subprocess.run(['pgrep', '-f', 'gzserver'], 
                              capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            # Check if process has been running for at least 10 seconds
            pids = result.stdout.strip().split()
            for pid in pids:
                try:
                    proc = psutil.Process(int(pid))
                    if time.time() - proc.create_time() > 10:
                        print("   Gazebo process stable")
                        return True
                except (psutil.NoSuchProcess, ValueError):
                    continue
        return False
    except Exception:
        return False

def send_shutdown_signal():
    """Send ROS shutdown signal to all nodes"""
    print("🔄 Sending graceful shutdown signal...")
    try:
        subprocess.run([
            "rostopic", "pub", "/shutdown_signal", "std_msgs/Bool", 
            "data: true", "-1"
        ], check=True, timeout=10, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(5)
    except Exception as e:
        print(f"Could not send ROS shutdown signal: {e}")

def run_simulation(params, duration, run_id, scenario_name, launch_file_path, ros_workspace, injector_save_dir, gazebo_timeout=120, skip_gazebo_check=False):
    print(f"\n--- Running Scenario: {scenario_name}, Run ID: {run_id} ---")
    print(f"Parameters: {params}, Duration: {duration}")
    
    # PRE-RUN: Ensure clean state
    if not wait_for_clean_system(60):
        print("⚠️  WARNING: Starting run with potentially unclean system")
    
    # Build roslaunch command
    roslaunch_cmd_args = [f'{key}:={value}' for key, value in params.items()]
    roslaunch_cmd_args.append(f"scenario_name:={scenario_name}")
    roslaunch_cmd_args.append(f"run_id:={run_id}")
    roslaunch_cmd_args.append(f"experiment_duration:={duration}")
    roslaunch_cmd_args.append(f"injector_save_dir:={injector_save_dir}")

    full_roslaunch_command = f"source /opt/ros/noetic/setup.bash && source {ros_workspace}/devel/setup.bash && roslaunch {launch_file_path} {' '.join(roslaunch_cmd_args)}"
    cmd = ["bash", "-c", full_roslaunch_command]
    
    process = None
    start_time = time.time()
    
    try:
        print(f"Executing: roslaunch {launch_file_path} [with {len(params)} parameters]")
        process = subprocess.Popen(cmd, preexec_fn=os.setsid, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Simulation started (PID: {process.pid}). Waiting for Gazebo...")
        
        # Wait for Gazebo to be ready with configurable timeout
        if not wait_for_gazebo_ready(gazebo_timeout, skip_gazebo_check):
            raise RuntimeError(f"Gazebo startup timeout after {gazebo_timeout}s")
        
        print("✅ Simulation ready - starting data collection timer...")
        
        # Monitor simulation progress
        poll_interval = 5
        while process.poll() is None:
            elapsed = time.time() - start_time
            remaining = duration - elapsed
            progress_pct = (elapsed / duration) * 100
            
            print(f"  Progress: {elapsed:.0f}s / {duration}s ({progress_pct:.1f}%) | Remaining: {remaining:.0f}s", end='\r')
            
            # Check for duration completion
            if elapsed >= duration + 10:  # Extra 10s grace period
                print("\n⏰ Duration reached. Initiating graceful shutdown...")
                send_shutdown_signal()
                break
            
            time.sleep(poll_interval)
        
        elapsed = time.time() - start_time
        print(f"\n✅ Roslaunch process completed after {elapsed:.0f}s.")
        
    except Exception as e:
        print(f"\n❌ Error running simulation: {e}")
        if process:
            kill_process_tree(process.pid)
        return False
    
    finally:
        # POST-RUN CLEANUP
        print("🧹 Starting post-run cleanup...")
        
        if process and process.poll() is None:
            print("🔄 Process still running - terminating...")
            kill_process_tree(process.pid)
        
        force_kill_ros_gazebo()
        
        # Verify cleanup
        cleanup_start = time.time()
        while not is_system_clean() and (time.time() - cleanup_start < 30):
            force_kill_ros_gazebo()
            time.sleep(2)
        
        if is_system_clean():
            print("✅ Cleanup successful - system ready for next run")
        else:
            print("⚠️  Some processes may still be running")

    return True

def collect_and_move_files(scenario_name, run_id, injector_save_dir, save_base_dir):
    """Collect generated files and move them to the save directory"""
    current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Try multiple patterns to find the data file
    patterns = [
        f"*{scenario_name}_run{run_id}*.csv",
        f"*{scenario_name}*run{run_id}*.csv", 
        f"*{scenario_name}*.csv",
        "*.csv"  # Last resort: any CSV file
    ]
    
    found_files = {}
    for pattern in patterns:
        matching_files = glob.glob(os.path.join(injector_save_dir, pattern))
        if matching_files:
            # Sort by modification time and take most recent
            matching_files.sort(key=os.path.getmtime, reverse=True)
            found_file = matching_files[0]
            
            # Check if file has reasonable size (not empty)
            if os.path.getsize(found_file) > 100:  # At least 100 bytes
                found_files = {"dataset_tf_drift": found_file}
                print(f"Found data file for {scenario_name} (Run {run_id}): {os.path.basename(found_file)}")
                
                # Move to save directory
                new_filename = f"dataset_tf_{scenario_name}_run{run_id}_{current_timestamp}.csv"
                new_path = os.path.join(save_base_dir, new_filename)
                os.rename(found_file, new_path)
                print(f"  → Moved to: {new_filename}")
                break
            else:
                print(f"Found file but it's too small: {found_file}")
    
    if not found_files:
        print(f"Warning: No valid data files found for {scenario_name} (Run {run_id}) in {injector_save_dir}.")
        # List what's actually in the directory for debugging
        try:
            files_in_dir = os.listdir(injector_save_dir)
            print(f"Files in {injector_save_dir}: {files_in_dir}")
        except Exception as e:
            print(f"Could not list directory contents: {e}")

    return found_files, current_timestamp

def save_metadata(scenario_name, run_id, current_timestamp, params, duration, found_files, save_base_dir):
    """Save experiment metadata to YAML file"""
    metadata = {
        "scenario_name": scenario_name,
        "run_id": run_id,
        "timestamp": current_timestamp,
        "params": params,
        "duration": duration,
        "sampling_frequency": 10.0,
        "collected_files": {prefix: os.path.basename(path) for prefix, path in found_files.items()}
    }
    metadata_file = os.path.join(save_base_dir, f"metadata_{scenario_name}_run{run_id}_{current_timestamp}.yaml")
    with open(metadata_file, "w") as f:
        yaml.dump(metadata, f)
    print(f"Saved metadata to {metadata_file}")

def main():
    args = parse_args()
    
    # Set default paths if not provided
    if args.launch_file_path is None:
        args.launch_file_path = os.path.join(args.ros_workspace, "src/stretch_ros/stretch_navigation/launch/navigation_gazebo_tf_drift.launch")
    
    if args.injector_save_dir is None:
        args.injector_save_dir = os.path.join(args.ros_workspace, "Plot/tf_analysis")
    
    # Create save directory
    os.makedirs(args.save_base_dir, exist_ok=True)
    
    # Load scenarios from YAML file
    scenarios = load_scenarios(args.scenarios_file)
    
    print(f"Configuration:")
    print(f"  ROS Workspace: {args.ros_workspace}")
    print(f"  Launch File: {args.launch_file_path}")
    print(f"  Save Directory: {args.save_base_dir}")
    print(f"  Injector Save Directory: {args.injector_save_dir}")
    print(f"  Scenarios File: {args.scenarios_file}")
    print(f"  Gazebo Timeout: {args.gazebo_timeout}s")
    print(f"  Skip Gazebo Check: {args.skip_gazebo_check}")
    print(f"  Total Scenarios: {len(scenarios)}")
    
    # Validate required files exist
    if not os.path.exists(args.launch_file_path):
        print(f"Error: Launch file not found: {args.launch_file_path}")
        exit(1)
    
    if not os.path.exists(args.injector_save_dir):
        print(f"Error: Injector save directory not found: {args.injector_save_dir}")
        exit(1)

    # Run all scenarios
    total_runs = sum(config.get("runs", 1) for config in scenarios.values())
    current_run = 0
    successful_runs = 0
    
    for scenario_name, config in scenarios.items():
        runs = config.get("runs", 1)
        duration = config.get("duration", 3500)
        params = config.get("params", {})
        
        print(f"\n=== Starting scenario: {scenario_name} ({runs} runs) ===")
        
        for i in range(runs):
            current_run += 1
            print(f"\nProgress: {current_run}/{total_runs}")
            
            # Run simulation
            success = run_simulation(params, duration, i + 1, scenario_name, 
                                   args.launch_file_path, args.ros_workspace, 
                                   args.injector_save_dir, args.gazebo_timeout,
                                   args.skip_gazebo_check)
            
            if success:
                successful_runs += 1
                # Collect and move files
                found_files, current_timestamp = collect_and_move_files(
                    scenario_name, i + 1, args.injector_save_dir, args.save_base_dir)
                
                # Save metadata
                if found_files:
                    save_metadata(scenario_name, i + 1, current_timestamp, params, 
                                duration, found_files, args.save_base_dir)
            else:
                print(f"❌ Run {scenario_name} (Run {i+1}) failed - skipping file collection")
            
            # Add delay between runs (skip after last run)
            if i < runs - 1:
                print(f"⏳ Pausing for 2 minutes before next run... (Run {i+1}/{runs} of {scenario_name})")
                time.sleep(30)  # 1 minutes

    print(f"\n🎉 All simulations completed!")
    print(f"Successful runs: {successful_runs}/{total_runs}")
    print(f"Datasets saved to: {args.save_base_dir}")
    
    # Print summary
    csv_files = [f for f in os.listdir(args.save_base_dir) if f.endswith(".csv")]
    yaml_files = [f for f in os.listdir(args.save_base_dir) if f.endswith(".yaml")]
    
    print(f"\nSummary:")
    print(f"  CSV files: {len(csv_files)}")
    print(f"  Metadata files: {len(yaml_files)}")

if __name__ == "__main__":
    main()
