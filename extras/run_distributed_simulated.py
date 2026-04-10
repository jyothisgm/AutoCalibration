import paramiko
import socket
import os
import time
from datetime import datetime

def create_incremental_folder(base_dir, base_name="hp_test_"):
    """Create a new folder in the specified directory with an incremental number."""
    number = 1
    while True:
        folder_name = f"{base_name}{number}"
        folder_path = os.path.join(base_dir, folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            return folder_path, number  # Return the created folder path
        elif os.path.exists(folder_path) and not os.listdir(folder_path):  # Empty folder check
            return folder_path, number
        number += 1

# SSH Credentials
username = "s3777103"
host_start = "u00650"

home_dir = os.path.expanduser("~")
key_path = home_dir + "/.ssh/id_ed25519"

folder_path = os.path.join(home_dir, "Documents/workspace/Thesis/AutoCalibration/")

remote_script = folder_path + "extras/test.py"
base_log_folder = folder_path + "logs_sim"
log_folder, trial = create_incremental_folder(base_log_folder)

run_script = folder_path + "gauss_newton.py"
python_interpreter = folder_path + ".venv/bin/python"

get_hname_cmd = "hostname"  # Get Hostname Command
python_command = f"{python_interpreter} {remote_script}"   # Test Python Command
kill_python = "pkill -f python"

check_gnr_running = "pgrep -f gauss_newton_real.py"
check_gn_running = "pgrep -f gauss_newton.py"

gpu_usage_cmd = "nvidia-smi --query-gpu=utilization.gpu --format=csv,nounits,noheader"
cpu_usage_cmd = "mpstat -P ALL 1 1 | grep \"all\" | awk '{print $NF}'"

# CUBOID_SIZES = ["compact", "small", "square", "normal", "tall", "wide", "coplanar"]
CUBOID_SIZES = ["normal"]

# ANGLE_FACTORS = [3, 4, 5, 6, 8, 9, 10, 12, 15, 18, 20, 24, 30, 36, 40, 45, 60, 72, 90, 120, 180, 360]
ANGLE_FACTORS = [3, 5, 6, 9, 10, 12, 15, 18, 24, 36, 60, 90, 180, 360]

# BEAD_LIST = [1, 2, 3, 4, 5, 6, 7]
BEAD_LIST = [3]

SCENARIO = list(range(0, 5))
# SCENARIO = [1, 2]

# LAMBDA_VALUES = ["GN", "LM_low", "LM_high"]
LAMBDA_VALUES = ["GN"]

# Initialize SSH client
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())  # Auto-accept unknown host keys

hostname = socket.gethostname()
print("Current Host: ", hostname.split(".")[0].upper())

counter = 0
while True:
    for i in range(1, 100):
        try:
            # Connect to the SSH server
            host = f"{host_start}{i:02}"
            if i in [1, 29]:
                print(f"Skipping: {host}")
                continue
            if hostname.split(".")[0].upper() == host.upper():
                print(f"Skipping current host: {host}")
                continue
            overflow = counter // len(SCENARIO) // len(ANGLE_FACTORS) // len(BEAD_LIST) // len(CUBOID_SIZES) // len(LAMBDA_VALUES)
            if overflow:
                print("overflow")
                print("Total hosts: ", counter)
                raise SystemExit(0)

            s       = SCENARIO[     counter % len(SCENARIO)]
            a       = ANGLE_FACTORS[counter // len(SCENARIO) % len(ANGLE_FACTORS)]
            k       = BEAD_LIST[    counter // len(SCENARIO) // len(ANGLE_FACTORS) % len(BEAD_LIST)]
            cuboid  = CUBOID_SIZES[ counter // len(SCENARIO) // len(ANGLE_FACTORS) // len(BEAD_LIST) % len(CUBOID_SIZES)]
            lam     = LAMBDA_VALUES[counter // len(SCENARIO) // len(ANGLE_FACTORS) // len(BEAD_LIST) // len(CUBOID_SIZES) % len(LAMBDA_VALUES)]

            run_rl =  f"nohup {python_interpreter} -u {run_script} -a {a} -s G{s} -k {k} -c {cuboid} -l {lam} > {log_folder}/run{counter:02d}_{host}.log 2>&1 &"
            print(f"----\nConnecting to: {host}  (K={k} N={a} G{s} cuboid={cuboid} lambda={lam})")
            ssh.connect(host, username=username, key_filename=key_path, timeout=10)

            # Run the Hostname command
            stdin, stdout, stderr = ssh.exec_command(get_hname_cmd)
            stdin.close()
            
            # Print the output
            print(stdout.read().decode(), end='')
            if stderr.read().decode():
                print("Errors:\n", stderr.read().decode(), end='')
                continue

            # Kill and Test Python
            # stdin, stdout, stderr = ssh.exec_command(kill_python)
            stdin, stdout, stderr = ssh.exec_command(python_command)
            stdin.close()
            
            # Print the output
            print(stdout.read().decode(), end='')
            if stderr.read().decode():
                print("Errors:\n", stderr.read().decode(), end='')

            stdin, stdout, stderr = ssh.exec_command('w')
            stdin.close()
            
            out = stdout.read().decode()
            active_users = out.split("\n")[2:]
            
            if stderr.read().decode():
                print("Errors:\n", stderr.read().decode(), end='')

            stdin, stdout, stderr = ssh.exec_command(cpu_usage_cmd)
            
            out_cpu = stdout.read().decode()
            idle_cpu = float(out_cpu.split('\n')[0].strip())
            print(f"IDLE CPU on {host} : {idle_cpu} %")

            if stderr.read().decode():
                print("Errors:\n", stderr.read().decode(), end='')

            stdin, stdout, stderr = ssh.exec_command(gpu_usage_cmd)
            
            out_gpu = stdout.read().decode()
            gpu_usage = float(out_gpu.strip())
            print(f"GPU Usage on {host} : {gpu_usage} %")

            if len(active_users) > 2 or gpu_usage > 20 or idle_cpu < 80:
                print(out, end='')
                continue
            if stderr.read().decode():
                print("Errors:\n", stderr.read().decode(), end='')
            
            # Check if gauss_newton.py is already running for the current user
            stdin, stdout, stderr = ssh.exec_command(check_gn_running)
            stdin.close()

            existing_pids = stdout.read().decode().strip()

            if existing_pids:
                print(f"Skipping {host}: gauss_newton.py already running (PID(s): {existing_pids})")
                ssh.close()
                continue

            # Check if gauss_newton.py is already running for the current user
            stdin, stdout, stderr = ssh.exec_command(check_gnr_running)
            stdin.close()

            existing_pids = stdout.read().decode().strip()

            if existing_pids:
                print(f"Skipping {host}: gauss_newton_real.py already running (PID(s): {existing_pids})")
                ssh.close()
                continue

            # Hyper Parameter Training
            stdin, stdout, stderr = ssh.exec_command(run_rl)
            stdin.close()

            print(f"Running: {run_rl}")
            # Print the output
            print(stdout.read().decode(), end='')
            if stderr.read().decode():
                print("Errors:\n", stderr.read().decode(), end='')
                continue
            counter += 1
            trial *= 10
            print(f"Running on {counter} hosts")

        except socket.gaierror:
            print(f"❌ Error: Could not resolve hostname '{host}'.")

        except paramiko.AuthenticationException:
            print(f"❌ Authentication failed for {host}. Check credentials.")

        except paramiko.SSHException as e:
            print(f"❌ SSH error on {host}: {e}")

        except Exception as e:
            print(f"❌ Unexpected error on {host}: {e}")
        print("")
        ssh.close()
    
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Completed full host sweep (0–100). Sleeping 5 minutes...")
    time.sleep(300)
