import paramiko
import socket
import os
import time
from datetime import datetime
from itertools import product

# SSH Credentials
username = "s3777103"
host_start = "u00650"

home_dir = os.path.expanduser("~")
key_path = home_dir + "/.ssh/id_ed25519"

folder_path = os.path.join(home_dir, "Documents/workspace/Thesis/AutoCalibration/")

remote_script   = folder_path + "extras/test.py"
base_log_folder = folder_path + "logs_cuboid_sizes"
run_script      = folder_path + "gauss_newton.py"
python_interpreter = folder_path + ".venv/bin/python"

get_hname_cmd    = "hostname"
python_command   = f"{python_interpreter} {remote_script}"
check_gn_running = "pgrep -f gauss_newton.py"
gpu_usage_cmd    = "nvidia-smi --query-gpu=utilization.gpu --format=csv,nounits,noheader"
cpu_usage_cmd    = "mpstat -P ALL 1 1 | grep \"all\" | awk '{print $NF}'"

SKIP_HOSTS = [1, 29, 80, 95]

CUBOID_SIZES = ["compact", "small", "square", "tall", "wide", "coplanar"]
ANGLE_FACTORS = [3, 5, 6, 9, 10, 12, 15, 18, 24, 36, 60, 90, 180, 360]
BEAD_LIST = [3]
SCENARIOS = [f"G{i}" for i in range(5)]


def create_log_folder(base_dir):
    number = 1
    while True:
        path = os.path.join(base_dir, f"run_{number}")
        if not os.path.exists(path):
            os.makedirs(path)
            return path
        elif not os.listdir(path):
            return path
        number += 1


# Build full job list: (cuboid, k, angle, scenario)
jobs = list(product(CUBOID_SIZES, BEAD_LIST, ANGLE_FACTORS, SCENARIOS))
total_jobs = len(jobs)
print(f"Total jobs: {total_jobs}")

log_folder = create_log_folder(base_log_folder)
print(f"Log folder: {log_folder}")

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

hostname = socket.gethostname()
print(f"Current host: {hostname.split('.')[0].upper()}\n")

counter = 0  # jobs dispatched

while True:
    for i in range(1, 100):
        if counter >= total_jobs:
            print(f"All {total_jobs} jobs dispatched.")
            raise SystemExit(0)

        try:
            host = f"{host_start}{i:02}"

            if i in SKIP_HOSTS:
                print(f"Skipping: {host}")
                continue
            if hostname.split(".")[0].upper() == host.upper():
                print(f"Skipping current host: {host}")
                continue

            cuboid, k, angle, scenario = jobs[counter]
            run_cmd = (
                f"nohup {python_interpreter} -u {run_script} "
                f"-c {cuboid} -k {k} -a {angle} -s {scenario} "
                f"> {log_folder}/run{counter:04d}_{host}.log 2>&1 &"
            )

            print(f"---- Connecting to: {host}  "
                  f"(job {counter+1}/{total_jobs}: cuboid={cuboid} K={k} N={angle} {scenario})")
            ssh.connect(host, username=username, key_filename=key_path, timeout=10)

            # hostname check
            stdin, stdout, stderr = ssh.exec_command(get_hname_cmd)
            stdin.close()
            print(stdout.read().decode(), end="")
            if stderr.read().decode():
                ssh.close()
                continue

            # Python environment check
            stdin, stdout, stderr = ssh.exec_command(python_command)
            stdin.close()
            print(stdout.read().decode(), end="")
            if stderr.read().decode():
                ssh.close()
                continue

            # Active users
            stdin, stdout, stderr = ssh.exec_command("w")
            stdin.close()
            w_out = stdout.read().decode()
            active_users = w_out.split("\n")[2:]

            # CPU idle
            stdin, stdout, stderr = ssh.exec_command(cpu_usage_cmd)
            stdin.close()
            cpu_out = stdout.read().decode().strip()
            idle_cpu = float(cpu_out.split("\n")[0].strip()) if cpu_out else 0.0
            print(f"  IDLE CPU : {idle_cpu:.1f}%")

            # GPU usage
            stdin, stdout, stderr = ssh.exec_command(gpu_usage_cmd)
            stdin.close()
            gpu_out = stdout.read().decode().strip()
            gpu_usage = float(gpu_out) if gpu_out else 100.0
            print(f"  GPU usage: {gpu_usage:.1f}%")

            if len(active_users) > 2 or gpu_usage > 20 or idle_cpu < 80:
                print(f"  Host busy — skipping")
                ssh.close()
                continue

            # Check if gauss_newton.py already running
            stdin, stdout, stderr = ssh.exec_command(check_gn_running)
            stdin.close()
            existing_pids = stdout.read().decode().strip()
            if existing_pids:
                print(f"  gauss_newton.py already running (PID {existing_pids}) — skipping")
                ssh.close()
                continue

            # Dispatch
            stdin, stdout, stderr = ssh.exec_command(run_cmd)
            stdin.close()
            print(f"  Dispatched: {run_cmd}")
            print(stdout.read().decode(), end="")

            counter += 1
            print(f"  Jobs dispatched: {counter}/{total_jobs}")

        except socket.gaierror:
            print(f"  Could not resolve hostname '{host}'")
        except paramiko.AuthenticationException:
            print(f"  Authentication failed for {host}")
        except paramiko.SSHException as e:
            print(f"  SSH error on {host}: {e}")
        except Exception as e:
            print(f"  Unexpected error on {host}: {e}")

        print("")
        ssh.close()

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Full sweep done. "
          f"Jobs dispatched: {counter}/{total_jobs}. Sleeping 5 minutes...")
    time.sleep(300)
