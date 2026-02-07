import paramiko
import socket
import os

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

remote_script = folder_path + "test.py"
base_log_folder = folder_path + "logs"
log_folder, trial = create_incremental_folder(base_log_folder)

run_script = folder_path + "gauss_newton.py"
python_interpreter = folder_path + ".venv/bin/python"

get_hname_cmd = "hostname"  # Get Hostname Command
python_command = f"{python_interpreter} {remote_script}"   # Test Python Command
kill_python = "pkill -f python"
gpu_usage_cmd = "nvidia-smi --query-gpu=utilization.gpu --format=csv,nounits,noheader"
cpu_usage_cmd = "mpstat -P ALL 1 1 | grep \"all\" | awk '{print $NF}'"

ANGLE_FACTORS = [60, 72, 90]
BEAD_LIST = [1, 2, 3, 4, 5, 6, 7]
BEAD_LIST = [4]
ANGLE_FACTORS = [90]

# Initialize SSH client
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())  # Auto-accept unknown host keys

hostname = socket.gethostname()
print("Current Host: ", hostname.split(".")[0].upper())

counter = 0
for i in range(0, 99):
    try:
        # Connect to the SSH server
        host = f"{host_start}{i:02}"
        if i in [7]:
            print(f"Skipping: {host}")
            continue
        if hostname.split(".")[0].upper() == host.upper():
            print(f"Skipping current host: {host}")
            continue
        overflow = counter // len(BEAD_LIST) // len(ANGLE_FACTORS)
        if overflow:
            print("overflow")
            break

        k = BEAD_LIST[counter % len(BEAD_LIST)]
        a = ANGLE_FACTORS[counter // len(ANGLE_FACTORS) % len(ANGLE_FACTORS)]
        
        run_rl =  f"nohup {python_interpreter} -u {run_script} -a {a} -k {k} > {log_folder}/run{counter}_{host}.log 2>&1 &"
        print(f"----\nConnecting to: {host}")
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
        stdin, stdout, stderr = ssh.exec_command(kill_python)
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

print("Total hosts: ", counter)