import subprocess
import time

def run_script(script_path, *args):
    command = ['python', script_path, *args]
    start_time = time.time()
    subprocess.run(command)
    end_time = time.time()
    execution_time = end_time - start_time
    return execution_time

# Remplacez 'smart_crop2.py' et 'smart_crop3.py' par les noms de vos scripts
script1_path = 'smart_crop2.py'
script2_path = 'smart_crop3.py'

# Arguments à passer à vos scripts
script_args = ['feature_det_weights', 'image.jpg', '0.5']

# Exécution et mesure du temps pour le premier script
time_script1 = run_script(script1_path, *script_args)
print(f"Temps d'exécution pour {script1_path}: {time_script1} secondes")

# Exécution et mesure du temps pour le deuxième script
time_script2 = run_script(script2_path, *script_args)
print(f"Temps d'exécution pour {script2_path}: {time_script2} secondes")

# Comparaison des temps d'exécution
if time_script1 < time_script2:
    print(f"{script1_path} est plus rapide.")
elif time_script1 > time_script2:
    print(f"{script2_path} est plus rapide.")
else:
    print("Les deux scripts ont des temps d'exécution égaux.")
