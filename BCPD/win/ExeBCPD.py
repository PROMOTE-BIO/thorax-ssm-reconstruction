import subprocess
import os

def ExecuteBCPD(mainDir, target, source, beta, lamda, invert):
    
    os.chdir(mainDir + '\\BCPD\\win')
    
    omg = '0.0'
    bet = str(beta)
    lmd = str(lamda)
    gma = '0.1'
    zet = '0.0'
    K = '300'
    J = '100'
    c = '1e-3'
    n = '100'
    modo1 = '-G geodesic,0.1,20,0.15'
    modo2 = '-DB,3000,0.1'

    if not invert:
        # Define the command and its arguments
        command = [
            './bcpd',
            '-x', target,
            '-y', source,
            '-w' + omg,
            '-l' + lmd,
            '-b' + bet,
            '-g' + gma,
            '-z' + zet,
            '-J' + J,
            '-K' + K,
            '-n' + n,
            '-c' + c,
            '-p',
            '-sA',
            modo1,
            #modo2
        ]
        
    else:
        command = [
            './bcpd',
            '-x', source,
            '-y', target,
            '-w' + omg,
            '-l' + lmd,
            '-b' + bet,
            '-g' + gma,
            '-z' + zet,
            '-J' + J,
            '-K' + K,
            '-n' + n,
            '-c' + c,
            '-p',
            '-sA',
            modo1,
            #modo2
        ]
        


    # Execute the command
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print("Command failed with error:", e)
        
    os.chdir(mainDir)
