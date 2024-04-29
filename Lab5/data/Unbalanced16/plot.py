import matplotlib.pyplot as plt
import pandas as pd
import glob

files = glob.glob('Log_*.csv')
dataframes = []

for file in files:
    df = pd.read_csv(file, usecols=['iter', 'time'])
    dataframes.append(df)

plt.figure(figsize=(10, 6))

procIndex = 0
for df in dataframes:
    plt.plot(df['iter'], df['time'], label=f'ProcNum {procIndex}')
    procIndex += 1

plt.xlabel('Iterations')
plt.ylabel('Time')
plt.title('Time vs Iterations')
plt.legend()
plt.show()
