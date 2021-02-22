

x = {'S': [1], 'A': [0], 'B': [1]}
y = {'J': [1], 'B': [0, 0], 'C': [0, 0], 'D': [6, 6], 'D6': [0, 0]}
z = dict(x)
for k, v in y.items():
    z[k] = [z[k], v] if k in z else v
    if isinstance(z[k][0],list):
        z[k] = [item for sublist in z[k] for item in sublist]



print(z)
