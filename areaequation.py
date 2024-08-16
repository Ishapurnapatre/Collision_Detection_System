import matplotlib.pyplot as plt
import numpy as np

# Define the data
classes = ['TW', 'RICK', 'LMV', 'HMV']
values = {
    'TW': [29799, 8732, 4284, 2419, 1518],    
    'RICK': [55692, 17820, 8816, 5100, 3400],
    'LMV': [67915, 20020, 10152, 5822, 3400],
    'HMV': [151956, 49350, 23904, 13843, 8858, 6177]
}

# Set up the figure and axis
fig, ax = plt.subplots()

# Plot each class
for cls in classes:
    ax.plot(values[cls], label=cls, marker='o')

# Add labels and title
ax.set_xlabel('Sample Index')
ax.set_ylabel('Value')
ax.set_title('Values for Different Classes')
ax.legend()

# Show the plot
plt.show()
