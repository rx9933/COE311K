a = [2, 3, 45, 1]
b = [2, 3, 4, 1]

# Create a mapping between original and sorted indices
indices = list(range(len(a)))
indices.sort(key=lambda x: a[x])

# Sort list a
a = [a[i] for i in indices]

# Shuffle list b using the same mapping
b = [b[i] for i in indices]

print(a)  # Output: [1, 2, 3, 45]
print(b)  # Output: [0, 1, 0, 1]
