from scipy.io import loadmat
import matplotlib.pyplot as plt

def get_signal(filepath):
    mat = loadmat(filepath)
    de_key = [key for key in mat.keys() if 'DE_time' in key][0]
    return mat[de_key].flatten()

# Load both a faulty and a healthy signal
faulty = get_signal('data/B007_1_123.mat')
healthy = get_signal('data/Time_Normal_1_098.mat')

# Plot both side by side for comparison
fig, axes = plt.subplots(2, 1, figsize=(10, 6))

axes[0].plot(healthy[:2000], color='green')
axes[0].set_title("Healthy Bearing")
axes[0].set_ylabel("Amplitude")

axes[1].plot(faulty[:2000], color='red')
axes[1].set_title("Faulty Bearing (Ball Fault, 0.007in)")
axes[1].set_xlabel("Sample number")
axes[1].set_ylabel("Amplitude")

plt.tight_layout()
plt.savefig('comparison_plot.png')
print("Saved comparison_plot.png")