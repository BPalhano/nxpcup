import matplotlib.pyplot as plt
import numpy as np

lat_prop     = np.load("latency_mc_proposed.npy")
lat_dave2    = np.load("latency_mc_dave2.npy")
lat_prop_512 = np.load("latency_mc_proposed_512.npy")

print("Médias de Latência (ms) normalizadas por DAVE-2:")
print(f"REDE PROPOSTA: {(lat_prop.mean() / lat_dave2.mean()- 1)*100:.2f}% mais lento")
print(f"REDE PROPOSTA (512): {(lat_prop_512.mean() / lat_dave2.mean()- 1)*100:.2f}% mais lento\n")

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

#axes[0].hist(lat_prop, bins=50, alpha=0.7, label="REDE PROPOSTA", 
#             color='blue')
axes[0].hist(lat_dave2, bins=50, alpha=0.7, label="DAVE-2", 
             color='orange')
axes[0].hist(lat_prop_512, bins=50, alpha=0.7, label="Proposed achitecture", 
             color='green')


axes[0].set_title("Latency distribuction")
axes[0].set_xlabel("Latency [ms]")
axes[0].set_ylabel("Probability Density")
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim(75, 20)

boxplot_data = [lat_dave2, lat_prop_512]
bp = axes[1].boxplot(boxplot_data, labels=["DAVE-2", "Proposed architecture"], patch_artist=True)
axes[1].set_title("Latency boxplot")
axes[1].set_ylabel("Latency [ms]")
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim(75, 200)

colors = ['lightblue', 'lightcoral']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

plt.tight_layout()
plt.show()
