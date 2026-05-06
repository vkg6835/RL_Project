import matplotlib.pyplot as plt
import os

def plot_error(errors):

    os.makedirs("results/plots",exist_ok=True)

    plt.plot(errors)
    plt.xlabel("Iteration")
    plt.ylabel("Angular Error")
    plt.title("Emotion Convergence")

    plt.savefig("results/plots/angular_error.png")

    plt.close()