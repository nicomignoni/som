import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec

# Load data
history   = np.load('history.npy')
train_pca = np.load('train_pca.npy')

# Border specs
n_frames = 90
h_border = 0.05
v_border = 0.1

# Model parameters
t_alpha, t_sigma = 25, 25
alpha0, sigma0 = 0.5, 2
t = np.linspace(0, n_frames+1, n_frames)
alpha_func = alpha0*np.exp(-t/t_alpha)
sigma_func = sigma0*np.exp(-t/t_sigma)

# Setting up the figure
fig   = plt.figure(figsize=[2.5*6.4, 4.8])
gs    = fig.add_gridspec(1, 4)
som   = fig.add_subplot(gs[0,:2])
alpha = fig.add_subplot(gs[0,2])
sigma = fig.add_subplot(gs[0,3])
fig.subplots_adjust(left=h_border, right=1-h_border,
                    bottom=v_border, top=1-v_border)

# SOM subplot
som.plot(train_pca[:, 0], train_pca[:, 1], 'bo', ms=6)
som.set_title("SOM training")
points, = som.plot([], [], 'ro', ms=3)

# Alpha subplot
alpha.plot(t, alpha_func)
alpha.set_ylim([0, max(sigma0, alpha0)])
alpha.set_title("Learning rate decay")
alpha_dot, = alpha.plot([], [], 'bo', ms=5)

# Sigma subplot
sigma.plot(t, sigma_func)
sigma.set_title("Neighborhood radius decay")
sigma.set_ylim([0, max(sigma0, alpha0)])
sigma_dot, = sigma.plot([], [], 'bo', ms=5)

def init():
    points.set_data([], [])
    sigma_dot.set_data([], [])
    alpha_dot.set_data([], [])
    return points, alpha_dot, sigma_dot 

def animate(i):
    points.set_data(history[i, :, 0], history[i, :, 1])
    alpha_dot.set_data(t[i], alpha_func[i])
    sigma_dot.set_data(t[i], sigma_func[i])
    return points, alpha_dot, sigma_dot

anim = animation.FuncAnimation(fig, animate, blit=True, init_func=init,
                             frames=n_frames, interval=50)
plt.show()
