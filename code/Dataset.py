#%%
import random
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
import torch
from sklearn.mixture import GaussianMixture

from scipy.stats import multivariate_normal
from scipy.interpolate import griddata

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture


def dataset_GMM(n_samples= 1000, show_plot=True):
    """Samples data points from a predefined GM with 3 components and 4 dimensions 
    Plot the result dataset
    returns : (np.array , np.array) : X_train and X_test, the train and test dataset
    """
    # Define the GMM parameters (mean, covariance, and weights) for each component
    means = np.array([
        [2.5, 2.5, 1.0, 2.0],  # Mean for component 1
        [0.1, 0.1, 2.5, 1.5],  # Mean for component 2
        [-2.5, 2.5, -0.5, 1]   # Mean for component 3
    ])
    covariances = np.array([[
        [1.0, 0.2, 0.4, 0.3],
        [0.2, 2.0, 0.5, 0.2],
        [0.4, 0.5, 3.0, 0.6],
        [0.3, 0.2, 0.6, 1.5]
        ],
        [
        [1.0, -0.2, 0.4, 0.3],
        [-0.2, 2.0, 0.5, 0.2],
        [0.4, 0.5, 3.0, -0.6],
        [0.3, 0.2, -0.6, 1.5]
        ],
        [
        [1.5, -0.3, 0.1, -0.5],
        [-0.3, 2.0, 0.2, 0.4],
        [0.1, 0.2, 1.5, 0.3],
        [-0.5, 0.4, 0.3, 2.0]
        ]])
    
    weights = np.array([0.5, 0.5, 0.5])  # Weights for each component

    samples_0 = np.random.multivariate_normal(means[0], covariances[0], size=int(n_samples * weights[0]))
    samples_1 = np.random.multivariate_normal(means[1], covariances[1], size=int(n_samples * weights[1]))
    samples_2 = np.random.multivariate_normal(means[2], covariances[2], size=int(n_samples * weights[2]))
    data_points = np.vstack((samples_0,samples_1, samples_2))
    
    if show_plot:
        # Calculate the density for each data point in both GMMs
        density_0 = multivariate_normal.pdf(data_points, means[0], covariances[0])
        density_1 = multivariate_normal.pdf(data_points, means[1], covariances[1])
        density_2 = multivariate_normal.pdf(data_points, means[2], covariances[2])
        density = weights[0] * density_0 + weights[1] * density_1 + weights[2] * density_2 
        # Plotting part
        fig = plt.figure(1)
        ax = fig.add_subplot(111, projection='3d')
        x1 = data_points[:, 0]
        x2 = data_points[:, 1]
        z = density
        # Plot the data as a surface
        ax.scatter(x1, x2, z, c=z,cmap='viridis')
        # Set labels and title
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        ax.set_zlabel('Density')
        ax.set_title('Dataset')
        # Enable interactive mode
        ax.mouse_init()
        
        plt.figure(2)
        # Create a 2D plane for the contour plot
        x1_plane, x2_plane = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))
        # Calculate density for the contour plot on the plane
        plane_points = np.column_stack((x1_plane.ravel(), x2_plane.ravel()))
        density_plane_0 = multivariate_normal.pdf(plane_points, means[0][:2], covariances[0][:2, :2])
        density_plane_1 = multivariate_normal.pdf(plane_points, means[1][:2], covariances[1][:2, :2])
        density_plane_2 = multivariate_normal.pdf(plane_points, means[2][:2], covariances[2][:2, :2])
        density_plane = weights[0] * density_plane_0 + weights[1] * density_plane_1 + weights[2] * density_plane_2
        density_plane = density_plane.reshape(x1_plane.shape)
        # Create the contour plot
        plt.contourf(x1_plane, x2_plane, density_plane, levels=10, cmap='viridis')
        plt.colorbar()
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.title('Contour Plot of Density')

        # Plotting part
        fig = plt.figure(3)
        ax = fig.add_subplot(111, projection='3d')
        x3 = data_points[:, 2]
        x4 = data_points[:, 3]
        z = density
        # Plot the data as a surface
        ax.scatter(x3, x4, z, c=z,cmap='viridis')
        # Set labels and title
        ax.set_xlabel('X3')
        ax.set_ylabel('X4')
        ax.set_zlabel('Density')
        ax.set_title('Dataset')
        # Enable interactive mode
        ax.mouse_init()
        
        plt.figure(4)
        # Create a 2D plane for the contour plot
        x1_plane, x2_plane = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))
        # Calculate density for the contour plot on the plane
        plane_points = np.column_stack((x1_plane.ravel(), x2_plane.ravel()))
        density_plane_0 = multivariate_normal.pdf(plane_points, means[0][2:], covariances[0][2:, 2:])
        density_plane_1 = multivariate_normal.pdf(plane_points, means[1][2:], covariances[1][2:, 2:])
        density_plane_2 = multivariate_normal.pdf(plane_points, means[2][2:], covariances[2][2:, 2:])
        density_plane = weights[0] * density_plane_0 + weights[1] * density_plane_1 + weights[2] * density_plane_2
        density_plane = density_plane.reshape(x1_plane.shape)
        # Create the contour plot
        plt.contourf(x1_plane, x2_plane, density_plane, levels=10, cmap='viridis')
        plt.colorbar()
        plt.xlabel('X3')
        plt.ylabel('X4')
        plt.title('Contour Plot of Density')
        
        # Show the plots
        plt.show()
    
    # Split train test dataset
    test_size = 0.2  # Test rate
    num_test_points = int(test_size * data_points.shape[0])
    np.random.shuffle(data_points)
    X_train = data_points[num_test_points:]
    X_test = data_points[:num_test_points] 
    return X_train, X_test


def fake_data_shuffle(real_data_tensor):
    """Schuffle the data randomely to create fake data.
    The data dimensio are kept , only the values are changing
    Args:
        real_data_tensor (2d torch tensor): the "xpos", a 2d pytorch tensor (multiples state action pairs) 
    Returns:
        pos_data, neg_data (torch tensor): the positive and negative data in torch tensors
    """
    # Transpose the tensor
    real_data_tensor_T_list = real_data_tensor.t().tolist()
    # Shuffle the values in each inner list
    for inner_list in real_data_tensor_T_list:
        random.shuffle(inner_list)
    
    fake_data_tensor = torch.tensor(real_data_tensor_T_list).T
    # Create a mask to remove the data that are similar 
    mask = torch.all(fake_data_tensor == real_data_tensor, dim=1)
    # Use the mask to remove rows from both tensors
    real_data= real_data_tensor[~mask]
    fake_data= fake_data_tensor[~mask]
    return real_data, fake_data

def fake_data_GGM(real_states, max_components_to_try=10):
    # TODO : this function is in creation
    """Fit a GMM to the real state data in order to create from thoses points some fake data points

    Args:
        real_states (torch matrix): matrix containing the real state value obersved
        max_components_to_try (int, optional): The number of gmm max to try to fit the dataset. Defaults to 10.
        n_samples (int, optional): number of fake samples generated. Defaults to 256.

    Returns:
        torch matrix: fake samples
    """
    n_samples = len(real_states)
    data = real_states.numpy()
    
    lowest_bic = np.inf
    best_n_components = 0
    # Fit GMMs for different component numbers and calculate AIC/BIC
    aic_scores = []
    bic_scores = []
    log_likelihoods = []
    for n_components in range(1, max_components_to_try + 1):
        gmm = GaussianMixture(n_components=n_components)
        gmm.fit(data)
        bic = gmm.bic(data)
        # TODO : choose and elbow/bic etc and justify it
        aic_scores.append(gmm.aic(data))
        bic_scores.append(gmm.bic(data))
        # if bic < lowest_bic:
        #     lowest_bic = bic
        #     best_n_components = n_components
        log_likelihoods.append(gmm.score_samples(data).mean())
    
    plt.plot(range(1, max_components_to_try + 1), log_likelihoods, label='log_likelihoods')
    #plt.plot(range(1, max_components_to_try + 1), aic_scores, label='AIC')
    #plt.plot(range(1, max_components_to_try + 1), bic_scores, label='BIC')
    plt.xlabel('Number of Components')
    plt.ylabel('Score')
    plt.legend()
    plt.show()        
    # Fit the best model to our dataset
    print(' Number of kernels :', best_n_components)
    best_gmm = GaussianMixture(n_components=3)
    best_gmm.fit(data)
    
    # Create n fake samples
    new_samples = best_gmm.sample(n_samples)[0]
    
    # Put it in a torch tensor
    fake_states = torch.from_numpy(new_samples).float()

    return real_states, fake_states, log_likelihoods



if __name__ == '__main__':
    # generate nonlinearly correlated data (to be used as positive data in a simulated dataset)
    # visualize the nonlinear relation between inputs with a simulated dataset with just 2 features
    data_points = dataset_GMM(n_samples= 1000, show_plot=True)
    df =  pd.DataFrame(data_points, columns=['dim1', 'dim2', 'dim3', 'dim4'])
    

#%%
