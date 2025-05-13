import numpy as np 
from scipy import stats
import matplotlib.pyplot as plt
import torch
from scipy.stats import norm, uniform, invgauss, gaussian_kde
from matplotlib.ticker import FormatStrFormatter
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter


class VisaulizeResults:

    @staticmethod
    def plot_perturbation(data1, data2=None):
        # Check if data has at least 6 columns
        if data1.shape[1] >= 6:
            # Set up the plot grid
            fig, axes = plt.subplots(3, 2, figsize=(8, 10))  # 3 rows, 2 columns

            # Define titles for subplots
            titles = ['Rotation X', 'Rotation Y', 'Rotation Z', 'Translation X', 'Translation Y', 'Translation Z']
            y_axis = ['Rotation X', 'Rotation Y', 'Rotation Z', 'Translation X', 'Translation Y', 'Translation Z']
            ticks = np.linspace(-0.5, 0.5, 9)
            # Loop over each subplot and plot data
            for i in range(3):
                # Left column for rotation (roll, pitch, yaw)
                axes[i, 0].scatter(range(len(data1)), data1[:, i],s=1, color='#0065BD', label='Input Perturbation')
                if data2 is not None:
                    axes[i, 0].scatter(range(len(data2)), data2[:, i], s=1, color='#E37222', alpha=0.5, label='Output Results')
                axes[i, 0].set_title(titles[i])
                axes[i, 0].set_title(y_axis[i])
                axes[i, 0].set_xlabel('Data Samples')
                axes[i, 0].set_ylabel(y_axis[i])
                axes[i, 0].legend()
                
                axes[i, 0].set_ylim(-20,20)
                axes[i, 0].autoscale(enable=False)


                # Right column for translation (x, y, z)
                axes[i, 1].scatter(range(len(data1)), data1[:, i+3],s=1, color='#0065BD', label='Input Perturbation')
                if data2 is not None:
                    axes[i, 1].scatter(range(len(data2)), data2[:, i+3],s=1,  color='#E37222', alpha=0.5, label='Output Results')
                axes[i, 1].set_title(titles[i+3])
                axes[i, 1].set_title(y_axis[i+3])
                axes[i, 1].set_xlabel('Data Samples')
                axes[i, 1].set_ylabel(y_axis[i+3])
                axes[i, 1].legend()
                axes[i, 1].set_ylim(-0.5,0.5)
                axes[i, 1].set_yticks(ticks)
                axes[i, 1].autoscale(enable=False)

                
            # Adjust layout and show plot
            plt.tight_layout()
            plt.savefig("plot_perturb.pdf", dpi=300,bbox_inches='tight')
            plt.show()
        else:
            print("The array must have at least 6 columns to plot in subplots.")
    
    @staticmethod
    def plot_calibration_correlation(initial_errors, predicted_errors):
        """
        Creates a 3x2 grid of scatter plots.
        For each calibration parameter, each sample is represented by a point
        with the x value equal to the initial error and the y value equal to the predicted error.
        
        Parameters:
        initial_errors: numpy array of shape (N, 6) for initial errors.
        predicted_errors: numpy array of shape (N, 6) for predicted errors.
        The 6 columns represent [Rot X, Rot Y, Rot Z, Trans X, Trans Y, Trans Z].
        """
        if initial_errors.shape[1] < 6 or predicted_errors.shape[1] < 6:
            print("Both arrays must have at least 6 columns.")
            return

        # Create a 3x2 grid of subplots
        fig, axes = plt.subplots(3, 2, figsize=(12, 12))
        
        # Define labels for rotation and translation
        rotation_labels = ['Rot X', 'Rot Y', 'Rot Z']
        translation_labels = ['Trans X', 'Trans Y', 'Trans Z']
        
        # Loop through the 3 rows. Left column (index 0) for rotations, right column (index 1) for translations.
        for row in range(3):
            # Rotation plots: use columns 0 for rotations.
            ax_rot = axes[row, 0]
            ax_rot.scatter(initial_errors[:, row], predicted_errors[:, row],
                        s=5, alpha=0.7, color='#0065BD')
            ax_rot.set_xlabel('Initial ' + rotation_labels[row] + ' [deg]')
            ax_rot.set_ylabel('Predicted ' + rotation_labels[row] + ' [deg]')
            ax_rot.set_title(rotation_labels[row])
            ax_rot.set_xlim(-20, 20)
            
            # Translation plots: use column 1 for translations.
            ax_trans = axes[row, 1]
            ax_trans.scatter(initial_errors[:, row + 3], predicted_errors[:, row + 3],
                            s=5, alpha=0.7, color='#0065BD')
            ax_trans.set_xlabel('Initial ' + translation_labels[row] + ' [m]')
            ax_trans.set_ylabel('Predicted ' + translation_labels[row] + ' [m]')
            ax_trans.set_title(translation_labels[row])
            ax_trans.set_xlim(-0.5, 0.5)
        
        plt.tight_layout()
        plt.savefig("calibration_correlation.pdf", dpi=300)
        plt.show()


    @staticmethod
    def analyze_data_distribution(data, data2=None, data_name = "Input Perturbation", data2_name = "Output Results"):
        #data = np.nan_to_num(data)  # Replace NaNs and infinities with numerical values

        num_cols = data.shape[1]

        y_lab = ['roll', 'pitch', 'yaw', 't_x', 't_y', 't_z',]

        fig, axes = plt.subplots(num_cols, 2, figsize=(16, 16))
        fig.subplots_adjust(hspace=0.5)
        
        # for col in range(num_cols):
        #     mean = np.mean(data[:, col])
        #     std = np.std(data[:, col])

        #     if std == 0 or np.isnan(std):
        #         print(f"Standard deviation for column {col} is {std}, which is not valid for PDF calculation.")
        #         continue
    # ...
                
        for col in range(num_cols):
            # Histogram for data
            axes[col, 0].hist(data[:, col], bins=20, fc=(1, 0, 0, 0.5), ls='dotted', lw=1, edgecolor='black', density=True, label=data_name)
            axes[col, 0].set_title(f'{y_lab[col]} - Histogram')

            if data2 is not None:
                # Histogram for data2
                axes[col, 0].hist(data2[:, col], bins=20, fc=(0, 0, 1, 0.5), ls='dashed', lw=1, edgecolor='black', density=True, label=data2_name)

            # Box Plot
            axes[col, 1].boxplot(data[:, col], vert=False)
            axes[col, 1].set_title(f'{y_lab[col]} - Box Plot')

            # Add a nominal Gaussian curve to the histogram plot
            mean, std = np.mean(data[:, col]), np.std(data[:, col])
            x = np.linspace(np.min(data[:, col]), np.max(data[:, col]), 100)
            y = stats.norm.pdf(x, mean, std)
            axes[col, 0].plot(x, y, 'r--', label='Gaussian Curve')
            
            if data2 is not None:
                mean, std = np.mean(data2[:, col]), np.std(data2[:, col])
                x = np.linspace(np.min(data2[:, col]), np.max(data2[:, col]), 100)
                y = stats.norm.pdf(x, mean, std)
                axes[col, 0].plot(x, y, 'b--', label='Gaussian Curve')
               
            # # Compute min and max for the uniform distribution parameters
            # a = np.min(data[:, col])
            # b = np.max(data[:, col])
            
            # # Generate x values over the range [a, b]
            # x = np.linspace(a, b, 100)
            # # Calculate the uniform PDF: f(x)=1/(b-a) for x in [a, b]
            # y = stats.uniform.pdf(x, loc=a, scale=b - a)
            
            # # Plot the uniform PDF curve
            # axes[col, 0].plot(x, y, 'b--', label='Uniform PDF')




            # Set axis labels and legends
            if col >=3:
                axes[col, 0].set_xlabel('meters')
            else: 
                axes[col, 0].set_xlabel('deg')

            if col >=3:
                axes[col, 1].set_xlabel('meters')
            else: 
                axes[col, 1].set_xlabel('deg')

            axes[col, 0].set_ylabel('Density Data Samples')
            axes[col, 0].legend()

        plt.show()
        
        
        def analyze_data_distribution(data, data2=None):
            num_cols = data.shape[1]
            y_lab = ['roll', 'pitch', 'yaw', 't_x', 't_y', 't_z']

            fig, axes = plt.subplots(num_cols, 2, figsize=(16, 16))
            fig.subplots_adjust(hspace=0.5)



    @staticmethod
    def visualize_feature_maps(model, input_data, device):

        # Define a hook function to capture feature maps
        def hook_fn(module, input, output):
            feature_maps = output.detach()
            activations.append(feature_maps)

        # List to store feature maps for each layer
        activations = []

        # Register the hook for the layers you want to visualize
        for layer in model.modules():
            if isinstance(layer, torch.nn.Conv2d):
                layer.register_forward_hook(hook_fn)

        # Pass the input through the model to trigger the forward pass
        with torch.no_grad():
            rgb_img = input_data['img'].to(device)
            uncalibed_depth_img = input_data['uncalibed_depth_img'].to(device)
            twist_rot, twist_tsl = model(rgb_img,uncalibed_depth_img)

        # Visualize the feature maps for each layer
        for i, feature_map in enumerate(activations):
            num_feature_maps = feature_map.shape[1]

            # Calculate the number of rows and columns for the subplots
            rows = num_feature_maps // 8
            cols = min(8, num_feature_maps)

            plt.figure(figsize=(16, 16))
            for j in range(num_feature_maps):
                plt.subplot(rows, cols, j + 1)
                plt.imshow(feature_map[0, j].cpu(), cmap='viridis')
                plt.axis('off')
            plt.suptitle(f"Layer {i + 1}")
            plt.show()


    @staticmethod   
    def visualize_block_activations(model, input_data, device):
        # Define a hook function to capture block-level activations
        def hook_fn(module, input, output):
            activations.append(output)

        # List to store activations for each block
        activations = []

        # Register the hook for the layers representing blocks
        for block_num, block in enumerate(model.feature_extractor.rgb_resnet.layer1):
            block.conv2.register_forward_hook(hook_fn)

        for block_num, block in enumerate(model.feature_extractor.rgb_resnet.layer2):
            block.conv2.register_forward_hook(hook_fn)

        for block_num, block in enumerate(model.feature_extractor.rgb_resnet.layer3):
            block.conv2.register_forward_hook(hook_fn)

        for block_num, block in enumerate(model.feature_extractor.rgb_resnet.layer4):
            block.conv2.register_forward_hook(hook_fn)

        # Register the hook for the layers representing blocks
        for block_num, block in enumerate(model.feature_extractor.depth_resnet[1].layer1):
            block.conv2.register_forward_hook(hook_fn)

        for block_num, block in enumerate(model.feature_extractor.depth_resnet[1].layer2):
            block.conv2.register_forward_hook(hook_fn)

        for block_num, block in enumerate(model.feature_extractor.depth_resnet[1].layer3):
            block.conv2.register_forward_hook(hook_fn)

        for block_num, block in enumerate(model.feature_extractor.depth_resnet[1].layer4):
            block.conv2.register_forward_hook(hook_fn)

        # Pass the input through the model to trigger the forward pass
        with torch.no_grad():
            rgb_img = input_data['img'].to(device)
            uncalibed_depth_img = input_data['uncalibed_depth_img'].to(device)
            twist_rot, twist_tsl = model(rgb_img,uncalibed_depth_img)

        # Visualize the activations for each block
        for i, activation in enumerate(activations):
            plt.figure(figsize=(16, 16))
            num_feature_maps = activation.shape[1]
            for j in range(num_feature_maps):
                plt.subplot(num_feature_maps // 8, 8, j + 1)
                plt.imshow(activation[0, j].cpu(), cmap='viridis')
                plt.axis('off')
            plt.suptitle(f"Block {i + 1} Activations")
            plt.show()


    @staticmethod
    def visualize_final_activations_and_feature_maps(model, input_data, device):
        # Define a hook function to capture the final activation
        def hook_fn(module, input, output):
            activations.append(output)

        # List to store final activations for each part
        activations = []

        # Register hooks for the final layers of rgb_resnet, depth_resnet, and aggregation
        model.feature_extractor.rgb_resnet.layer4[-1].register_forward_hook(hook_fn)
        model.depth_resnet[1].layer4[-1].register_forward_hook(hook_fn)
        model.downsamplerNew.conv10.register_forward_hook(hook_fn)
        model.regression.tr_conv.register_forward_hook(hook_fn)
        model.regression.rot_conv.register_forward_hook(hook_fn)
        

        # Pass the input through the model to trigger the forward pass
        with torch.no_grad():
            rgb_img = input_data['img'].to(device)
            uncalibed_depth_img = input_data['uncalibed_depth_img'].to(device)
            twist_rot, twist_tsl = model(rgb_img,uncalibed_depth_img)
        
        titles = ["rgb_resnet - Final Activation", "depth_resnet - Final Activation", "aggregation - Final Activation", "aggregation - Translation Activation", "aggregation - Rotation Activation"]
        # Visualize the final activations
        for i, activation in enumerate(activations):
            plt.figure(figsize=(8, 8))
            plt.imshow(activation[0, 0].cpu(), cmap='viridis')  # Assuming the activation is 2D
            plt.axis('off')
            plt.title(titles[i])
            plt.show()
            
                
    @staticmethod
    def plot_rel_error(rot_improvement, trans_improvement):                
        
        improvements = np.concatenate([rot_improvement, trans_improvement])
        labels = ['Rot X', 'Rot Y', 'Rot Z', 'Trans X', 'Trans Y', 'Trans Z']

        # Create the bar chart
        plt.figure(figsize=(10, 6))
        bar_positions = np.arange(len(labels))
        plt.bar(bar_positions, improvements, color='blue', alpha=0.7)

        # Add a straight line at 100%
        plt.axhline(y=100, color='r', linestyle='--')

        # Set plot details
        plt.xlabel('Error Type')
        plt.ylabel('Improvement (%)')
        plt.title('Improvement in Rotation and Translation Errors')
        plt.xticks(bar_positions, labels)
        plt.ylim(-50, 200)  # Set y-axis limits from -50% to 200%

        # Show the plot
        plt.show()
                
    @staticmethod
    def plot_abs_error(initial_rot_error, predicted_rot_error, initial_trans_error, predicted_trans_error):  
                
        # Concatenate errors for plotting
        errors = np.concatenate([initial_rot_error, predicted_rot_error, initial_trans_error, predicted_trans_error])
        labels = ['Rot X', 'Rot Y', 'Rot Z', 'Trans X', 'Trans Y', 'Trans Z']

        # Split the labels for rotation and translation
        rot_labels = labels[:3]
        trans_labels = labels[3:]

        # Create the plot
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Plot rotation errors
        rot_positions = np.arange(len(rot_labels))
        ax1.scatter(rot_positions - 0.1, initial_rot_error, color='red', label='Initial Error')
        ax1.scatter(rot_positions + 0.1, predicted_rot_error, color='green', label='Predicted Error')

        # Set plot details for rotation
        ax1.set_xlabel('Error Type')
        ax1.set_ylabel('Rotation Error (degrees)', color='black')
        ax1.tick_params(axis='y')
        ax1.set_xticks(np.arange(len(labels)))
        ax1.set_xticklabels(labels)
        ax1.set_ylim(-10, 10)  # Set y-axis limits for rotation
        ax1.axhline(y=0, color='black', linestyle='--')  # Add a line at y=0

        # Create a second y-axis for translation errors
        ax2 = ax1.twinx()

        # Plot translation errors
        trans_positions = np.arange(len(trans_labels)) + 3  # Offset by 3 for the next set of labels
        ax2.scatter(trans_positions - 0.1, initial_trans_error, color='red')
        ax2.scatter(trans_positions + 0.1, predicted_trans_error, color='green')

        # Set plot details for translation
        ax2.set_ylabel('Translation Error (meters)', color='black')
        ax2.tick_params(axis='y')
        ax2.set_ylim(-0.2, 0.2)  # Set y-axis limits for translation

        # Add a legend
        fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=2)

        # Show the plot
        plt.title('Initial and Corrected Errors in Rotation and Translation')
        plt.show()
    
    @staticmethod
    def plot_abs_error2(initial_decalib, predicted_decalib):
        """
        Plots initial vs. predicted errors for both rotation and translation.
        
        Parameters:
        initial_decalib: numpy array of shape (N, 6), where columns 0-2 are rotation errors (degrees)
                        and columns 3-5 are translation errors (meters).
        predicted_decalib: numpy array of shape (N, 6) with the same layout.
        """
        # Split the arrays into rotation and translation components
        initial_rot_error = initial_decalib[:, :3]    # (N,3)
        initial_trans_error = initial_decalib[:, 3:]    # (N,3)
        predicted_rot_error = predicted_decalib[:, :3]  # (N,3)
        predicted_trans_error = predicted_decalib[:, 3:]  # (N,3)
        
        labels = ['Rot X', 'Rot Y', 'Rot Z', 'Trans X', 'Trans Y', 'Trans Z']
        
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # Plot rotation errors on ax1 (x positions 0,1,2)
        for i in range(3):
            # Create an array of x coordinates for this axis (all points at the same x)
            x_initial = np.full(initial_rot_error.shape[0], i - 0.1)
            x_predicted = np.full(predicted_rot_error.shape[0], i + 0.1)
            # Plot scatter points for initial and predicted rotation errors
            ax1.scatter(x_initial, initial_rot_error[:, i], color='red', label='Initial Rotation Error' if i == 0 else "")
            ax1.scatter(x_predicted, predicted_rot_error[:, i], color='green', label='Predicted Rotation Error' if i == 0 else "")
        
        ax1.set_xlabel('Error Type')
        ax1.set_ylabel('Rotation Error (degrees)', color='black')
        # We want xticks at positions 0, 1, 2 for rotation and 3, 4, 5 for translation
        ax1.set_xticks(np.arange(6))
        ax1.set_xticklabels(labels)
        ax1.set_ylim(-10, 10)
        ax1.axhline(y=0, color='black', linestyle='--')
        
        # Create a secondary y-axis for translation errors (x positions 3,4,5)
        ax2 = ax1.twinx()
        for i in range(3):
            x_initial = np.full(initial_trans_error.shape[0], i + 3 - 0.1)
            x_predicted = np.full(predicted_trans_error.shape[0], i + 3 + 0.1)
            ax2.scatter(x_initial, initial_trans_error[:, i], color='red', label='Initial Translation Error' if i == 0 else "")
            ax2.scatter(x_predicted, predicted_trans_error[:, i], color='green', label='Predicted Translation Error' if i == 0 else "")
        
        ax2.set_ylabel('Translation Error (meters)', color='black')
        ax2.set_ylim(-0.5, 0.5)
        
        # Combine legends from both axes and set the title
        fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=2)
        plt.title('Initial and Corrected Errors in Rotation and Translation')
        plt.show()



    @staticmethod
    def analyze_data_distribution2(data, data2=None, data3=None, data4=None, 
                              data_name="Input Perturbation", 
                              data2_name="Output Results",
                              data3_name="Data3",
                              data4_name="Data4"):
        """
        Plots Gaussian curves for 6 error components using a 3x2 grid.
        
        Parameters:
        data: numpy array (N,6) where columns 0-2 are rotation errors (in deg)
                and columns 3-5 are translation errors (in meters).
        data2, data3, data4: optional numpy arrays (N,6) with similar layout.
        data_name, data2_name, data3_name, data4_name: labels for each dataset.
        """
        # Create a 3x2 grid: left column for rotation (roll, pitch, yaw),
        # right column for translation (t_x, t_y, t_z)
        fig, axes = plt.subplots(3, 2, figsize=(16, 16))
        fig.subplots_adjust(hspace=0.5, wspace=0.3)
        
        # Labels for rows
        rotation_labels = ['roll', 'pitch', 'yaw']
        translation_labels = ['t_x', 't_y', 't_z']
        
        # Define datasets with color, linestyle, and label.
        datasets = [
            {"data": data,   "label": data_name,   "color": "black",    "linestyle": "-"},
            {"data": data2,  "label": data2_name,  "color": "#0065BD",   "linestyle": "--"},
            {"data": data3,  "label": data3_name,  "color": "#003359",  "linestyle": "-."},
            {"data": data4,  "label": data4_name,  "color": "#E37222", "linestyle": "-"}
        ]
        
        # Plot rotation distributions (columns 0-2)
        for row in range(3):
            ax = axes[row, 0]
            for ds in datasets:
                if ds["data"] is not None:
                    # Rotation error is in column index = row (0: roll, 1: pitch, 2: yaw)
                    d = ds["data"][:, row]
                    mean = np.mean(d)
                    std = np.std(d)
                    # Create x range for plotting the Gaussian PDF
                    x = np.linspace(np.min(d), np.max(d), 100)
                    y = stats.norm.pdf(x, mean, std)
                    ax.plot(x, y, color=ds["color"], linestyle=ds["linestyle"], label=ds["label"])
            ax.set_title(f'{rotation_labels[row]} Distribution')
            ax.set_xlabel('Decalibration [deg]')
            ax.set_ylabel('Density')
            ax.set_ylim(0,4)
            ax.set_xlim(-20, 20)
            ax.legend()
            
        # Plot translation distributions (columns 3-5)
        for row in range(3):
            ax = axes[row, 1]
            for ds in datasets:
                if ds["data"] is not None:
                    # Translation error is in column index = row + 3 (3: t_x, 4: t_y, 5: t_z)
                    d = ds["data"][:, row + 3]
                    mean = np.mean(d)
                    std = np.std(d)
                    x = np.linspace(np.min(d), np.max(d), 100)
                    y = stats.norm.pdf(x, mean, std)
                    ax.plot(x, y, color=ds["color"], linestyle=ds["linestyle"], label=ds["label"])
            ax.set_title(f'{translation_labels[row]} Distribution')
            ax.set_xlabel('Decalibration [m]')
            ax.set_ylabel('Density')
            ax.set_xlim(-0.5, 0.5)
            ax.set_ylim(0,14)
            ax.legend()
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def analyze_data_distribution_kde(data, data2=None, data3=None, data4=None, 
                                  data_name="Data1", data2_name="Data2", 
                                  data3_name="Data3", data4_name="Data4"):
        """
        Plots KDE curves for 6 error components using a 3x2 grid.
        Left column: rotation (roll, pitch, yaw) in degrees.
        Right column: translation (t_x, t_y, t_z) in meters.
        
        Each subplot will display up to 4 KDE curves (one per dataset) with
        distinct color and linestyle.
        """
        # Create a 3x2 grid of subplots.
        fig, axes = plt.subplots(3, 2, figsize=(16, 16))
        fig.subplots_adjust(hspace=0.5, wspace=0.3)
        
        # Define labels for rows.
        rotation_labels = ['Rotation X', 'Rotation Y', 'Rotation Z']
        translation_labels = ['Translation X', 'Translation Y', 'Translation Z']
        
        # Prepare datasets with properties.
        datasets = [
            {"data": data,   "label": data_name,   "color": "black",    "linestyle": "-"},
            {"data": data2,  "label": data2_name,  "color": "#0065BD",   "linestyle": "--"},
            {"data": data3,  "label": data3_name,  "color": "#A2AD00",  "linestyle": "-."},
            {"data": data4,  "label": data4_name,  "color": "#E37222", "linestyle": "-"}
        ]
        
        # For rotation subplots: x-axis limits [-20,20] and manually set ticks including endpoints.
        rotation_ticks = np.linspace(-20, 20, 9)  # e.g., [-20, -10, 0, 10, 20]
        # For translation subplots: x-axis limits [-0.5,0.5]
        translation_ticks = np.linspace(-0.5, 0.5, 9)  # e.g., [-0.5, -0.25, 0, 0.25, 0.5]
        # Plot KDE curves for rotation errors (columns 0-2)

        for row in range(3):
            ax = axes[row, 0]
            for ds in datasets:
                if ds["data"] is not None:
                    # Extract rotation error data: column index = row (0: roll, 1: pitch, 2: yaw)
                    d = ds["data"][:, row]
                    # Fit KDE to the data.
                    density = gaussian_kde(d)
                    # Generate x values over the data range.
                    x = np.linspace(np.min(d), np.max(d), 100)
                    y = density(x)
                    ax.plot(x, y, color=ds["color"], linestyle=ds["linestyle"], label=ds["label"])
            ax.set_title(f'{rotation_labels[row]}')
            ax.set_xlabel('Decalibration [deg]')
            ax.set_xticks(rotation_ticks)
            ax.set_ylabel('Density')
            ax.set_xlim(-20, 20)
            #ax.set_ylim(0,4)
            ax.set_ylim(bottom=0)
            ax.legend()
            
        # Plot KDE curves for translation errors (columns 3-5)
        for row in range(3):
            ax = axes[row, 1]
            for ds in datasets:
                if ds["data"] is not None:
                    # Extract translation error data: column index = row + 3 (3: t_x, 4: t_y, 5: t_z)
                    d = ds["data"][:, row + 3]
                    density = gaussian_kde(d)
                    x = np.linspace(np.min(d), np.max(d), 100)
                    y = density(x)
                    ax.plot(x, y, color=ds["color"], linestyle=ds["linestyle"], label=ds["label"])
            ax.set_title(f'{translation_labels[row]}')
            ax.set_xlabel('Decalibration [m]')
            ax.set_xticks(translation_ticks)
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax.set_ylabel('Density')
            ax.set_xlim(-0.5, 0.5)
            ax.set_ylim(0,14)
            #ax.set_ylim(bottom=0)
            ax.legend()
        
        plt.tight_layout()
        plt.show()

    
    @staticmethod
    def analyze_data_distribution_kde2(data, data2=None, data3=None, data4=None, 
                                  data_name="Data1", data2_name="Data2", 
                                  data3_name="Data3", data4_name="Data4"):
        """
        Plots KDE curves for 6 error components using a 3x2 grid.
        Left column: rotation (roll, pitch, yaw) in degrees.
        Right column: translation (t_x, t_y, t_z) in meters.
        
        Each subplot will display up to 4 KDE curves (one per dataset) with
        distinct color and linestyle.
        
        For the first dataset, instead of fitting the KDE directly to the provided data,
        we first generate a uniform sample over the range of the data and then fit a KDE.
        """
        # Create a 3x2 grid of subplots.
        fig, axes = plt.subplots(3, 2, figsize=(10, 12))
        fig.subplots_adjust(hspace=0.5, wspace=0.3)
        
        # Define labels for rows.
        rotation_labels = ['Rotation X', 'Rotation Y', 'Rotation Z']
        translation_labels = ['Translation X', 'Translation Y', 'Translation Z']
        
        # Prepare datasets with properties.
        datasets = [
            {"data": data,   "label": data_name,   "color": "black",    "linestyle": "-"},
            {"data": data2,  "label": data2_name,  "color": "#0065BD",   "linestyle": "--"},
            {"data": data3,  "label": data3_name,  "color": "#A2AD00",  "linestyle": "-."},
            {"data": data4,  "label": data4_name,  "color": "#E37222", "linestyle": "-"}
        ]
        
        # For rotation subplots: x-axis limits [-20,20] and manually set ticks.
        rotation_ticks = np.linspace(-20, 20, 9)
        # For translation subplots: x-axis limits [-0.5,0.5].
        translation_ticks = np.linspace(-0.5, 0.5, 9)
        
        # Plot KDE curves for rotation errors (left column).
        for row in range(3):
            ax = axes[row, 0]
            for idx, ds in enumerate(datasets):
                if ds["data"] is not None:
                    # Extract rotation error data: column index = row (0: roll, 1: pitch, 2: yaw)
                    d = ds["data"][:, row]
                    # For the first dataset, generate a uniform sample over [min, max]
                    if idx == 0:
                        low, high = np.min(d), np.max(d)
                        # Generate a uniform sample of the same length as d.
                        uniform_sample = np.random.uniform(low, high, size=len(d))
                        density = gaussian_kde(uniform_sample)
                    else:
                        density = gaussian_kde(d)
                    x = np.linspace(np.min(d), np.max(d), 100)
                    y = density(x)
                    ax.plot(x, y, color=ds["color"], linestyle=ds["linestyle"], label=ds["label"])
            ax.set_title(rotation_labels[row],  fontsize=10)
            ax.set_xlabel('Decalibration [deg]',  fontsize=10)
            ax.set_xticks(rotation_ticks,  fontsize=8)
            ax.set_ylabel('Density',  fontsize=10)
            ax.set_xlim(-20, 20)
            ax.set_ylim(bottom=0)
            ax.legend( fontsize=9)
            
        # Plot KDE curves for translation errors (right column).
        for row in range(3):
            ax = axes[row, 1]
            for idx, ds in enumerate(datasets):
                if ds["data"] is not None:
                    # Extract translation error data: column index = row + 3 (3: t_x, 4: t_y, 5: t_z)
                    d = ds["data"][:, row + 3]
                    if idx == 0:
                        low, high = np.min(d), np.max(d)
                        uniform_sample = np.random.uniform(low, high, size=len(d))
                        density = gaussian_kde(uniform_sample)
                    else:
                        density = gaussian_kde(d)
                    x = np.linspace(np.min(d), np.max(d), 100)
                    y = density(x)
                    ax.plot(x, y, color=ds["color"], linestyle=ds["linestyle"], label=ds["label"])
            ax.set_title(translation_labels[row],  fontsize=10)
            ax.set_xlabel('Decalibration [m]' , fontsize=10)
            ax.set_xticks(translation_ticks, fontsize=8)
            ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
            ax.set_ylabel('Density',  fontsize=10)
            ax.set_xlim(-0.5, 0.5)
            ax.set_ylim(0, 14)
            ax.legend(fontsize=9)
        
        plt.tight_layout()
        plt.savefig("kde_distribution.pdf", dpi=300, bbox_inches='tight')
        plt.show()

    
    @staticmethod
    def analyze_avg_distribution_kde(data, data2=None, data3=None, data4=None,data5=None, data6=None, 
                                 data_name="Data1", data2_name="Data2", 
                                 data3_name="Data3", data4_name="Data4",
                                 data5_name="Data5", data6_name="Data6"):
        """
        Given datasets that are (N,2) arrays where:
        - Column 0: average rotation error (in degrees)
        - Column 1: average translation error (in meters)
        
        Plots KDE curves for rotation and translation errors on a 1x2 grid.
        """
        datasets = [
            {"data": data,   "label": data_name,   "color": "#0065BD",    "linestyle": "--"},
            {"data": data2,  "label": data2_name,  "color": "#A2AD00",   "linestyle": "--"},
            {"data": data3,  "label": data3_name,  "color": "#E37222",  "linestyle": "--"},
            {"data": data4,   "label": data4_name,   "color": "#0065BD",    "linestyle": "-"},
            {"data": data5,  "label": data5_name,  "color": "#A2AD00",   "linestyle": "-"},
            {"data": data6,  "label": data6_name,  "color": "#E37222",  "linestyle": "-"},
            
        ]
        
        # For rotation subplots: x-axis limits [-20,20] and manually set ticks including endpoints.
        rotation_ticks = np.linspace(-20, 20, 9)  # e.g., [-20, -10, 0, 10, 20]
        # For translation subplots: x-axis limits [-0.5,0.5]
        translation_ticks = np.linspace(-0.5, 0.5, 9)  # e.g., [-0.5, -0.25, 0, 0.25, 0.5]
        # Plot KDE curves for rotation errors (columns 0-2)

        # Create a 1x2 grid: left for rotation, right for translation.
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # For rotation errors (first column) with x-axis limits [-20, 20]
        ax_rot = axes[0]
        for ds in datasets:
            if ds["data"] is not None:
                d = ds["data"][:, 0]  # Average rotation errors
                density = gaussian_kde(d)
                x = np.linspace(-20, 20, 100)
                y = density(x)
                ax_rot.plot(x, y, color=ds["color"], linestyle=ds["linestyle"], label=ds["label"])
        #ax_rot.set_title("Average Rotation Error KDE")
        ax_rot.set_xlabel("Rotation Error (degrees)")
        ax_rot.set_ylabel("Density")
        #ax_rot.set_xticks(rotation_ticks)
        ax_rot.set_xlim(0, 20)
        ax_rot.set_ylim(bottom=0)
        ax_rot.legend()
        
        # For translation errors (second column) with x-axis limits [-0.5, 0.5]
        ax_trans = axes[1]
        for ds in datasets:
            if ds["data"] is not None:
                d = ds["data"][:, 1]  # Average translation errors
                density = gaussian_kde(d)
                x = np.linspace(-0.5, 0.5, 100)
                y = density(x)
                ax_trans.plot(x, y, color=ds["color"], linestyle=ds["linestyle"], label=ds["label"])
        #ax_trans.set_title("Average Translation Error KDE")
        ax_trans.set_xlabel("Translation Error [m]")
        ax_trans.set_ylabel("Density")
        ax_trans.set_xlim(0, 0.5)
        #ax_trans.set_xticks(translation_ticks)
        ax_trans.set_ylim(bottom=0)
        #ax_trans.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax_trans.legend()
        
        plt.tight_layout()
        plt.show()

    @staticmethod
    def analyze_avg_distribution_kde_2(data, data2=None, data3=None, data4=None, 
                                    data5=None, data6=None,
                                    data_name="Data1", data2_name="Data2", 
                                    data3_name="Data3", data4_name="Data4",
                                    data5_name="Data5", data6_name="Data6"):
        """
        Given datasets that are (N,2) arrays where:
        - Column 0: average rotation error (in degrees)
        - Column 1: average translation error (in meters)
        
        Plots, on a 1x2 grid:
        - Left: For rotation errors with theoretical PDFs (based on the declared distribution type)
                overlaid with the corresponding KDE curves.
        - Right: For translation errors.
        
        Each dataset dictionary should include a "dist" key that is one of:
            "uniform", "gaussian", or "invgaussian"
        """
        # Define datasets with properties and distribution type.
        # For example, data1 is uniform, data2 is gaussian, data3 is inverse Gaussian,
        # and data4, data5, data6 are gaussian.
        datasets = [
            {"data": data,   "label": data_name,   "color": "#0065BD", "linestyle": "--", "dist": "uniform"},
            {"data": data2,  "label": data2_name,  "color": "#A2AD00", "linestyle": "--", "dist": "gaussian"},
            {"data": data3,  "label": data3_name,  "color": "#E37222", "linestyle": "--", "dist": "invgaussian"},
            {"data": data4,  "label": data4_name,  "color": "#0065BD", "linestyle": "-",  "dist": "gaussian"},
            {"data": data5,  "label": data5_name,  "color": "#A2AD00", "linestyle": "-",  "dist": "gaussian"},
            {"data": data6,  "label": data6_name,  "color": "#E37222", "linestyle": "-",  "dist": "gaussian"}
        ]
        
        # Define x-axis grid for rotation and translation.
        # Here we assume average rotation errors are in the range [0,20] and translation in [0,0.5].
        x_rot = np.linspace(0, 20, 100)
        x_trans = np.linspace(0, 0.5, 100)
        
        # Create a 1x2 grid: left for rotation, right for translation.
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Left subplot: Rotation errors.
        ax_rot = axes[0]
        for ds in datasets:
            if ds["data"] is not None:
                # Get average rotation errors (column 0).
                d = ds["data"][:, 0]
                
                # Compute theoretical PDF based on distribution type.
                if ds["dist"] == "uniform":
                    a, b = np.min(d), np.max(d)
                    pdf_theory = np.where((x_rot >= a) & (x_rot <= b), 1.0/(b - a), 0)
                elif ds["dist"] == "gaussian":
                    mu = np.mean(d)
                    sigma = np.std(d)
                    pdf_theory = norm.pdf(x_rot, loc=mu, scale=sigma)
                elif ds["dist"] == "invgaussian":
                    # Fit parameters for inverse Gaussian. We let invgauss.fit estimate shape, loc, scale.
                    # Note: Inverse Gaussian is defined for x>0.
                    params = invgauss.fit(d)
                    pdf_theory = invgauss.pdf(x_rot, *params)
                else:
                    pdf_theory = np.zeros_like(x_rot)
                
                # Fit KDE.
                kde = gaussian_kde(d)
                pdf_kde = kde(x_rot)
                
                # Plot theoretical PDF (dotted) and KDE (solid) for this dataset.
                ax_rot.plot(x_rot, pdf_theory, color=ds["color"], linestyle=ds["linestyle"], 
                            label=f"{ds['label']} (Theory)")
                ax_rot.plot(x_rot, pdf_kde, color=ds["color"], linestyle="-", 
                            label=f"{ds['label']} (KDE)")
        
        ax_rot.set_title("Average Rotation Error")
        ax_rot.set_xlabel("Rotation Error (degrees)")
        ax_rot.set_ylabel("Density")
        ax_rot.set_xlim(0, 20)
        ax_rot.set_ylim(bottom=0)
        ax_rot.legend(fontsize='small', loc='upper right')
        
        # Right subplot: Translation errors.
        ax_trans = axes[1]
        for ds in datasets:
            if ds["data"] is not None:
                # Get average translation errors (column 1).
                d = ds["data"][:, 1]
                
                # Compute theoretical PDF based on distribution type.
                if ds["dist"] == "uniform":
                    a, b = np.min(d), np.max(d)
                    pdf_theory = np.where((x_trans >= a) & (x_trans <= b), 1.0/(b - a), 0)
                elif ds["dist"] == "gaussian":
                    mu = np.mean(d)
                    sigma = np.std(d)
                    pdf_theory = norm.pdf(x_trans, loc=mu, scale=sigma)
                elif ds["dist"] == "invgaussian":
                    params = invgauss.fit(d)
                    pdf_theory = invgauss.pdf(x_trans, *params)
                else:
                    pdf_theory = np.zeros_like(x_trans)
                
                # Fit KDE.
                kde = gaussian_kde(d)
                pdf_kde = kde(x_trans)
                
                # Plot both curves.
                ax_trans.plot(x_trans, pdf_theory, color=ds["color"], linestyle=ds["linestyle"], 
                            label=f"{ds['label']} (Theory)")
                ax_trans.plot(x_trans, pdf_kde, color=ds["color"], linestyle="-", 
                            label=f"{ds['label']} (KDE)")
        
        ax_trans.set_title("Average Translation Error")
        ax_trans.set_xlabel("Translation Error (meters)")
        ax_trans.set_ylabel("Density")
        ax_trans.set_xlim(0, 0.5)
        ax_trans.set_ylim(bottom=0)
        ax_trans.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax_trans.legend(fontsize='small', loc='upper right')
        
        plt.tight_layout()
        plt.show()


    @staticmethod
    def plot_theoretical_curves(data, data2=None, data3=None, data4=None, 
                            data5=None, data6=None,
                            data_name="Data1", data2_name="Data2", 
                            data3_name="Data3", data4_name="Data4",
                            data5_name="Data5", data6_name="Data6"):
        """
        Plots the theoretical PDFs for average errors from datasets that are (N,2) arrays.
        - Column 0: average rotation error (degrees)
        - Column 1: average translation error (meters)
        
        Each dataset dictionary should include a "dist" key specifying:
        "uniform", "gaussian", or "invgaussian"
        
        Two subplots are produced: one for rotation and one for translation.
        """
        # Define the datasets and their properties including declared distribution.
        datasets = [
            {"data": data,   "label": data_name,   "color": "#0065BD", "linestyle": "--", "dist": "uniform"},
            {"data": data2,  "label": data2_name,  "color": "#A2AD00", "linestyle": "--", "dist": "gaussian"},
            {"data": data3,  "label": data3_name,  "color": "#E37222", "linestyle": "--", "dist": "invgaussian"},
            {"data": data4,  "label": data4_name,  "color": "#0065BD", "linestyle": "-",  "dist": "gaussian"},
            {"data": data5,  "label": data5_name,  "color": "#A2AD00", "linestyle": "-",  "dist": "gaussian"},
            {"data": data6,  "label": data6_name,  "color": "#E37222", "linestyle": "-",  "dist": "gaussian"}
        ]
        
        # Define x ranges:
        x_rot = np.linspace(0, 20, 100)       # For rotation errors (degrees)
        x_trans = np.linspace(0, 0.5, 100)      # For translation errors (meters)
        
        # Create a 1x2 grid of subplots.
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # --- Plot theoretical curves for rotation errors ---
        ax_rot = axes[0]
        for ds in datasets:
            if ds["data"] is not None:
                d = ds["data"][:, 0]  # Average rotation errors from column 0
                if ds["dist"] == "uniform":
                    a, b_val = np.min(d), np.max(d)
                    pdf = np.where((x_rot >= a) & (x_rot <= b_val), 1.0/(b_val - a), 0)
                elif ds["dist"] == "gaussian":
                    mu = np.mean(d)
                    sigma = np.std(d)
                    pdf = norm.pdf(x_rot, loc=mu, scale=sigma)
                elif ds["dist"] == "invgaussian":
                    params = invgauss.fit(d)
                    pdf = invgauss.pdf(x_rot, *params)
                else:
                    pdf = np.zeros_like(x_rot)
                ax_rot.plot(x_rot, pdf, color=ds["color"], linestyle=ds["linestyle"],
                            label=f"{ds['label']} (Theory)")
        ax_rot.set_title("Theoretical Rotation Error PDFs")
        ax_rot.set_xlabel("Rotation Error (degrees)")
        ax_rot.set_ylabel("Density")
        ax_rot.set_xlim(0, 20)
        ax_rot.set_ylim(bottom=0)
        ax_rot.legend(fontsize='small', loc='upper right')
        
        # --- Plot theoretical curves for translation errors ---
        ax_trans = axes[1]
        for ds in datasets:
            if ds["data"] is not None:
                d = ds["data"][:, 1]  # Average translation errors from column 1
                if ds["dist"] == "uniform":
                    a, b_val = np.min(d), np.max(d)
                    pdf = np.where((x_trans >= a) & (x_trans <= b_val), 1.0/(b_val - a), 0)
                elif ds["dist"] == "gaussian":
                    mu = np.mean(d)
                    sigma = np.std(d)
                    pdf = norm.pdf(x_trans, loc=mu, scale=sigma)
                elif ds["dist"] == "invgaussian":
                    params = invgauss.fit(d)
                    pdf = invgauss.pdf(x_trans, *params)
                else:
                    pdf = np.zeros_like(x_trans)
                ax_trans.plot(x_trans, pdf, color=ds["color"], linestyle=ds["linestyle"],
                            label=f"{ds['label']} (Theory)")
        ax_trans.set_title("Theoretical Translation Error PDFs")
        ax_trans.set_xlabel("Translation Error (meters)")
        ax_trans.set_ylabel("Density")
        ax_trans.set_xlim(0, 0.5)
        ax_trans.set_ylim(bottom=0)
        ax_trans.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax_trans.legend(fontsize='small', loc='upper right')
        
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_smoothed_curves(data, data2=None, data3=None, data4=None, 
                            data5=None, data6=None,
                            data_name="Data1", data2_name="Data2", 
                            data3_name="Data3", data4_name="Data4",
                            data5_name="Data5", data6_name="Data6"):
        """
        Plots the smooth (KDE) curves for average errors from datasets that are (N,2) arrays.
        This function uses gaussian_kde to compute a smoothed version of the empirical PDF.
        
        The layout is a 1x2 grid:
        - Left subplot: average rotation errors (degrees) with x-axis [0,20].
        - Right subplot: average translation errors (meters) with x-axis [0,0.5].
        """
        datasets = [
            {"data": data,   "label": data_name,   "color": "#0065BD", "linestyle": "--"},
            {"data": data2,  "label": data2_name,  "color": "#A2AD00", "linestyle": "--"},
            {"data": data3,  "label": data3_name,  "color": "#E37222", "linestyle": "--"},
            {"data": data4,  "label": data4_name,  "color": "#0065BD", "linestyle": "-"},
            {"data": data5,  "label": data5_name,  "color": "#A2AD00", "linestyle": "-"},
            {"data": data6,  "label": data6_name,  "color": "#E37222", "linestyle": "-"}
        ]
        
        # Define x ranges:
        x_rot = np.linspace(0, 20, 100)
        x_trans = np.linspace(0, 0.5, 100)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # --- Plot KDE (smoothed) curves for rotation errors ---
        ax_rot = axes[0]
        for ds in datasets:
            if ds["data"] is not None:
                d = ds["data"][:, 0]  # Average rotation errors
                kde = gaussian_kde(d,bw_method="silverman")
                pdf = kde(x_rot)
                ax_rot.plot(x_rot, pdf, color=ds["color"], linestyle=ds['linestyle'], 
                            label=f"{ds['label']} (KDE)")
        ax_rot.set_title("Smoothed Rotation Error KDEs")
        ax_rot.set_xlabel("Rotation Error (degrees)")
        ax_rot.set_ylabel("Density")
        ax_rot.set_xlim(0, 20)
        ax_rot.set_ylim(bottom=0)
        ax_rot.legend(fontsize='small', loc='upper right')
        
        # --- Plot KDE (smoothed) curves for translation errors ---
        ax_trans = axes[1]
        for ds in datasets:
            if ds["data"] is not None:
                d = ds["data"][:, 1]  # Average translation errors
                kde = gaussian_kde(d)
                pdf = kde(x_trans)
                ax_trans.plot(x_trans, pdf, color=ds["color"], linestyle=ds['linestyle'], 
                            label=f"{ds['label']} (KDE)")
        ax_trans.set_title("Smoothed Translation Error KDEs")
        ax_trans.set_xlabel("Translation Error (meters)")
        ax_trans.set_ylabel("Density")
        ax_trans.set_xlim(0, 0.5)
        ax_trans.set_ylim(bottom=0)
        ax_trans.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax_trans.legend(fontsize='small', loc='upper right')
        
        plt.tight_layout()
        plt.show()



    @staticmethod
    def plot_smoothed_curves2(data, data2=None, data3=None, data4=None, 
                            data5=None, data6=None,
                            data_name="Data1", data2_name="Data2", 
                            data3_name="Data3", data4_name="Data4",
                            data5_name="Data5", data6_name="Data6"):
        """
        Plots the smoothed (KDE) curves for average errors from datasets that are (N,2) arrays.
        This function uses gaussian_kde to compute a smoothed version of the empirical PDF.
        
        The layout is a 1x2 grid:
        - Left subplot: average rotation errors (degrees) with x-axis [0,20].
        - Right subplot: average translation errors (meters) with x-axis [0,0.5].
        
        For the first dataset, the data is first fit by a theoretical uniform PDF over the
        data’s range and then smoothed via KDE.
        For the third dataset, a theoretical inverse Gaussian sample is generated (using the data’s mean)
        and then smoothed via KDE.
        The other datasets are directly smoothed using KDE.
        """
        datasets = [
            {"data": data,   "label": data_name,   "color": "#0065BD", "linestyle": "--"},
            {"data": data2,  "label": data2_name,  "color": "#A2AD00", "linestyle": "--"},
            {"data": data3,  "label": data3_name,  "color": "#E37222", "linestyle": "--"},
            {"data": data4,  "label": data4_name,  "color": "#0065BD", "linestyle": "-"},
            {"data": data5,  "label": data5_name,  "color": "#A2AD00", "linestyle": "-"},
            {"data": data6,  "label": data6_name,  "color": "#E37222", "linestyle": "-"}
        ]
        
        # Define x ranges for rotation and translation.
        x_rot = np.linspace(0, 20, 100)
        x_trans = np.linspace(0, 0.5, 100)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # --- Plot KDE (smoothed) curves for rotation errors ---
        ax_rot = axes[0]
        for idx, ds in enumerate(datasets):
            if ds["data"] is not None:
                # d: average rotation errors from column 0.
                d = ds["data"][:, 0]
                # Depending on dataset index, process differently:
                if idx == 1:
                    # First dataset: force a theoretical uniform PDF.
                    low, high = np.min(d), np.max(d)
                    uniform_sample = np.random.uniform(low, high, size=len(d))
                    kde = gaussian_kde(uniform_sample, bw_method="silverman")
                # elif idx == 0:
                #     # Third dataset: force a theoretical inverse Gaussian.
                #     # Estimate the mean parameter (mu) from d (ensure positive values).
                #     mu_est = np.mean(d[d > 0]) if np.any(d > 0) else 1.0
                #     # Generate sample from the inverse Gaussian.
                #     # invgauss.rvs expects a shape parameter 'mu' and optional scale.
                #     invgauss_sample = invgauss.rvs(mu_est, scale=mu_est, size=len(d))
                #     kde = gaussian_kde(invgauss_sample)
                else:
                    # Other datasets: fit KDE directly.
                    kde = gaussian_kde(d)
                pdf = kde(x_rot)
                ax_rot.plot(x_rot, pdf, color=ds["color"], linestyle=ds["linestyle"], 
                            label=f"{ds['label']} (KDE)")
        ax_rot.set_title("Smoothed Rotation Error KDEs")
        ax_rot.set_xlabel("Rotation Error (degrees)")
        ax_rot.set_ylabel("Density")
        ax_rot.set_xlim(0, 20)
        ax_rot.set_ylim(bottom=0)
        ax_rot.legend(fontsize='small', loc='upper right')
        
        # --- Plot KDE (smoothed) curves for translation errors ---
        ax_trans = axes[1]
        for idx, ds in enumerate(datasets):
            if ds["data"] is not None:
                # d: average translation errors from column 1.
                d = ds["data"][:, 1]
                if idx == 1:
                    # First dataset: force a theoretical uniform PDF.
                    low, high = np.min(d), np.max(d)
                    uniform_sample = np.random.uniform(low, high, size=len(d))
                    kde = gaussian_kde(uniform_sample, bw_method="silverman")
                elif idx == 0:
                    # Third dataset: force a theoretical inverse Gaussian.
                    mu_est = np.mean(d[d > 0]) if np.any(d > 0) else 0.1
                    invgauss_sample = invgauss.rvs(mu_est, scale=mu_est, size=len(d))
                    kde = gaussian_kde(invgauss_sample)
                else:
                    kde = gaussian_kde(d)
                pdf = kde(x_trans)
                ax_trans.plot(x_trans, pdf, color=ds["color"], linestyle=ds["linestyle"], 
                            label=f"{ds['label']} (KDE)")
        ax_trans.set_title("Smoothed Translation Error KDEs")
        ax_trans.set_xlabel("Translation Error (meters)")
        ax_trans.set_ylabel("Density")
        ax_trans.set_xlim(0, 0.5)
        ax_trans.set_ylim(bottom=0)
        ax_trans.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax_trans.legend(fontsize='small', loc='upper right')
        
        plt.tight_layout()
        plt.show()

    

    @staticmethod
    def plot_smoothed_curves3(data, data2=None, data3=None, data4=None, 
                            data5=None, data6=None,
                            data_name="Data1", data2_name="Data2", 
                            data3_name="Data3", data4_name="Data4",
                            data5_name="Data5", data6_name="Data6"):
        """
        Plots smoothed (KDE) curves for average errors from datasets that are (N,2) arrays.
        This function uses seaborn.kdeplot uniformly for all inputs without forcing 
        a theoretical uniform or inverse Gaussian fit.
        
        The layout is a 1x2 grid:
        - Left subplot: average rotation errors (degrees) with x-axis [0,20].
        - Right subplot: average translation errors (meters) with x-axis [0,0.5].
        """
        datasets = [
            {"data": data,   "label": data_name,   "color": "#0065BD", "linestyle": "--"},
            {"data": data2,  "label": data2_name,  "color": "#A2AD00", "linestyle": "--"},
            {"data": data3,  "label": data3_name,  "color": "#E37222", "linestyle": "--"},
            {"data": data4,  "label": data4_name,  "color": "#0065BD", "linestyle": "-"},
            {"data": data5,  "label": data5_name,  "color": "#A2AD00", "linestyle": "-"},
            {"data": data6,  "label": data6_name,  "color": "#E37222", "linestyle": "-"}
        ]
        
        # Create a 1x2 subplot grid
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        
        # --- Plot smoothed KDE curves for rotation errors ---
        ax_rot = axes[0]
        for ds in datasets:
            if ds["data"] is not None:
                # Extract rotation errors from column 0
                d = ds["data"][:, 0]
                sns.kdeplot(d, bw_adjust=1, ax=ax_rot, color=ds["color"], linestyle=ds["linestyle"],
                            label=f"{ds['label']}")
        ax_rot.set_title("Rotation Error",fontsize=10)
        ax_rot.set_xlabel("Rotation Error [deg]",fontsize=10)
        ax_rot.set_ylabel("Density",fontsize=10)
        ax_rot.set_xlim(0, 20)
        ax_rot.set_ylim(bottom=0)
        ax_rot.legend(loc='upper right',fontsize=8)
        
        # --- Plot smoothed KDE curves for translation errors ---
        ax_trans = axes[1]
        for ds in datasets:
            if ds["data"] is not None:
                # Extract translation errors from column 1
                d = ds["data"][:, 1]
                sns.kdeplot(d, bw_adjust=1, ax=ax_trans, color=ds["color"], linestyle=ds["linestyle"],
                            label=f"{ds['label']}")
        ax_trans.set_title("Translation Error",fontsize=10)
        ax_trans.set_xlabel("Translation Error [m]",fontsize=10)
        ax_trans.set_ylabel("Density",fontsize=10)
        ax_trans.set_xlim(0, 0.5)
        ax_trans.set_ylim(bottom=0)
        ax_trans.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax_trans.legend(loc='upper right',fontsize=8)
        
        plt.tight_layout()
        plt.savefig("diff_distributions.pdf", dpi=300, bbox_inches='tight')
        plt.show()

    

    @staticmethod
    def plot_recall_curve(errors1, errors2=None, errors3=None, 
                          thresholds_rot=None, thresholds_trans=None,
                            label1=None, label2=None, label3=None):
        """
        Plots recall curves for two calibration error datasets on a 1x2 grid.
        
        Parameters:
        errors1: NumPy array of shape (N,2) for method 1
                Column 0: average rotation error (degrees)
                Column 1: average translation error (meters)
        errors2: NumPy array of shape (N,2) for method 2 (same layout)
        thresholds_rot: Array-like thresholds for rotation error (e.g. np.linspace(0,2,100))
        thresholds_trans: Array-like thresholds for translation error (e.g. np.linspace(0,0.1,100))
        label1: Label for method 1 (string)
        label2: Label for method 2 (string)
        
        The function computes recall (fraction of samples with error <= threshold)
        at each threshold and plots the recall curves in a 1x2 grid. It adds a vertical 
        dashed line at the threshold where recall reaches 90% for each method and also a 
        horizontal dashed line at recall = 0.9.
        
        Internally, method 1 is styled with color "#0065BD" and linestyle "--",
        and method 2 with color "#E37222" and linestyle "-".
        """
        # Internal styling for the two datasets.
        dataset_list = [
            {"data": errors1,   "label": label1,   "color": "#0065BD", "linestyle": "--"},
            {"data": errors2,  "label": label2,  "color": "#A2AD00", "linestyle": "--"},
            {"data": errors3,  "label": label3,  "color": "#E37222", "linestyle": "--"}
        ]
        
        fig, axes = plt.subplots(1, 2, figsize=(8,4))
        
        # Plot rotation recall curves on left subplot.
        for ds in dataset_list:
            if ds["data"] is not None:
                rot_errors = ds["data"][:, 0]
                recall_rot = np.array([np.mean(rot_errors <= t) for t in thresholds_rot])
                axes[0].plot(thresholds_rot, recall_rot, color=ds["color"], linestyle=ds["linestyle"],
                            linewidth=2,marker='D', markersize=5, label=ds["label"])
                idx = np.where(recall_rot >= 0.9)[0]
                if idx.size > 0:
                    thresh_90 = thresholds_rot[idx[0]]
                    axes[0].axvline(x=thresh_90, color=ds["color"], linestyle='--',
                                    linewidth=1, label=f"{ds['label']} 90% ({thresh_90:.2f}°)")
        
        # Add a horizontal line at 90% recall for rotation.
        axes[0].axhline(y=0.9, color='gray', linestyle=':', linewidth=1, label="90% Recall")
        
        axes[0].set_title("Recall Curve: Rotation Error",fontsize=10)
        axes[0].set_xlabel("Rotation Error Threshold [deg]",fontsize=10)
        axes[0].set_ylabel("Recall",fontsize=10)
        axes[0].set_xlim(thresholds_rot[0], thresholds_rot[-1])
        axes[0].set_ylim(0, 1.0)
        axes[0].legend(loc='lower right',fontsize=8)
        
        # Plot translation recall curves on right subplot.
        for ds in dataset_list:
            if ds["data"] is not None:
                trans_errors = ds["data"][:, 1]
                recall_trans = np.array([np.mean(trans_errors <= t) for t in thresholds_trans])
                axes[1].plot(thresholds_trans, recall_trans, color=ds["color"], linestyle=ds["linestyle"],
                            linewidth=2,marker='D', markersize=5, label=ds["label"])
                idx = np.where(recall_trans >= 0.9)[0]
                if idx.size > 0:
                    thresh_90 = thresholds_trans[idx[0]]
                    axes[1].axvline(x=thresh_90, color=ds["color"], linestyle='--',
                                    linewidth=1, label=f"{ds['label']} 90% ({thresh_90:.2f} m)")
            
        # Add a horizontal line at 90% recall for translation.
        axes[1].axhline(y=0.9, color='gray', linestyle=':', linewidth=1, label="90% Recall")
        
        axes[1].set_title("Recall Curve: Translation Error",fontsize=10)
        axes[1].set_xlabel("Translation Error Threshold [m]",fontsize=10)
        axes[1].set_ylabel("Recall",fontsize=10)
        axes[1].set_xlim(thresholds_trans[0], thresholds_trans[-1])
        axes[1].set_ylim(0, 1.0)
        axes[1].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        axes[1].legend(loc='lower right',fontsize=8)
        
        plt.tight_layout()
        plt.savefig("recall.pdf", dpi = 300, bbox_inches='tight')
        plt.show()


class VisualizeResults:

    @staticmethod
    def plot_perturbation(data1, data2=None):
    # Check if data has at least 6 columns
        if data1.shape[1] >= 6:
            # Define labels for each individual plot
            labels = ['Rotation X', 'Rotation Y', 'Rotation Z', 'Translation X', 'Translation Y', 'Translation Z']
            
            # Loop over each of the 6 dimensions and plot individually
            for i in range(6):
                plt.figure(figsize=(5, 4))
                plt.scatter(range(len(data1)), data1[:, i], s=1, color='#0065BD', label='Input Perturbation')
                if data2 is not None:
                    plt.scatter(range(len(data2)), data2[:, i], s=1, color='#E37222', alpha=0.5, label='Output Results')
                
                plt.xlabel('Data Samples')
                plt.ylabel(labels[i])
                #plt.title(labels[i])
                plt.legend(loc='upper right')
                
                # Set y-limits and ticks based on rotation vs. translation
                if i < 3:  # Rotation plots
                    plt.ylim(-20, 20)
                else:      # Translation plots
                    plt.ylim(-0.5, 0.5)
                    ticks = np.linspace(-0.5, 0.5, 9)
                    plt.yticks(ticks)
                
                plt.tight_layout()
                # Save the figure with a name corresponding to the label (spaces replaced with underscores)
                filename = "plot_perturbation_" + labels[i].replace(" ", "_") + ".pdf"
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                plt.show()
        else:
            print("The array must have at least 6 columns to plot individually.")

    @staticmethod
    def analyze_data_distribution_kde2(data, data2=None, data3=None, data4=None, 
                                    data_name="Data1", data2_name="Data2", 
                                    data3_name="Data3", data4_name="Data4"):
        """
        Plots KDE curves for 6 error components, saving each plot individually.
        Left side: rotation errors (roll, pitch, yaw) in degrees.
        Right side: translation errors (t_x, t_y, t_z) in meters.
        
        Each plot displays up to 4 KDE curves (one per dataset) with distinct color and linestyle.
        
        For the first dataset, a uniform sample over the range of the data is generated and used to fit the KDE.
        """
        from scipy.stats import gaussian_kde  # Ensure gaussian_kde is imported

        # Define labels and ticks.
        rotation_labels = ['Rotation X', 'Rotation Y', 'Rotation Z']
        translation_labels = ['Translation X', 'Translation Y', 'Translation Z']
        rotation_ticks = np.linspace(-20, 20, 9)
        translation_ticks = np.linspace(-0.5, 0.5, 9)
        
        # Prepare dataset properties.
        datasets = [
            {"data": data,   "label": data_name,  "color": "black",    "linestyle": "-"},
            {"data": data2,  "label": data2_name, "color": "#0065BD",  "linestyle": "--"},
            {"data": data3,  "label": data3_name, "color": "#A2AD00",  "linestyle": "-."},
            {"data": data4,  "label": data4_name, "color": "#E37222",  "linestyle": "-"}
        ]
        
        # Plot and save each rotation error individually.
        for row in range(3):
            fig, ax = plt.subplots(figsize=(5, 4))
            for idx, ds in enumerate(datasets):
                if ds["data"] is not None:
                    # Extract rotation error data: column index = row (0: roll, 1: pitch, 2: yaw)
                    d = ds["data"][:, row]
                    # For the first dataset, fit KDE to a uniform sample over the data range.
                    if idx == 0:
                        low, high = np.min(d), np.max(d)
                        uniform_sample = np.random.uniform(low, high, size=len(d))
                        density = gaussian_kde(uniform_sample)
                    else:
                        density = gaussian_kde(d)
                    x = np.linspace(np.min(d), np.max(d), 100)
                    y = density(x)
                    ax.plot(x, y, color=ds["color"], linestyle=ds["linestyle"], label=ds["label"])
            
            ax.set_xlabel('Decalibration [deg]', fontsize=10)
            ax.set_xticks(rotation_ticks)
            ax.tick_params(axis='x', labelsize=8)
            ax.set_ylabel('Density', fontsize=10)
            ax.set_xlim(-20, 20)
            ax.set_ylim(bottom=0)
            ax.legend(fontsize=9)
            
            plt.tight_layout()
            filename = "All_" + rotation_labels[row].replace(" ", "_") + ".svg"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.show()
            
        # Plot and save each translation error individually.
        for row in range(3):
            fig, ax = plt.subplots(figsize=(5, 4))
            for idx, ds in enumerate(datasets):
                if ds["data"] is not None:
                    # Extract translation error data: column index = row+3 (3: t_x, 4: t_y, 5: t_z)
                    d = ds["data"][:, row + 3]
                    if idx == 0:
                        low, high = np.min(d), np.max(d)
                        uniform_sample = np.random.uniform(low, high, size=len(d))
                        density = gaussian_kde(uniform_sample)
                    else:
                        density = gaussian_kde(d)
                    x = np.linspace(np.min(d), np.max(d), 100)
                    y = density(x)
                    ax.plot(x, y, color=ds["color"], linestyle=ds["linestyle"], label=ds["label"])
            
            ax.set_xlabel('Decalibration [m]', fontsize=10)
            ax.set_xticks(translation_ticks)
            ax.tick_params(axis='x', labelsize=8)
            ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
            ax.set_ylabel('Density', fontsize=10)
            ax.set_xlim(-0.5, 0.5)
            ax.set_ylim(0, 14)
            ax.legend(fontsize=9)
            
            plt.tight_layout()
            filename = "All_" + translation_labels[row].replace(" ", "_") + ".svg"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.show()
    

    @staticmethod
    def plot_smoothed_curves3(data, data2=None, data3=None, data4=None, 
                            data5=None, data6=None,
                            data_name="Data1", data2_name="Data2", 
                            data3_name="Data3", data4_name="Data4",
                            data5_name="Data5", data6_name="Data6"):
        """
        Plots smoothed (KDE) curves for average errors from datasets that are (N,2) arrays.
        For rotation errors (column 0) and translation errors (column 1),
        two separate figures are created and saved individually.
        Font sizes for labels, ticks, and legend remain as specified in the original code.
        """
        datasets = [
            {"data": data,   "label": data_name,   "color": "#0065BD", "linestyle": "--"},
            {"data": data2,  "label": data2_name,  "color": "#A2AD00", "linestyle": "--"},
            {"data": data3,  "label": data3_name,  "color": "#E37222", "linestyle": "--"},
            {"data": data4,  "label": data4_name,  "color": "#0065BD", "linestyle": "-"},
            {"data": data5,  "label": data5_name,  "color": "#A2AD00", "linestyle": "-"},
            {"data": data6,  "label": data6_name,  "color": "#E37222", "linestyle": "-"}
        ]
        
        # --- Plot and save rotation error curves ---
        fig, ax = plt.subplots(figsize=(5, 4))
        for ds in datasets:
            if ds["data"] is not None:
                # Extract rotation errors from column 0
                d = ds["data"][:, 0]
                sns.kdeplot(d, bw_adjust=1, ax=ax, color=ds["color"], linestyle=ds["linestyle"],
                            label=ds["label"])
        ax.set_xlabel("Rotation Error [deg]", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.set_xlim(0, 20)
        ax.set_ylim(bottom=0)
        ax.legend(loc='upper right', fontsize=8)
        plt.tight_layout()
        plt.savefig("Diff_dist_rotation_error.svg", dpi=300, bbox_inches='tight')
        plt.show()
        
        # --- Plot and save translation error curves ---
        fig, ax = plt.subplots(figsize=(5, 4))
        for ds in datasets:
            if ds["data"] is not None:
                # Extract translation errors from column 1
                d = ds["data"][:, 1]
                sns.kdeplot(d, bw_adjust=1, ax=ax, color=ds["color"], linestyle=ds["linestyle"],
                            label=ds["label"])
        ax.set_xlabel("Translation Error [m]", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.set_xlim(0, 0.5)
        ax.set_ylim(bottom=0)
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.legend(loc='upper right', fontsize=8)
        plt.tight_layout()
        plt.savefig("Diff_dist_translation_error.svg", dpi=300, bbox_inches='tight')
        plt.show()




    @staticmethod
    def plot_recall_curve(errors1, errors2=None, errors3=None, 
                        thresholds_rot=None, thresholds_trans=None,
                        label1=None, label2=None, label3=None):
        """
        Plots recall curves for calibration error datasets.
        For each error type (rotation and translation), a separate figure is created and saved.
        
        Parameters:
        errors1: NumPy array of shape (N,2) for method 1 (column 0: rotation [deg], column 1: translation [m])
        errors2: NumPy array of shape (N,2) for method 2 (same layout)
        errors3: NumPy array of shape (N,2) for method 3 (same layout)
        thresholds_rot: Array-like thresholds for rotation error
        thresholds_trans: Array-like thresholds for translation error
        label1, label2, label3: Labels for each method.
        
        Internal styling:
        - Method 1: color "#0065BD", linestyle "--"
        - Method 2: color "#A2AD00", linestyle "--"
        - Method 3: color "#E37222", linestyle "--"
        
        A vertical dashed line is added at the threshold where recall reaches 90%
        for each dataset, and a horizontal dashed line is added at recall = 0.9.
        """
        # Internal styling for the datasets.
        dataset_list = [
            {"data": errors1, "label": label1, "color": "#0065BD", "linestyle": "-"},
            {"data": errors2, "label": label2, "color": "#A2AD00", "linestyle": "-"},
            {"data": errors3, "label": label3, "color": "#E37222", "linestyle": "-"}
        ]
        
        # --- Plot rotation recall curves (for errors in column 0) ---
        fig, ax = plt.subplots(figsize=(5, 4))
        for ds in dataset_list:
            if ds["data"] is not None:
                rot_errors = ds["data"][:, 0]
                recall_rot = np.array([np.mean(rot_errors <= t) for t in thresholds_rot])
                ax.plot(thresholds_rot, recall_rot, color=ds["color"], linestyle=ds["linestyle"],
                        linewidth=2, label=ds["label"])
                idx = np.where(recall_rot >= 0.9)[0]
                if idx.size > 0:
                    thresh_90 = thresholds_rot[idx[0]]
                    ax.axvline(x=thresh_90, color=ds["color"], linestyle='--',
                            linewidth=1, label=f"{ds['label']} 90% ({thresh_90:.2f}°)")
        
        # Add horizontal dashed line at recall = 0.9
        ax.axhline(y=0.9, color='gray', linestyle=':', linewidth=1, label="90% Recall")
        
        ax.set_xlabel("Rotation Error Threshold [deg]", fontsize=10)
        ax.set_ylabel("Recall", fontsize=10)
        ax.set_xlim(thresholds_rot[0], thresholds_rot[-1])
        ax.set_ylim(0, 1.0)
        ax.legend(loc='lower right', fontsize=8)
        
        plt.tight_layout()
        plt.savefig("recall_rotation_A2.svg", dpi=300, bbox_inches='tight')
        plt.show()
        
        # --- Plot translation recall curves (for errors in column 1) ---
        fig, ax = plt.subplots(figsize=(5, 4))
        for ds in dataset_list:
            if ds["data"] is not None:
                trans_errors = ds["data"][:, 1]
                recall_trans = np.array([np.mean(trans_errors <= t) for t in thresholds_trans])
                ax.plot(thresholds_trans, recall_trans, color=ds["color"], linestyle=ds["linestyle"],
                        linewidth=2, label=ds["label"])
                idx = np.where(recall_trans >= 0.9)[0]
                if idx.size > 0:
                    thresh_90 = thresholds_trans[idx[0]]
                    ax.axvline(x=thresh_90, color=ds["color"], linestyle='--',
                            linewidth=1, label=f"{ds['label']} 90% ({thresh_90:.2f} m)")
        
        # Add horizontal dashed line at recall = 0.9
        ax.axhline(y=0.9, color='gray', linestyle=':', linewidth=1, label="90% Recall")
        
        ax.set_xlabel("Translation Error Threshold [m]", fontsize=10)
        ax.set_ylabel("Recall", fontsize=10)
        ax.set_xlim(thresholds_trans[0], thresholds_trans[-1])
        ax.set_ylim(0, 1.0)
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.legend(loc='lower right', fontsize=8)
        
        plt.tight_layout()
        plt.savefig("recall_translation_A2.svg", dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def analyze_data_distribution_boxplot(data, data2=None, data3=None, data4=None, 
                                        data_name="Data1", data2_name="Data2", 
                                        data3_name="Data3", data4_name="Data4"):
        """The function creates subplots for rotation and translation errors without outliers.
        The titles and axis labels have increased font sizes.
        """
        # Create a list of the models and their corresponding names.
        models = [data, data2, data3, data4]
        names = [data_name, data2_name, data3_name, data4_name]

        # Loop over each model dataset.
        for i, model in enumerate(models):
            if model is None:
                continue  # Skip if this dataset is not provided.
            
            # Create a figure with 1 row and 2 columns.
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
            
            # Rotation errors: columns 0, 1, 2.
            rotation_data = [model[:, 0], model[:, 1], model[:, 2]]
            axes[0].boxplot(rotation_data, showfliers=False)
            axes[0].set_xticklabels(['Rotation X', 'Rotation Y', 'Rotation Z'], fontsize=12)
            axes[0].set_ylabel('Error Value [deg]', fontsize=14)
            axes[0].set_title(f'{names[i]} - Rotation Errors', fontsize=17)
            axes[0].set_ylim(-20, 20)
            axes[0].tick_params(axis='both', labelsize=12)
            
            # Translation errors: columns 3, 4, 5.
            translation_data = [model[:, 3], model[:, 4], model[:, 5]]
            axes[1].boxplot(translation_data, showfliers=False)
            axes[1].set_xticklabels(['Translation X', 'Translation Y', 'Translation Z'], fontsize=12)
            axes[1].set_ylabel('Error Value [m]', fontsize=14)
            axes[1].set_title(f'{names[i]} - Translation Errors', fontsize=17)
            axes[1].set_ylim(-0.5, 0.5)
            axes[1].set_yticks(np.arange(-0.5, 0.51, 0.1))
            axes[1].tick_params(axis='both', labelsize=12)
            
            plt.tight_layout()
            
            # Save the figure as an SVG file with dpi=300 and a tight bounding box.
            filename = f"A2_boxplots_{names[i].replace(' ', '_')}.svg"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.show()
            plt.close(fig)
