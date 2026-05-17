#Convolutional Features & Response Map 
import numpy as np
#Inputs & Filter
input_image = np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 0, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0],
], dtype=float)
edge_filter = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1],
], dtype=float)
H, W   = input_image.shape
kH, kW = edge_filter.shape
out_H  = H - kH + 1   # = 3
out_W  = W - kW + 1   # = 3
#====================================================
#Convolution Step by Step with Details
#=====================================================
print("=" * 60)
print("Convolution — Step by Step")
print(f"Input Shape : {input_image.shape}")
print(f"Filter Shape: {edge_filter.shape}")
print(f"Output Shape: ({out_H}, {out_W}) [valid padding]")
print("=" * 60)
#=====================================================================
feature_map = np.zeros((out_H, out_W)) #Initialize feature map with zeros
epoch = 1
for r in range (out_H): # R is the row index for the output feature map
    for c in range (out_W): # C is the column index for the output feature map 
        window = input_image[r:r+kH,c:c+kW]#Extract the 3x3 patch from the input image corresponding to the current position of the filter
        prod = window * edge_filter #Element-wise multiplication between the input patch and the filter
        val = np.sum(prod)#Sum all the values in the resulting matrix to get a single scalar value that represents the response of the filter at that position
        feature_map[r,c] = val #Assign the computed value to the corresponding position in the output feature map
        print(f"\nStep {epoch:02d} → Window Position: row={r}, col={c}") #Print the current step number and the position of the window in the input image
        print("Input Patch:") #Print the extracted input patch
        print(window)
        print("Element-wise Multiply with Filter:") #Print the result of the element-wise multiplication
        print(prod)
        print(f"Sum = {val} → Feature Map[{r},{c}] = {val}") #Print the computed value and its position in the feature map
        epoch += 1 #Increment the step counter
print("\nFinal Feature Map (Response Map):") #After processing all positions, print the final feature map
print(feature_map)
#=========================================================
# Create Relu activation function
print("\n" + "=" * 60)
print("\nApplying ReLU Activation Function on the Feature Map:") #Apply ReLU activation function to the feature map
# Example of Feature map 4*4 
feature_map_example = np.array([
    [-2, 0, 3, -1],
    [4, -5, 1, 2],
    [-3, 6, -4, 0],
    [1, -1, 2, -2]
], dtype=float)
print("Original Feature Map (with negative values):")
print(feature_map_example)
pooled_map = np.zeros((2,2))#Initialize the pooled map with zeros
for pr in range(2): # Row index for the pooled map [Pooled is the meaning of the output after max pooling]
    for pc in range(2): # Column index for the pooled map
        region = feature_map_example[pr*2:(pr+1)*2,pc*2:(pc+1)*2] #Extract the 2x2 region from the feature map corresponding to the current position of the pooling window
        pooled_map[pr,pc] = np.max(region) #Apply max pooling by taking the maximum value from the extracted region and assign it to the corresponding position in the pooled map
        print(f"\n Pooling Region for pooled map[{pr},{pc}]: {region.flatten()} -> max = {np.max(region)}") #Print the current pooling region and the maximum value selected for the pooled map 
print("\n After Max Pooling (2x2):") #After processing all pooling regions, print the resulting pooled map
print(pooled_map)
#=========================================================
#Padding because the output feature map is smaller than the input image
print("\n" + "=" * 60)
print("\nApplying Padding to the Input Image:") #Apply padding to the input image to maintain the spatial dimensions after convolution
padding_size =np.pad(input_image, pad_width=1, mode='constant', constant_values=0) #Add a border of zeros around the input image
print("Input Image shape: ", input_image.shape) #Print the shape of the original input image
print("Padded Input Image shape: ", padding_size.shape) #Print the shape of the padded input image
print("Padded Input Image:") #Print the padded input image
print(padding_size)
feature_map_padded = np.zeros_like(input_image) #Initialize the feature map for the padded input image with the same shape as the original input image
for r in range(input_image.shape[0]): # Row index for the output feature map
    for c in range(input_image.shape[1]): # Column index for the output feature map
        window = padding_size[r:r+kH, c:c+kW] #Extract the 3x3 patch from the padded input image corresponding to the current position of the filter
        prod = window * edge_filter #Element-wise multiplication between the extracted patch and the filter
        val = np.sum(prod) #Sum all the values in the resulting matrix to get a single scalar value that represents the response of the filter at that position
        feature_map_padded[r,c] = val #Assign the computed value to the corresponding position in the output feature map for the padded input image
print("\nFeature Map after Convolution with Padding:") #After processing all positions, print the resulting feature map for the padded input image
print(feature_map_padded)