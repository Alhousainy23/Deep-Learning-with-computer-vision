#Part 3 ==> Deep Learning: Multi-Layer CNN
import numpy as np
#==============================================
def relu_function(x): return np.maximum(0, x)
#==============================================
def conv2d(image,kernel,padding=0):#convolutional 2D with optional padding
    if padding: image = np.pad(image,padding,mode='constant')
    H,W= image.shape; kH,kW=kernel.shape #output size
    output= np.zeros((H-kH+1,W-kW+1)) #convolution operation
    for r in range(output.shape[0]):# Shape 0 because search for rows
        for c in range(output.shape[1]):# Shape 1 because search for columns
            output[r,c]= np.sum(image[r:r+kH,c:c+kW]*kernel)
    return output
#==============================================
def max_pool2d(feature_map,size=2):#Size = 2 means reduce to half
    H,W = feature_map.shape; output = np.zeros((H//size,W//size))#Output size
    for r in range(output.shape[0]):
        for c in range(output.shape[1]):
            output[r,c]= np.max(feature_map[r*size:(r+1)*size,c*size:(c+1)*size])
    return output
#==============================================
#Input image 8x8
np.random.seed(0)#For reproducibility
image = np.round(np.random.rand(8,8)*2).astype(float)
#Layer 1: Edge Filter
edge_kernel = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]],dtype=float)
layer1_conv = conv2d(image,edge_kernel)
layer1_relu = relu_function(layer1_conv)
layer1_pool = max_pool2d(layer1_relu,size=2)
#Layer 2: Horizontal Pattern
horiz_kernel = np.array([[-1,-1,-1],[2,2,2],[-1,-1,-1]],dtype=float)
layer2_conv = conv2d(layer1_pool,horiz_kernel)
layer2_relu = relu_function(layer2_conv)
print("="*60)
print("A) Manual CNN (NumPy only)")
print("Layer 1 — Edge Detection:")
print("Convolution Output:\n", layer1_conv)
print("ReLU Output:\n", layer1_relu)
print("Max Pooling Output:\n", layer1_pool)
print("\nLayer 2 — Horizontal Pattern Detection:")
print("Convolution Output:\n", layer2_conv)
print("ReLU Output:\n", layer2_relu)
#==============================================
#CNN by using PyTorch (Recommended for learning)
print("(B) CNN with PyTorch (Recommended for learning)")
try: #Using try-except to handle cases where PyTorch might not be installed
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    # Define a simple CNN model
    class SimpleCNN(nn.Module):
        def __init__(self,num_classes=10): #Assuming 10 classes for classification
            super(SimpleCNN,self).__init__()
            #Layer 1 : Edge Detection
            self.conv1 = nn.Sequential(nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,padding=1),#same padding
                                                 nn.Relu(),nn.MaxPool2d(kernel_size=2,stride=2))#Reduce size by half
            #Layer 2 : Corners & Curves
            self.conv2 = nn.Sequential(nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1),
                                                 nn.Relu(),nn.MaxPool2d(kernel_size=2,stride=2))
            #Layer 3 : Complex patterns & textures
            self.conv3 = nn.Sequential(nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1),
                                                 nn.Relu(),nn.MaxPool2d(kernel_size=2,stride=2))
            #Fully Connected Layer for classification
            self.classifier= nn.Sequential(nn.ApabtiveAvgPool2d((1,1)),#Global average pooling
                                                    nn.Flatten(),#Flatten to 1D
                                                    nn.Linear(128,num_classes))#Output layer
        def forward(self,x):
            x = self.conv1(x) #Layer 1
            x = self.conv2(x) #Layer 2
            x = self.conv3(x) #Layer 3
            x = self.classifier(x) #Classification layer
            return x
    # Instantiate the model and print summary
    model = SimpleCNN(num_classes=10)
    print(model)
    dumy_input = torch.randn(1,1,32,32) #Batch size 1, 1 channel, 32x32 image
    with torch.np_grad():
        l1 = model.conv1(dumy_input)
        l2 = model.conv2(l1)
        l3 = model.conv3(l2)
        print("\nLayer 1 output shape:", l1.shape)
        print("Layer 2 output shape:", l2.shape)
        print("Layer 3 output shape:", l3.shape)
        print("\nSummary — What each layer learns:")
        print("=" * 60)
        #=====================================================
        # #Total Learnable Parameters Calculation
        total = sum(p.numel() for p in model.parameters() if p.requires_grad) #numel() counts total elements in the tensor
        print(f"Total learnable parameters: {total}")
        print("Each number learning by using Propagation and Backpropagation algorithms")
        #=============================================
        #Extraxt filter weights from Layer 1
        filters = model.conv1[0].weight.data.cpu().numpy() #Extracting weights of the first convolutional layer
        print("\nLayer 1 filter weights shape:", filters.shape)
        print("Each filter shape:", filters[0].shape) #Shape of each filter (in_channels, kH, kW)
        print("\nFirst filter coefficients (Learnable!):")
        print(np.round(filters[0,0],3)) #Print the first filter's coefficients rounded
except ImportError:print("PyTorch is not installed. Please install it to run the CNN part.")
#==============================================
#using TensorFlow/Keras (Optional, for comparison)
print("\n(C) CNN with TensorFlow/Keras (Optional, for comparison)")
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models,keras
    model_keras = models.Sequential([
        #Layer 1: Edge Detection
        keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu', input_shape=(32, 32, 1)),
        keras.layers.MaxPooling2D(pool_size=2, strides=2),
        #Layer 2: Corners & Curves
        keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'),
        keras.layers.MaxPooling2D(pool_size=2, strides=2),
        #Layer 3: Complex patterns & textures
        keras.layers.Conv2D(128, kernel_size=3, padding='same', activation='relu'),
        keras.layers.MaxPooling2D(pool_size=2, strides=2),
        #Classification layer
        keras.layers.GlobalAveragePooling2D(),keras.Layers.Dense(10, activation='softmax')#Assuming 10 classes
    ],name="Simple_CNN_Keras")
    model_keras.summary()
    #Filter weights extraction from Layer 1
    filters_keras = model_keras.layers[0].get_weights()[0] #Extracting weights of the first convolutional layer
    print("\nLayer 1 filter weights shape:", filters_keras.shape)
    print("Each filter shape:", filters_keras[:, :, 0, 0].shape) #Shape of each filter (kH, kW)
    print("\nFirst filter coefficients (Learnable!):")  
    print(np.round(filters_keras[:, :, 0, 0], 3)) #Print the first filter's coefficients rounded
except ImportError:print("TensorFlow is not installed. Please install it to run the Keras CNN part.")
#==============================================
print("\n" + "=" * 60)
print("Summary — What each layer learns:")
print("=" * 60)
summary = [
    ("Layer 1", "Edges & gradients",       "Horizontal/vertical edges"),
    ("Layer 2", "Corners & curves",        "Edge combinations, corners, curves"),
    ("Layer 3", "Textures & parts",        "Merged patterns, textures, object parts"),
    ("Layer 4+","Complex patterns/objects","Shapes, faces, objects — high-level features"),
]
print(f"  {'Layer':<10} {'What it detects':<30} {'English Explanation'}")
print("  " + "-" * 65)
for row in summary:
    print(f"  {row[0]:<10} {row[1]:<30} {row[2]}")

print("""
Key equations:
  Feature Map pixel = Σ (input_patch * filter_weights) + bias
  After ReLU        = max(0, Feature Map pixel)
  Backprop update   = weight -= lr * ∂Loss/∂weight
""")