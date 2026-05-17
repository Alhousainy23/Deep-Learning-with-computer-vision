#Part 1 ==> Learnable Filter Coefficients
import numpy as np 
#Input Image about 5*5 
input_image = np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 0, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0],
], dtype=float)
print("=" * 50)
#==============================================================
#Types of different filters
#Edge Detector
edge_filter = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1],
], dtype=float)
#Blur / Smooth
blur_filter = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
], dtype=float) / 9.0   # normalize
#Custom (Laplacian)
custom_filter = np.array([
    [ 0,  1,  0],
    [ 1, -4,  1],
    [ 0,  1,  0],
], dtype=float)
filters = {
    "Edge Detector": edge_filter,
    "Blur/Smooth":   blur_filter,
    "Custom (Laplacian)": custom_filter,
}
print("\nFilter types and their coefficients:")
for name,f in filters.items():
    print(f"\n [{name}]")
    print(f)
#==============================================================
#Convolution Operation function
def manual_conv2d(image,kernel):
    image_h , image_w = image.shape
    kernel_h , kernel_w = kernel.shape
    output_h = image_h - kernel_h + 1
    output_w = image_w - kernel_w + 1
    output = np.zeros((output_h,output_w))
    for i in range(output_h):
        for j in range(output_w):
            region = image[i:i+kernel_h, j:j+kernel_w]
            output[i,j] = np.sum(region * kernel)
    return output
#==============================================================
#Apply each filter to the input image
print("\nApplying filters to the input image:")
for name , f in filters.items():
    fmap=manual_conv2d(input_image,f)
    print(f"\n  [{name} Feature Map]:")
    print(fmap)
#==============================================================
print("===" * 50)
#Update filter coefficients (simulate learning)
print("\nUpdating filter coefficients (simulating learning)...")
# Example: Slightly modify the edge detector filter
np.random.seed(42)  # for reproducibility
learnable_filter = np.random.rand(3, 3) * 2 - 1  # random values in range [-1, 1]
target_filter = edge_filter  # we want to learn something similar to the edge detector
lr = 0.1  # learning rate
for epoch in range(1,6):
    # Gradient descent step (simple update towards target filter)
    grad = learnable_filter - target_filter  # simple gradient (difference)
    learnable_filter -= lr * grad  # update filter coefficients
    loss =np.mean(grad ** 2)  # mean squared error as loss
    print(f" Epoch {epoch}: loss = {loss:.4f} -> filter approaching edge detector")
print(f"]n Final Learned Filter Coefficients:\n{learnable_filter}")