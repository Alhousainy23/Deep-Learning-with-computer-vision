#Convolutional Terminologies & Size Calculations 
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  
import torch
import torch.nn as nn
#Create Function beacuse Show Tensors 
def show(name,tensor):
    s = tensor.shape 
    print(f"{name:30s} -> {tuple(s)}"
          f" [N={s[0]},C={s[1]},H={s[2]},W={s[3]}]")
#=================================================================
#Input related to features map 
print("="*60)
print("1. Input Tensors")
print("="*60)
#Inputs shape is ==> (Batch , Channels , Height , Width)
batch_szie = 1
in_channels = 3 #RGB
H,W = 8,8 
x= torch.randn(batch_szie,in_channels,H,W)
show("Input (Batch size , Channels , Height ,Width)" ,x)
print(f"Batch = {batch_szie},Channels = {in_channels} , Height = {H} , Width = {W}")
print("="*60)
#======================================================================================
#Craete Kernel | Filter dimensions
print("2. Kernel [Filter] Paramteres")
print("="*60)
Kernel_Height,Kernel_width= 3 ,3 #Kernel Size 
out_channels = 16 # Each filter extract number of features map 
padding = 0 #Default Padding to image 
stride = 1 # Number of step = 1 
#Create Convolutional Operation 
conv = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,
                 kernel_size=(Kernel_Height,Kernel_width),padding=padding,stride=stride)
#Weights is the Learnable Filter Cofficients 
print(f"Filter shape(weights): {tuple(conv.weight.shape)}")
print(f"= (Output Channels = {out_channels},input_channels = {in_channels},KH= {Kernel_Height},KW = {Kernel_width})")
print(f"Learnable Params in this layer : "
      f"{sum(p.numel() for p in conv.parameters()):,}\n")
#=============================================================================================================
#Output Size Formula 
print("="*60)
print("3. Output Size Formula") 
print("="*60)
#Equation = floor((H +2P -KH)/S)+1
import math 
def calc_output_size(H,KH,P,S):return math.floor((H +2 * P -KH) /S)+1
'''Applied This Equation '''
out_height = calc_output_size(H,Kernel_Height,padding,stride)
out_width  = calc_output_size(W,Kernel_width,padding,stride)
print(f"Formula : out = floor ((H +2P -KH) /S) +1")
print(f"Height = {H},KH = {Kernel_Height},Padding = {padding},Stride={stride}")
print(f"Out_Height = floor (({H} + 2x {padding}- {Kernel_Height}) / {stride}) +1 = {out_height}")
print(f"Out_W = floor(({W} + 2x{padding} - {Kernel_width}) / {stride}) +1 = {out_width}\n")
#Vertication for Pytorch Work 
with torch.no_grad(): y = conv(x)
show("Output Features Map ",y)
print(f" = {out_channels} Feature Maps , each {out_height} * {out_width}\n")
#================================================================================================================
#4. Padding 
print("="*60)
print("4. Padding Types")
print("="*60)
#Vaild Padding (p=0) #Size is Reducing in this Type of padding 
conv_valid = nn.Conv2d(in_channels,8,kernel_size=3,padding=0,stride=1)
out_valid = conv_valid(x)
print(f" Valid(P=0) : Input {H} * {W} -> Output is = {out_valid.shape[2]} * {out_valid.shape[3]}")
print(f"This Filter is Reducing by (KH-1) ={Kernel_Height-1} From each direction")
#**********************************
#Same Padding : This padding is the same size 
padding_same = (Kernel_Height -1)//2 #From this 3*3 
conv_same = nn.Conv2d(in_channels,8,kernel_size=3,padding=padding_same,stride=1)
out_same = conv_same(x)
print(f"Same (Padding = {padding_same}): Input {H} * {W} -> {out_same.shape[2]}*{out_same.shape[3]}")
print("The size remained the same.")
#************************************
#Style Transfer is used to Reflect Padding 
reflect_pad=nn.ReflectionPad2d(1)
x_padded = reflect_pad(x)
print(f"Reflect padding: Input {H}*{W} -> Padded {x_padded.shape[2]}*{x_padded.shape[3]}")
print(f"Is Useful for Image Generation\n")
#===============================================================================================================
#Stride effect on the size 
print("="*60)
print("5. Stride Effect")
print("="*60)
for s in [1,2]:
    p = 1 # Same Is padding 
    conv_stride = nn.Conv2d(in_channels,8,kernel_size=3,padding=p,stride=s)
    out_stride=conv_stride(x)
    print(f" Stride is = {s}: {H}*{W} ->{out_stride.shape[2]} * {out_stride.shape[3]}"
          f"{'(Downsampling)' if s > 1 else '(Safe size)'}")
print()
#============================================================================================================
#6. Step 6 related Pooling because reduce the size after convolutional operation 
print("="*60)
print("6. Polling Layers")
print("="*60)
#First Take Output from Convolutional Operation 
conv_pretrained=nn.Conv2d(in_channels,16,kernel_size=3,padding=1)
feature_map= conv_pretrained(x)
print(f"Before Polling: {tuple(feature_map.shape)}")

#Max Pooling 
max_pool = nn.MaxPool2d(kernel_size=2,stride=2)
out_max_pool = max_pool(feature_map)
print(f"After MaxPolling2D(2*2) This result is : {tuple(out_max_pool.shape)} This is half size")

#Average Pooling 
avg_pool = nn.AvgPool2d(kernel_size=2,stride=2)
out_avg = avg_pool(feature_map)
print(f" After AvgPool2d(2*2) This result is : {tuple(out_avg.shape)}")

#Global Average Pooling 
gap = nn.AdaptiveAvgPool2d(1)
out_gap = gap(feature_map)
print(f" After GlobalAvgPool: {tuple(out_gap.shape)}")
#===========================================================================================================
#7. Activation Function by used Relu Function 
print("="*60)
print("7. Activation Function by using RELU")
print("="*60)
relu=nn.ReLU()
out_relu= relu(out_max_pool)
neg_before= (out_max_pool<0).sum().item()
neg_after = (out_relu<0).sum().item()
print(f"Before Using Relu: {neg_before} Negative Value")
print(f"After Using Relu : {neg_after} Negative Value But all values is 0 because used Relu")
#================================================================================================================
#Complete CNN Block 
print("="*60)
print("8. Complete CONV Block --> Conv +Batch Normalization + Relu + Polling")
print("="*60)
class ConvBlock(nn.Module):
    def __init__(self,input_channel,output_channel,kernel=3,p=1,stride=1,pool=True):
        super().__init__()
        self.conv = nn.Conv2d(input_channel,output_channel,kernel,padding=p,stride=stride,bias=False)
        self.batch_normalization=nn.BatchNorm2d(output_channel)
        self.relu = nn.ReLU()
        self.pool= nn.MaxPool2d(2,2) if pool else nn.Identity()
    def forward(self,x):return self.pool(self.relu(self.batch_normalization(self.conv(x))))
#Applied 3 Sequence Layers 
x_in = torch.randn(1,3,32,32)
b1= ConvBlock(3,32)
b2= ConvBlock(32,64)
b3= ConvBlock(64,128,pool=False)
print(f"Input : {tuple(x_in.shape)}")
with torch.no_grad():
    o1= b1(x_in)
    o2= b2(o1)
    o3= b3(o2)
print(f"Block1: {tuple(o1.shape)}")
print(f"Block2 {tuple(o2.shape)}")
print(f"Block3: {tuple(o3.shape)}")
#=================================================================================================================
# 9. SUMMARY — ملخص المعادلات
# ══════════════════════════════════════════
print("\n" + "=" * 60)
print("SUMMARY — Size Calculation Cheatsheet")
print("=" * 60)

examples = [
    # (H,  W,   kH, kW, P, S, label)
    (224, 224,  3,  3,  0, 1, "VGG-style, no pad"),
    (224, 224,  3,  3,  1, 1, "Same padding"),
    (224, 224,  3,  3,  1, 2, "Stride=2, downsampling"),
    (28,  28,   5,  5,  0, 1, "MNIST, 5×5 kernel"),
    (14,  14,   3,  3,  1, 1, "After MaxPool"),
]

print(f"\n  {'Input':12} {'Kernel':8} {'P':3} {'S':3} → {'Output':12} {'Label'}")
print("  " + "─" * 65)
for h, w, kh, kw, p, s, lbl in examples:
    oh = calc_output_size(h, kh, p, s)
    ow = calc_output_size(w, kw, p, s)
    print(f"  {h}×{w}{'':6} {kh}×{kw}{'':4} {p:<3} {s:<3} → {oh}×{ow}{'':6} {lbl}")