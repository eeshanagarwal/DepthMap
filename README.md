# DepthMap
Generating depth maps from stereo images; Exploring performance optimization using Intel SSE/SIMD instructions and parallelization through OpenMP

In this project I wrote a program to generate depth maps from stereo images (8-bit grayscale bitmap images) using a simple block matching algorithm. 
In order to achieve depth perception, the program generates a depth map in the form of an image where each "pixel" is a value from 0 to 255 inclusive, representing how far away the object at that pixel is. In this case, 0 means infinity, while 255 means as close as can be.
By using images of a particular scene taken by two camera offsetted by a certain amount, the program can tell how far away an object is by comparing the position of the object is in say the "left image" with respect to right image". 
The algorithom divides the left image into small patches and for every patch it finds it's corresponding one in the right image.
After doing this it compares the two matches and checks for similarity by checking if they have a small Squared Euclidean Distance. 
Once we find the feature in the right image that's most similar, we check how far away from the original feature it is, and that tells us how close by or far away the object is.

To achieve optimization, two features were explored. Firstly, Intel SSE/SIMD instructions were used to give a signifcant speed up. Secondly, the code was parallized using OpenMP directives.
More information on both of these can be found here:
http://openmp.org/wp/
and https://software.intel.com/sites/products/documentation/doclib/iss/2013/compiler/cpp-lin/GUID-7478B278-2240-44D8-B396-1DC508E3656E.htm

Description of the files:
- calcDepthNaive.c contains unoptimzied code that implements the above described algorithom for generating depth maps 
- calcDepthOptimized.c contains optimized code for the above mentioned algorithom by implementing Intel SSE/SIMD instructions
- calcDepthOptimized_openMP.c  contains optimized code for the above mentioned algorithom by implementing Intel SSE/SIMD instructions and parallezing the program to run on multiple threads using the OpenMP API.
- bench-output.txt details the optimization in Gflop/s achieved by using Intel SSE/SIMD instructions for a set of feature and image sizes
- bench-output_openMP.txt details the optimization in Gflop/s achieved by using OpenMP for a set of feature and image sizes
