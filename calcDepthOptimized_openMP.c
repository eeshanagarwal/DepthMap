// CS 61C Fall 2014 Project 3

// include SSE intrinsics
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <x86intrin.h>
#endif
#include <memory.h>

#include "calcDepthOptimized.h"
#include "calcDepthNaive.h"
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))


__m128 calc_diff(__m128 sqrdsum, int leftX, int leftY, int rightX, int rightY, int imageWidth, float *left, float *right){
	#pragma omp parallel 
	{
		__m128 difference = _mm_sub_ps(_mm_loadu_ps(left + (leftY * imageWidth + leftX)), _mm_loadu_ps(right + (rightY * imageWidth + rightX)));
		__m128 temp_sqrdDifference = _mm_mul_ps(difference, difference);
		sqrdsum = _mm_add_ps(sqrdsum, temp_sqrdDifference);
	}
	return sqrdsum;
}

void calcDepthOptimized(float *depth, float *left, float *right, int imageWidth, int imageHeight, int featureWidth, int featureHeight, int maximumDisplacement)
{
	int threads, id, end;
	#pragma omp parallel 
	{
		id = omp_get_thread_num();
		threads = omp_get_num_threads();
		for(int ctr = 0; ctr<imageHeight*imageWidth/threads; ctr++)
		{
			depth[ctr*threads+id]= 0;
		}
	}
	end = imageHeight*imageWidth/threads*threads;
    for(int ctr = end; ctr<imageHeight*imageWidth; ctr++)
		depth[ctr]= 0;

	//memset(depth, 0, imageHeight*imageWidth*sizeof(float));

	#pragma omp parallel for 
	for (int y = featureHeight; y < imageHeight - featureHeight; y++)
	{
		for (int x = featureWidth; x < imageWidth - featureWidth; x++)
		{
			float minimumSquaredDifference = -1;
			int minimumDy = 0;
			int minimumDx = 0;
			
			for (int dy = MAX(-maximumDisplacement, featureHeight - y); dy <= MIN(maximumDisplacement, imageHeight - featureHeight - y - 1); dy++)
			{
				for (int dx = MAX(-maximumDisplacement, featureWidth - x); dx <= MIN(maximumDisplacement, imageWidth - featureWidth - x - 1); dx++)
				{

					float sqrdDifference = 0;
					__m128 sqrdDifferenceSum = _mm_setzero_ps();
					
					int edges = featureWidth % 2 == 0 ? featureWidth: featureWidth - 2;

					int leftY =0;
					int rightY=0;
					int leftX =0;
					int rightX=0;
					
					for (int boxY = -featureHeight; boxY <= featureHeight; boxY++)
					{
						for (int boxX = -featureWidth; boxX < edges; boxX+=4)
						{
							 leftY = y + boxY;
							 rightY = y + dy + boxY;
							 leftX = x + boxX;
							 rightX = x + dx + boxX;

							 sqrdDifferenceSum = calc_diff(sqrdDifferenceSum, leftX, leftY, rightX, rightY, imageWidth, left, right);
						}
						
						leftX = x + edges;
						rightX = x + dx + edges;
						leftY = y + boxY;
						rightY = y + dy + boxY;

						float diff = 0.0;
						int left_bound = 0;
						int right_bound = 0;

						if (featureWidth % 2 == 0)
						{
							diff = left[leftY * imageWidth + leftX] - right[rightY * imageWidth + rightX];
							sqrdDifference += diff * diff;
						}
						else
						{

							left_bound = leftY * imageWidth + leftX;
							right_bound = rightY * imageWidth + rightX;
							
						    diff = left[left_bound] - right[right_bound];
                            sqrdDifference += diff * diff;
                            
                            diff = left[left_bound + 1] - right[right_bound + 1];
                            sqrdDifference += diff * diff;

                            diff = left[left_bound + 2] - right[right_bound + 2]; 
                            sqrdDifference += diff * diff; 
						}
					}

					float values[4] = {0,0,0,0};
					_mm_storeu_ps(values, sqrdDifferenceSum);
					sqrdDifference += values[0] + values[1] + values[2] + values[3];

					if (((minimumSquaredDifference == sqrdDifference) && (displacementNaive(dx, dy) < displacementNaive(minimumDx, minimumDy))) || (minimumSquaredDifference == -1) || (minimumSquaredDifference > sqrdDifference))
					{
						minimumSquaredDifference = sqrdDifference;
						minimumDx = dx;
						minimumDy = dy;
					}
				}
			}

			if (minimumSquaredDifference != -1 && maximumDisplacement != 0)
			{ 
				depth[y * imageWidth + x] = displacementNaive(minimumDx, minimumDy);	
			}
		}
	}
}

