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

void calcDepthOptimized(float *depth, float *left, float *right, int imageWidth, int imageHeight, int featureWidth, int featureHeight, int maximumDisplacement)
{
	memset(depth, 0, imageHeight*imageWidth*sizeof(float));

	int y = featureHeight;
	int x = featureWidth;

	float minimumSquaredDifference = -1;
	int minimumDy = 0;
	int minimumDx = 0;

	int dy = 0;
	int dx = 0;

	float sqrdDifference = 0;

	__m128 sqrdDifferenceSum = _mm_setzero_ps();
	int edges = 0;

	int boxX = - featureWidth;
	int boxY = -featureHeight;

	int leftX = 0;
	int leftY = 0;
	int rightX = 0;
	int rightY = 0;

	__m128 difference = _mm_setzero_ps();
	__m128 temp_sqrdDifference = _mm_setzero_ps();

	float values[4] = {0,0,0,0};
	float diff = 0.0;
	int flag = 0;
	int left_bound = 0;
	int right_bound = 0;
	
	for (; y < imageHeight - featureHeight; y++)
	{
		for (; x < imageWidth - featureWidth; x++)
		{
			for (dx = MAX(-maximumDisplacement, featureWidth - x); dx <= MIN(maximumDisplacement, imageWidth - featureWidth - x - 1); dx++)
			{
				for (dy = MAX(-maximumDisplacement, featureHeight - y); dy <= MIN(maximumDisplacement, imageHeight - featureHeight - y - 1); dy++)
				{
					sqrdDifference = 0;
					sqrdDifferenceSum = _mm_setzero_ps();
					boxX = -featureWidth;
					
					edges = featureWidth % 2 == 0 ? featureWidth: featureWidth - 2;
					flag = 0;

					for (; boxX < edges; boxX+=4)
					{
						for (; boxY <= featureHeight; boxY++)
						{
							leftY = y + boxY;
							rightY = y + dy + boxY;
							leftX = x + boxX;
							rightX = x + dx + boxX;
 
							difference = _mm_sub_ps(_mm_loadu_ps(left + (leftY * imageWidth + leftX)), _mm_loadu_ps(right + (rightY * imageWidth + rightX)));
							temp_sqrdDifference = _mm_mul_ps(difference, difference);
							sqrdDifferenceSum = _mm_add_ps(sqrdDifferenceSum, temp_sqrdDifference);

							if (flag == 0)
							{
								leftX = x + edges;
								rightX = x + dx + edges;

								

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
						}
						
						flag = 1;
						boxY = -featureHeight;
					}

					boxY = -featureHeight;
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

			minimumSquaredDifference = -1;
			minimumDy = 0;
			minimumDx = 0;
		}
		x = featureWidth;
	}
}
