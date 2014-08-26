/*此程序默认引力常量为“1”，每个body的质量为“1”*/
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <windows.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#define BLOCK_SIZE 256
#define SOFTENING 1e-9f

typedef struct { float4 *pos, *vel; } BodySystem;

__global__
void bodyForce(float4 *p, float4 *v, float dt, int n, int div) 
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < div) 
  {//计算“i”的受力
    float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;

    for (int tile = 0; tile < gridDim.x; tile++) 
    {
      __shared__ float3 spos[BLOCK_SIZE]; //共享内存
      float4 tpos = p[tile * blockDim.x + threadIdx.x];
      spos[threadIdx.x] = make_float3(tpos.x, tpos.y, tpos.z);
      __syncthreads(); //块内线程同步

      for (int j = 0; j < BLOCK_SIZE; j++) 
      {
        float dx = spos[j].x - p[i].x;
        float dy = spos[j].y - p[i].y;
        float dz = spos[j].z - p[i].z;
        float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
        float invDist = rsqrtf(distSqr);
        float invDist3 = invDist * invDist * invDist;

        Fx += dx * invDist3; Fy += dy * invDist3; Fz += dz * invDist3; // 计算x,y,z方向的受力
      }
      __syncthreads(); //块内线程同步
    }

    v[i].x += dt*Fx; v[i].y += dt*Fy; v[i].z += dt*Fz; // 计算x,y,z方向的速率
  }
}

extern "C" void cumain(int nBodies, float *h_buf, float dt, int div);
extern "C" void cumain(int nBodies, float *h_buf, float dt, int div)
{ 
	int bytes0 = 2*nBodies*sizeof(float4);
	int bytes1 = 2*div*sizeof(float4);

	float *d_buf;
	cudaMalloc(&d_buf, bytes0);
    BodySystem d_p = { (float4*)d_buf, ((float4*)d_buf) + nBodies };

    int nBlocks = (nBodies + BLOCK_SIZE - 1) / BLOCK_SIZE;
  
    cudaMemcpy(d_buf, h_buf, bytes0, cudaMemcpyHostToDevice);
    bodyForce<<<nBlocks, BLOCK_SIZE>>>(d_p.pos, d_p.vel, dt, nBodies, div);
    cudaMemcpy(h_buf, d_buf, bytes1, cudaMemcpyDeviceToHost);

	cudaFree(d_buf);
}
