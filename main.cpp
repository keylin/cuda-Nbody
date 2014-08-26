#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <windows.h>
#include <time.h>
#include "omp.h"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#define SOFTENING 1e-9f

typedef struct { float4 *pos, *vel; } BodySystem;

void randomizeBodies(float *data, int n) 
{
  for (int i = 0; i < n; i++) 
  {
    data[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
  }
}

extern "C" void cumain(int nBodies, float *h_buf, float dt, int n);

void get_bodyForce(float4 *p, float4 *v, float dt, int n, int div, float *h_buf) 
{
	omp_set_nested(1);//允许嵌套并行
    #pragma omp parallel num_threads(2)
	{
		 int tid=omp_get_thread_num();
         int tnum=omp_get_num_threads();
	     //printf("stid=%d,tnum=%d\n",tid,tnum);

		   if(tid == 0)
		   {
			   clock_t s,f;
			   s = clock();
			   //printf("n=%d,dt=%lf,h_buf=%p,div=%d\n",n,dt,h_buf,div);
			   cumain(n, h_buf, dt, div);
			   f = clock();
			   double duration = (double)(f - s)/CLOCKS_PER_SEC;
			   //printf("gPU time=%f seconds\n",duration);
			   //printf("Gtid=%d,tnum=%d\n",tid,tnum);
		   }
		   else
		   {
			   clock_t s,f;
			   s = clock();
			   #pragma omp parallel for schedule(dynamic)
			   for (int i = div; i < n; i++) 
			   {
				   int mtid=omp_get_thread_num();
                   int mtnum=omp_get_num_threads();

				   float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;
				   for (int j = 0; j < n; j++) 
				   {
						float dx = p[j].x - p[i].x;
						float dy = p[j].y - p[i].y;
						float dz = p[j].z - p[i].z;
						float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
						float invDist = 1.0f / sqrtf(distSqr);
						float invDist3 = invDist * invDist * invDist;
						Fx += dx * invDist3; Fy += dy * invDist3; Fz += dz * invDist3;
					}
					v[i].x += dt*Fx; v[i].y += dt*Fy; v[i].z += dt*Fz; // 计算x,y,z方向的速率

				    //printf("Ctid=%d,tnum=%d\n",mtid,mtnum);
				}
			    f = clock();
			    double duration = (double)(f - s)/CLOCKS_PER_SEC;
				//printf("cPU time=%f seconds\n",duration);
		   }
	}
}

void bodyForce(float4 *p, float4 *v, float dt, int n, int div, float *h_buf) 
{
	#pragma omp parallel for schedule(dynamic)
	for (int i = div; i < n; i++) 
	{
			float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;
			for (int j = 0; j < n; j++) 
			{
				float dx = p[j].x - p[i].x;
				float dy = p[j].y - p[i].y;
				float dz = p[j].z - p[i].z;
				float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
				float invDist = 1.0f / sqrtf(distSqr);
				float invDist3 = invDist * invDist * invDist;
				Fx += dx * invDist3; Fy += dy * invDist3; Fz += dz * invDist3;
			}
			v[i].x += dt*Fx; v[i].y += dt*Fy; v[i].z += dt*Fz; // 计算x,y,z方向的速率
	}
}

void mymain(int BodyDiv) 
{
  
  int nBodies = 20000;
  //if (argc > 1) nBodies = atoi(argv[1]);
  
  clock_t s,f;
  double duration,CGTime,GTime,CTime;
  
  const float dt = 0.01f; // 步长
  const int nIters = 5;  // 迭代次数

  int bytes = 2*nBodies*sizeof(float4);
  float *h_buf = (float*)malloc(bytes);
  BodySystem h_p = { (float4*)h_buf, ((float4*)h_buf) + nBodies };
  randomizeBodies(h_buf, 8*nBodies); // 初始化body信息
  //printf("h_buf=%p\n",h_buf);

  GTime = 0;
  for (int iter = 1; iter <= nIters; iter++) 
  {
	  s = clock();
	  
	  //printf("n=%d,dt=%lf,h_buf=%p,div=%d\n",nBodies,dt,h_buf,BodyDiv);
	  cumain(nBodies, h_buf, dt, nBodies);

	  for (int i = 0 ; i < nBodies; i++) 
      { // 更新各body位置坐标
		  h_p.pos[i].x += h_p.vel[i].x*dt;
          h_p.pos[i].y += h_p.vel[i].y*dt;
          h_p.pos[i].z += h_p.vel[i].z*dt;
       }
	  f = clock();
	  duration = (double)(f - s)/CLOCKS_PER_SEC;
	  if( iter > 1)
		  GTime += duration;
  }
  printf("\n\ngPU time: %.3f seconds\n", GTime);

  CTime = 0;
  for (int iter = 1; iter <= nIters; iter++) 
  {
	  s = clock();
	  
	  //printf("n=%d,dt=%lf,h_buf=%p,div=%d\n",nBodies,dt,h_buf,BodyDiv);
	  bodyForce(h_p.pos, h_p.vel, dt, nBodies, 0 ,h_buf);

	  for (int i = 0 ; i < nBodies; i++) 
      { // 更新各body位置坐标
		  h_p.pos[i].x += h_p.vel[i].x*dt;
          h_p.pos[i].y += h_p.vel[i].y*dt;
          h_p.pos[i].z += h_p.vel[i].z*dt;
       }
	  f = clock();
	  duration = (double)(f - s)/CLOCKS_PER_SEC;
	  if( iter > 1)
		  CTime += duration;
  }
  printf("cPU time: %.3f seconds\n", CTime);

  double TRate = CTime/(GTime+CTime);
  CGTime = 0;
  //BodyDiv = 11000;
  double ARate = (double)BodyDiv/(double)nBodies;
  printf("Theory Rate:%lf\nCurrent Rate:%lf\nBodyDiv:%d\n",TRate,ARate,BodyDiv);
  for (int iter = 1; iter <= nIters; iter++) 
  {
	  s = clock();
	  
	  //printf("n=%d,dt=%lf,h_buf=%p,div=%d\n",nBodies,dt,h_buf,BodyDiv);
	  get_bodyForce(h_p.pos, h_p.vel, dt, nBodies, BodyDiv ,h_buf);

	  for (int i = 0 ; i < nBodies; i++) 
      { // 更新各body位置坐标
		  h_p.pos[i].x += h_p.vel[i].x*dt;
          h_p.pos[i].y += h_p.vel[i].y*dt;
          h_p.pos[i].z += h_p.vel[i].z*dt;
       }
	  f = clock();
	  duration = (double)(f - s)/CLOCKS_PER_SEC;
	  if( iter > 1)
		  CGTime += duration;
  }
  printf("cPU and gPU time: %.3f seconds\n\n", CGTime);

  double CRate,GRate;
  CRate = CTime/CGTime;
  GRate = GTime/CGTime;

  printf("相对gPU加速比：%lf\n",GRate);
  printf("相对cPU加速比：%lf\n",CRate);

  free(h_buf);
}

void main()
{
	int n;
	freopen("cuda-result1.out","w",stdout);
	for(n = 10000; n<=15000; n+=100)
		mymain(n);
	fclose(stdout);
}