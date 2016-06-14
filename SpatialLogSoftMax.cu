#include "luaT.h"
#include "THC.h"


#define MINUS_LOG_THRESHOLD -18.42
#define SPATIALLOGSOFTMAX_THREADS 128

// Parallelization across each feature point.
__global__ void cunn_SpatialLogSoftMax_updateOutput_kernel
(float *output, float *input, int feature_size, int spatial_size, int data_size,
 float constant)
{
  int idx = (threadIdx.x + blockDim.x*blockIdx.x);
  idx = (idx/spatial_size)*feature_size + idx % spatial_size;

  if (idx < data_size) {
    int next_idx = idx + feature_size;
    float logsum = 0.0;
    float max = -2e38;
    // max
    for(int i = idx; i < next_idx; i += spatial_size) {
      if (max < input[i]) max = input[i];
    }

    // logsum
    for(int i = idx; i < next_idx; i += spatial_size) {
		if (!isnan(input[i])) {
		  logsum += __expf(input[i]-max);
	  	}
    }
		logsum += constant;
    logsum = __logf(logsum) + max;

    // logsoftmax
    for(int i = idx; i < next_idx; i += spatial_size){
      output[i] = input[i] - logsum;
    }
  }
}


__global__ void cunn_SpatialLogSoftMax_updateGradInput_kernel(float *gradInput, float *output, float *gradOutput, int feature_size, int spatial_size, int data_size)
{
  int idx = (threadIdx.x + blockDim.x*blockIdx.x);
  idx = (idx/spatial_size)*feature_size + idx % spatial_size;

  if (idx < data_size) {
    int next_idx = idx + feature_size;
    float gradSum = 0.0;
    // Compute the sum of gradients
    for(int i = idx; i < next_idx; i += spatial_size){
      gradSum += gradOutput[i];
    }
    // Compute the new gradient
    for(int i = idx; i < next_idx; i += spatial_size){
      gradInput[i] = gradOutput[i] - __expf(output[i])*gradSum;
    }
  }
}

static int cunn_SpatialLogSoftMax_updateOutput(lua_State *L)
{
	THCState *state = getCutorchState(L);

	float constant = 0;
	if (luaT_getfieldcheckboolean(L, 1, "constant_present")) {
		constant = expf(luaT_getfieldchecknumber(L, 1, "constant"));
	}
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
//  THCudaTensor *output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");
  int nframe = 0, dim = 0, height = 1, width = 1;

  if (input->nDimension == 1) {
    nframe = 1;
    dim = input->size[0];
  } else if(input->nDimension == 2) {
    nframe = input->size[0];
    dim = input->size[1];
  } else if (input->nDimension == 3) {
    nframe = 1;
    dim = input->size[0];
    width = input->size[1];
    height = input->size[2];
  } else if (input->nDimension == 4) {
    nframe = input->size[0];
    dim = input->size[1];
    width = input->size[2];
    height = input->size[3];
  } else {
    THArgCheck(0, 2, "4d input maximum expected");
  }

  // Get input and output
  input = THCudaTensor_newContiguous(state, input);
//  THCudaTensor_resizeAs(output, input);

  int spatial_size = width*height;
  int feature_size = dim*spatial_size;
  int data_size = feature_size*nframe;
  int nthreads = spatial_size*nframe;
  int nblocks = nthreads/SPATIALLOGSOFTMAX_THREADS;
  if (nthreads % SPATIALLOGSOFTMAX_THREADS != 0){
    nblocks = nblocks + 1;
  }

  dim3 blocks(nblocks,1,1);
  dim3 threads(SPATIALLOGSOFTMAX_THREADS,1,1);

  cunn_SpatialLogSoftMax_updateOutput_kernel<<<blocks,threads>>>
		(THCudaTensor_data(state, input), THCudaTensor_data(state, input),
		 feature_size, spatial_size, data_size, constant);

  cudaError errcode = cudaGetLastError();
  if(errcode != cudaSuccess)
    THError(cudaGetErrorString(errcode));

  THCudaTensor_free(state, input);
  return 1;
}

static int cunn_SpatialLogSoftMax_updateGradInput(lua_State *L)
{
	THCState *state = getCutorchState(L);
  THCudaTensor *gradOutput = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
  THCudaTensor *output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");
//  THCudaTensor *gradInput = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");
  int nframe = 0, dim = 0, height = 1, width = 1;
  
  if (output->nDimension == 1){
    nframe = 1;
    dim = output->size[0];
  } else if (output->nDimension == 2){
    nframe = output->size[0];
    dim = output->size[1];
  } else if (output->nDimension == 3){
    nframe = 1;
    dim = output->size[0];
    width = output->size[1];
    height = output->size[2];
  } else if (output->nDimension == 4){
    nframe = output->size[0];
    dim = output->size[1];
    width = output->size[2];
    height = output->size[3];
  } else {
    THError("4d output maximum expected");
  }

  //Get the data
  output = THCudaTensor_newContiguous(state, output);
  gradOutput = THCudaTensor_newContiguous(state, gradOutput);
//  THCudaTensor_resizeAs(state, gradInput, output);

  int spatial_size = width*height;
  int feature_size = dim*spatial_size;
  int data_size = feature_size*nframe;
  int nthreads = spatial_size*nframe;
  int nblocks = nthreads/SPATIALLOGSOFTMAX_THREADS;
  if (nthreads % SPATIALLOGSOFTMAX_THREADS != 0){
    nblocks = nblocks + 1;
  }
  
  dim3 blocks(nblocks,1,1);
  dim3 threads(SPATIALLOGSOFTMAX_THREADS,1,1);

  cunn_SpatialLogSoftMax_updateGradInput_kernel<<<blocks,threads>>>(THCudaTensor_data(state, gradOutput), 
								    THCudaTensor_data(state, output), 
								    THCudaTensor_data(state, gradOutput),
								    feature_size, spatial_size, data_size);

  cudaError errcode = cudaGetLastError();
  if(errcode != cudaSuccess)
    THError(cudaGetErrorString(errcode));

  THCudaTensor_free(state, gradOutput);
  THCudaTensor_free(state, output);
  return 1;
}

static const struct luaL_Reg cunn_SpatialLogSoftMax__ [] = {
  {"SpatialLogSoftMax_updateOutput", cunn_SpatialLogSoftMax_updateOutput},
  {"SpatialLogSoftMax_updateGradInput", cunn_SpatialLogSoftMax_updateGradInput},
  {NULL, NULL}
};

void cunn_SpatialLogSoftMax_init(lua_State *L)
{
	luaL_openlib(L, "nn", cunn_SpatialLogSoftMax__, 0);
}
