extern "C" {
    #include "lua.h"
    #include "lualib.h"
    #include "lauxlib.h"
}

#include "luaT.h"

#include <stdio.h>
#include <assert.h>
#include <stdint.h>
#include <unistd.h>

#include <opencv/cv.h>
#include <opencv/highgui.h>

#include<TH/TH.h>

int warp_affine(lua_State *L)
{
    THFloatTensor *src_ = (THFloatTensor*)luaT_checkudata(L, 1, "torch.FloatTensor");
    THFloatTensor *dst_ = (THFloatTensor*)luaT_checkudata(L, 2, "torch.FloatTensor");
    THFloatTensor *mat_ = (THFloatTensor*)luaT_checkudata(L, 3, "torch.FloatTensor");

    float *src = THFloatTensor_data(src_);
    float *dst = THFloatTensor_data(dst_);
    float *mat = THFloatTensor_data(mat_);

    int src_c = THFloatTensor_size(src_, 0);
    int src_h = THFloatTensor_size(src_, 1);
    int src_w = THFloatTensor_size(src_, 2);
    int dst_c = THFloatTensor_size(dst_, 0);
    int dst_h = THFloatTensor_size(dst_, 1);
    int dst_w = THFloatTensor_size(dst_, 2);
    assert(THFloatTensor_nElement(mat_) >= 6);

    CvMat warp_mat = cvMat(2, 3, CV_32FC1, mat);
	for (int i = 0; i < src_c; i++) {
		CvMat src_mat = cvMat(src_h, src_w, CV_32FC1, src + i * src_h * src_w);
		CvMat dst_mat = cvMat(dst_h, dst_w, CV_32FC1, dst + i * dst_h * dst_w);
		cvWarpAffine(&src_mat, &dst_mat, &warp_mat, CV_INTER_CUBIC + CV_WARP_FILL_OUTLIERS);
	}

    return 0;
}

static const struct luaL_Reg funcs[] = {
	{"warp_affine", warp_affine},
    {NULL, NULL}
};

extern "C" int luaopen_libcv(lua_State *L) {
    luaL_openlib(L, "cv", funcs, 0);
    return 1;
}
