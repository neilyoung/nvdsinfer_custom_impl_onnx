/*
 * Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */


#include <cstring>
#include <iostream>
#include "nvdsinfer_custom_impl.h"

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define CLIP(a,min,max) (MAX(MIN(a, max), min))

/* This is a sample bounding box parsing function for the sample SSD UFF
 * detector model provided with the TensorRT samples. */

extern "C"
bool NvDsInferParseCustomONNX (std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
        NvDsInferNetworkInfo  const &networkInfo,
        NvDsInferParseDetectionParams const &detectionParams,
        std::vector<NvDsInferObjectDetectionInfo> &objectList);

/* C-linkage to prevent name-mangling */
extern "C"
bool NvDsInferParseCustomONNX (std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
        NvDsInferNetworkInfo  const &networkInfo,
        NvDsInferParseDetectionParams const &detectionParams,
        std::vector<NvDsInferObjectDetectionInfo> &objectList)
{

  static int bboxLayerIndex = -1;
  static int scoresLayerIndex = -1;
  static NvDsInferDimsCHW bboxLayerDims;

  /* Find the bbox layer */
  if (bboxLayerIndex == -1) {
    for (unsigned int i = 0; i < outputLayersInfo.size(); i++) {
      if (strcmp(outputLayersInfo[i].layerName, "boxes") == 0) {
        bboxLayerIndex = i;
        getDimsCHWFromDims(bboxLayerDims, outputLayersInfo[bboxLayerIndex].inferDims);
        break;
      }
    }
    if (bboxLayerIndex == -1) {
      std::cerr << "Could not find bbox layer buffer while parsing" << std::endl;
      return false;
    }
  }

  /* Find the scores layer */
  if (scoresLayerIndex == -1) {
    for (unsigned int i = 0; i < outputLayersInfo.size(); i++) {
      if (strcmp(outputLayersInfo[i].layerName, "scores") == 0) {
        scoresLayerIndex = i;
        break;
      }
    }
    if (scoresLayerIndex == -1) {
      std::cerr << "Could not find scores layer buffer while parsing" << std::endl;
      return false;
    }
  }
  uint32_t numBoxes = bboxLayerDims.c;
  uint32_t numCoord = bboxLayerDims.h;

  float *bbox = (float *) outputLayersInfo[bboxLayerIndex].buffer;
  float *conf = (float *) outputLayersInfo[scoresLayerIndex].buffer;
  uint32_t mNumClasses = detectionParams.numClassesConfigured;
  
  for (uint32_t n = 0; n < numBoxes; n++)
  {
    uint32_t maxClass = 0;
    float    maxScore = -1000.0f;

    for (uint32_t m = 1; m < mNumClasses; m++) {
      const float score = conf[n * mNumClasses + m];
      if (score < detectionParams.perClassThreshold[m])
					continue;
      if (score > maxScore) {
					maxScore = score;
					maxClass = m;
			}
    }
    // check if there was a detection
		if (maxClass <= 0)
				continue; 

    // std::cout << " maxScore " << maxScore << " maxClass " << maxClass  << std::endl;

    const float* coord = bbox + n * numCoord;

    NvDsInferObjectDetectionInfo object;
    object.classId = maxClass;
    object.detectionConfidence = maxScore;
    object.left = coord[0] * networkInfo.width;
    object.top = coord[1] * networkInfo.height;
    object.width = coord[2] * networkInfo.width - coord[0] * networkInfo.width;
    object.height = coord[3] * networkInfo.height - coord[1] * networkInfo.height;
    std::cerr << "id: " << object.classId << ", conf: " <<  object.detectionConfidence << ", left: " << object.left << ", top: " << object.top << ", width: " << object.width << ", height: " << object.height << std::endl;
    objectList.push_back(object);
  }
  
   return true;
}

/* Check that the custom function has been defined correctly */
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomONNX);




