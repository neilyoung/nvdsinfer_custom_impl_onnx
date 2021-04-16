# BBox parser lib for Nvidia DeepStream SDK 5.1 and ONNX models

This project has been derived from the `nvdsinfer_custom_impl_ssd` sample project from the Nvidia DeepStream SDK 5.1 in order to work with re-trained ONNX models. Special thanks to `dusty_nv` for the help to adapt it to the environment of an ONNX model.

- Clone this repo into `/opt/nvidia/deepstream/deepstream-5.1/sources/objectDetector_SSD` and change into the cloned project directory

- Run `make`

- Link to the lib from your PGIE config:

```bash
output-blob-names: "boxes;scores"
parse-bbox-func-name: "NvDsInferParseCustomONNX"
custom-lib-path: "/path/to/lib/libnvdsinfer_custom_impl_onnx.so"
```

- No warranty
