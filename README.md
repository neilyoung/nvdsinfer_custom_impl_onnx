# BBox parser lib for Nvidia DeepStream SDK 5.1 and ONNX models

This project has been derived from the `nvdsinfer

- Clone into `/opt/nvidia/deepstream/deepstream-5.1/sources/objectDetector_SSD`

- Run `make`

- Link to the lib from your PGIE config

```bash
output-blob-names: "boxes;scores"
parse-bbox-func-name: "NvDsInferParseCustomONNX"
custom-lib-path: "/path/to/lib/libnvdsinfer_custom_impl_onnx.so"
```

