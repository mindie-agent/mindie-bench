ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG BASE_IMAGE_ID
ARG PROFILE
COPY manifest.json project_runtime.py /bench/runtime-recipe/
COPY historical/ /bench/runtime-recipe/historical/
COPY ascend_probe.py /bench/ascend_probe.py
RUN python3 -B /bench/runtime-recipe/project_runtime.py \
      --manifest /bench/runtime-recipe/manifest.json \
      --profile "${PROFILE}" --base-image "${BASE_IMAGE_ID}" \
      --runtime-root /vllm-workspace \
      --historical /bench/runtime-recipe/historical \
      --apply --output /bench/runtime-function-projection.json \
    && rm -rf /bench/runtime-recipe
