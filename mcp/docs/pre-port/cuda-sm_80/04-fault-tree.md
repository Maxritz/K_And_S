# Fault Tree
port to `cuda` does not build or load
  +-- toolchain/host rows PASS
port builds but the artifact is not what was intended
  +-- artifact rows PASS (build first)
port builds but results are wrong or a kernel restructured
  +-- capability rows PASS
port builds and is correct but streams slowly
  +-- pcie.rebar