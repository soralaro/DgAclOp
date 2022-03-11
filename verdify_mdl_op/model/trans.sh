export ASCEND_GLOBAL_LOG_LEVEL=1
export ASCEND_SLOG_PRINT_TO_STDOUT=1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/Ascend/fwkacllib/lib64:/usr/local/Ascend/atc/lib64
export PYTHONPATH=$PYTHONPATH:/usr/local/Ascend/fwkacllib/python/site-packages:/usr/local/Ascend/toolkit/python/site-packages:/usr/local/Ascend/atc/python/site-packages:/usr/local/Ascend/pyACL/python/site-packages/acl
export PATH=$PATH:/usr/local/Ascend/fwkacllib/ccec_compiler/bin:/usr/local/Ascend/fwkacllib/bin:/usr/local/Ascend/atc/bin
export ASCEND_AICPU_PATH=/usr/local/Ascend/
export ASCEND_OPP_PATH=/usr/local/Ascend/opp
export TOOLCHAIN_HOME=/usr/local/Ascend/toolkit
atc --singleop=FaceAlign.json --output=FaceAlign --soc_version=Ascend310
