get_ipython().run_line_magic('env', 'TOOLKIT_BUCKET=s3://jsimon-public-us/')
get_ipython().run_line_magic('env', 'TOOLKIT_NAME=toolkit.tgz')
get_ipython().run_line_magic('env', 'TOOLKIT_DIR=l_deeplearning_deploymenttoolkit_2017.1.0.5852')

get_ipython().run_line_magic('env', 'MODEL_BUCKET=s3://jsimon-public-us/')
get_ipython().run_line_magic('env', 'MODEL_NAME=Inception-BN')

get_ipython().run_line_magic('env', 'OPT_DIR=/opt/intel/deeplearning_deploymenttoolkit/deployment_tools/model_optimizer/model_optimizer_mxnet')
get_ipython().run_line_magic('env', 'OPT_PRECISION=FP16')
get_ipython().run_line_magic('env', 'OPT_FUSE=YES')

get_ipython().run_cell_magic('bash', '', '\necho "*** Downloading toolkit"\naws s3 cp $TOOLKIT_BUCKET$TOOLKIT_NAME .\necho "*** Installing toolkit"\ntar xfz $TOOLKIT_NAME\ncd $TOOLKIT_DIR\nchmod 755 install.sh\nsudo ./install.sh -s silent.cfg \necho "*** Done"')

get_ipython().run_cell_magic('bash', '', '\n#conda create -n intel_toolkit -y\npython -m ipykernel install --user --name intel_toolkit --display-name "intel_toolkit"\n\nsource activate intel_toolkit\ncd $OPT_DIR\npip install -r requirements.txt ')

get_ipython().run_cell_magic('bash', '', '\necho "*** Downloading model"\naws s3 cp $MODEL_BUCKET$MODEL_NAME"-symbol.json" .\naws s3 cp $MODEL_BUCKET$MODEL_NAME"-0000.params" .\necho "*** Done"')

get_ipython().run_cell_magic('bash', '', '\nsource activate intel_toolkit\necho "*** Converting model"\npython $OPT_DIR/mo_mxnet_converter.py --models-dir . --output-dir . --model-name $MODEL_NAME --precision $OPT_PRECISION --fuse $OPT_FUSE\nls mxnet_$MODEL_NAME*\necho "*** Done"')



