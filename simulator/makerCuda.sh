str=`echo $LD_LIBRARY_PATH | grep '../lib/impl/cuda'`
 if [ -z "$str" ] ; then source ./export_libs.sh; fi
make -f MakefileCuda clean && make -f MakefileCuda && ./nnsimulatorcuda $1
