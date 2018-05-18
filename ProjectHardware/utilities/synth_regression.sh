#!/bin/bash

if [[ $1 = "zynq" ]]; then
	export PIR_HOME=${REGRESSION_HOME}
	export CLOCK_FREQ_MHZ=125
	# Prep the spreadsheet
	cd ${REGRESSION_HOME}
	tid=`python3 ${REGRESSION_HOME}/next-spatial/spatial-lang/utilities/gdocs.py "prepare_sheet" "$hash" "$apphash" "$timestamp" "Zynq"`
	echo $tid > ${REGRESSION_HOME}/data/tid
	echo $hash > ${REGRESSION_HOME}/data/hash
	echo $apphash > ${REGRESSION_HOME}/data/ahash
elif [[ $1 = "zcu" ]]; then
	export PIR_HOME=${REGRESSION_HOME}
	export CLOCK_FREQ_MHZ=125
	# Prep the spreadsheet
	cd ${REGRESSION_HOME}
	tid=`python3 ${REGRESSION_HOME}/next-spatial/spatial-lang/utilities/gdocs.py "prepare_sheet" "$hash" "$apphash" "$timestamp" "ZCU"`
	echo $tid > ${REGRESSION_HOME}/data/tid
	echo $hash > ${REGRESSION_HOME}/data/hash
	echo $apphash > ${REGRESSION_HOME}/data/ahash
elif [[ $1 = "arria10" ]]; then
	export PIR_HOME=${REGRESSION_HOME}
	export CLOCK_FREQ_MHZ=125
	# Prep the spreadsheet
	cd ${REGRESSION_HOME}
	tid=`python3 ${REGRESSION_HOME}/next-spatial/spatial-lang/utilities/gdocs.py "prepare_sheet" "$hash" "$apphash" "$timestamp" "Arria10"`
	echo $tid > ${REGRESSION_HOME}/data/tid
	echo $hash > ${REGRESSION_HOME}/data/hash
	echo $apphash > ${REGRESSION_HOME}/data/ahash
elif [[ $1 = "aws" ]]; then
	export PIR_HOME=${REGRESSION_HOME}
	export CLOCK_FREQ_MHZ=250
	# Prep the spreadsheet
	cd ${REGRESSION_HOME}
	tid=`python3 ${REGRESSION_HOME}/next-spatial/spatial-lang/utilities/gdocs.py "prepare_sheet" "$hash" "$apphash" "$timestamp" "AWS"`
	echo $tid > ${REGRESSION_HOME}/data/tid
	echo $hash > ${REGRESSION_HOME}/data/hash
	echo $apphash > ${REGRESSION_HOME}/data/ahash
fi

export PATH=/usr/bin:/local/ssd/home/mattfel/aws-fpga/hdk/common/scripts:/opt/Xilinx/SDx/2017.1/Vivado/bin:/opt/Xilinx/SDx/2017.1/SDK/bin:/opt/Xilinx/Vivado/2017.1/bin:/opt/Xilinx/SDK/2017.1/bin:$PATH
export LM_LICENSE_FILE=1717@cadlic0.stanford.edu:7195@cadlic0.stanford.edu:7193@cadlic0.stanford.edu:/opt/Xilinx/awsF1.lic:27000@cadlic0.stanford.edu:$LM_LICENSE_FILE
export VCS_HOME=/cad/synopsys/vcs/K-2015.09-SP2-7
export QVER=17.1
alias qsys=/opt/intelFPGA_pro/$QVER/qsys/bin/qsys-edit
export PATH=/usr/bin:$VCS_HOME/amd64/bin:/opt/intelFPGA_pro/$QVER/quartus/bin:$PATH
export QSYS_ROOTDIR=/opt/intelFPGA_pro/$QVER/qsys/bin/
export PATH=/opt/intelFPGA_pro/$QVER/quartus/sopc_builder/bin:$PATH
export ALTERAOCLSDKROOT=/opt/intelFPGA_pro/$QVER/hld
export LM_LICENSE_FILE=/opt/intelFPGA_pro/licenses/arria10-license.dat:$LM_LICENSE_FILE
export ALTERAD_LICENSE_FILE=/opt/intelFPGA_pro/licenses/arria10-license.dat
export USING_THIS_QUARTUS=`which quartus`

# Current hash matches previous hash, skip test
if [[ $tid = "-1" ]]; then
	sleep 3600 # Wait an hour
	rm -rf ${REGRESSION_HOME}/next-spatial
else 
	cd ${REGRESSION_HOME}
	rm -rf ${REGRESSION_HOME}/last-spatial
	mv ${REGRESSION_HOME}/spatial ${REGRESSION_HOME}/last-spatial
	mv ${REGRESSION_HOME}/next-spatial ${REGRESSION_HOME}/spatial

	echo "Moving to ${REGRESSION_HOME}/spatial/spatial-lang"
	cd ${REGRESSION_HOME}/spatial/spatial-lang

	if [[ $1 = "zynq" ]]; then
		bin/regression 2 nobranch Zynq Dense Sparse
	elif [[ $1 = "zcu" ]]; then
		bin/regression 2 nobranch ZCU Dense Sparse
	elif [[ $1 = "arria10" ]]; then
		bin/regression 2 nobranch Arria10 Dense Sparse
	elif [[ $1 = "aws" ]]; then
		bin/regression 2 nobranch AWS Dense Sparse
	fi
fi
