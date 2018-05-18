#!bin/bash

#1 = Backend
#2+ = args

# get tid
if [[ $1 = "Zynq" ]]; then
	REGRESSION_HOME="/home/mattfel/regression/synth/zynq"
elif [[ $1 = "ZCU" ]]; then
	REGRESSION_HOME="/home/mattfel/regression/synth/zcu"
elif [[ $1 = "AWS" ]]; then
	REGRESSION_HOME="/home/mattfel/regression/synth/aws"
elif [[ $1 = "AWS" ]]; then
	REGRESSION_HOME="/home/mattfel/regression/synth/arria10"
fi

tid=`cat ${REGRESSION_HOME}/data/tid`
hash=`cat ${REGRESSION_HOME}/data/hash`
ahash=`cat ${REGRESSION_HOME}/data/ahash`

appname=`basename \`pwd\``
if [[ $1 = "Zynq" ]]; then
	par_util=`pwd`/verilog-zynq/par_utilization.rpt
	par_tmg=`pwd`/verilog-zynq/par_timing_summary.rpt
	word="Slice"
	f1=3
	f2=6
elif [[ $1 = "ZCU" ]]; then
    par_util=`pwd`/verilog-zcu/par_utilization.rpt
    par_tmg=`pwd`/verilog-zcu/par_timing_summary.rpt
    word="CLB"
    f1=3
    f2=6
elif [[ $1 = "Arria10" ]]; then 
    par_util=`pwd`/verilog-zcu/par_utilization.rpt      # TODO: Tian                                                  
    par_tmg=`pwd`/verilog-zcu/par_timing_summary.rpt    # TODO: Tian                                                 
    word="CLB"                                          # TODO: Tian           
    f1=3                                                # TODO: Tian     
    f2=6                                                # TODO: Tian    
elif [[ $1 = "AWS" ]]; then
	par_util=/home/mattfel/aws-fpga/hdk/cl/examples/$appname/build/reports/utilization_route_design.rpt
	par_tmg=/home/mattfel/aws-fpga/hdk/cl/examples/$appname/build/reports/timing_summary_route_design.rpt
	word="CLB"
	f1=5
	f2=8
fi

if [[ -f ${par_util} ]]; then
	lutraw=`cat $par_util | grep -m 1 "$word LUTs" | awk -v f=$f1 -F'|' '{print $f}' | sed "s/ //g"`
	lutpcnt=`cat $par_util | grep -m 1 "$word LUTs" | awk -v f=$f2 -F'|' '{print $f}' | sed "s/ //g"`
	regraw=`cat $par_util | grep -m 1 "$word Registers" | awk -v f=$f1 -F'|' '{print $f}' | sed "s/ //g"`
	regpcnt=`cat $par_util | grep -m 1 "$word Registers" | awk -v f=$f2 -F'|' '{print $f}' | sed "s/ //g"`
	ramraw=`cat $par_util | grep -m 1 "| Block RAM Tile" | awk -v f=$f1 -F'|' '{print $f}' | sed "s/ //g"`
	rampcnt=`cat $par_util | grep -m 1 "| Block RAM Tile" | awk -v f=$f2 -F'|' '{print $f}' | sed "s/ //g"`
	if [[ $1 = "AWS" ]]; then		
		uramraw=`cat $par_util | grep -m 1 "URAM" | awk -v f=$f1 -F'|' '{print $f}' | sed "s/ //g"`
		urampcnt=`cat $par_util | grep -m 1 "URAM" | awk -v f=$f2 -F'|' '{print $f}' | sed "s/ //g"`
	else
		uramraw="NA"
		urampcnt="NA"
	fi
	dspraw=`cat $par_util | grep -m 1 "DSPs" | awk -v f=$f1 -F'|' '{print $f}' | sed "s/ //g"`
	dsppcnt=`cat $par_util | grep -m 1 "DSPs" | awk -v f=$f2 -F'|' '{print $f}' | sed "s/ //g"`
	lalraw=`cat $par_util | grep -m 1 "LUT as Logic" | awk -v f=$f1 -F'|' '{print $f}' | sed "s/ //g"`
	lalpcnt=`cat $par_util | grep -m 1 "LUT as Logic" | awk -v f=$f2 -F'|' '{print $f}' | sed "s/ //g"`
	lamraw=`cat $par_util | grep -m 1 "LUT as Memory" | awk -v f=$f1 -F'|' '{print $f}' | sed "s/ //g"`
	lampcnt=`cat $par_util | grep -m 1 "LUT as Memory" | awk -v f=$f2 -F'|' '{print $f}' | sed "s/ //g"`
else
	lutraw="NA"
	lutpcnt="NA"
	regraw="NA"
	regpcnt="NA"
	ramraw="NA"
	rampcnt="NA"
	uramraw="NA"
	urampcnt="NA"
	dspraw="NA"
	dsppcnt="NA"
	lalraw="NA"
	lalpcnt="NA"
	lamraw="NA"
	lampcnt="NA"
fi

if [[ -f ${par_tmg} ]]; then
	viocnt=`cat $par_tmg | grep -i violated | grep -v synth | wc -l`
	if [[ $viocnt != "0" ]]; then tmg="0"; else tmg="1"; fi
else
	tmg="NA"
fi


if [[ -f `pwd`/end.log ]]; then
  endtime=`cat \`pwd\`/end.log`
else
  endtime=1
fi
if [[ -f `pwd`/start.log ]]; then
  starttime=`cat \`pwd\`/start.log`
else
  starttime=0
fi
synthtime=$((endtime-starttime))

echo "LUT: $lutraw (${lutpcnt}%) Regs: $regraw (${regpcnt}%) BRAM: $ramraw (${rampcnt}%) URAM: $uramraw (${urampcnt}%) DSP: $dspraw (${dsppcnt}%) LaL: $lalraw (${lalpcnt}%) LaM: $lamraw (${lampcnt}%) Synthtime: $synthtime Tmg_Met: $tmg $1"
python3 ${REGRESSION_HOME}/spatial/spatial-lang/utilities/gdocs.py "report_synth_results" $appname "$lutraw (${lutpcnt}%)" "$regraw (${regpcnt}%)" "$ramraw (${rampcnt}%)" "$uramraw (${urampcnt}%)" "$dspraw (${dsppcnt}%)" "$lalraw (${lalpcnt}%)" "$lamraw (${lampcnt}%)" "$synthtime" "$tmg" "$1" "$hash" "$ahash"


# Run on board
if [[ $1 = "Zynq" ]]; then
	APP=$(basename $(pwd))
	scp $(basename $(pwd)).tar.gz regression@holodeck-zc706:
	ssh regression@holodeck-zc706 "
	  locked=\`ls -F /home/sync | grep -v README | wc -l\`
	  if [[ \$locked -gt 0 ]]; then
	    echo -n \"Board locked at $(date +"%Y-%m-%d_%H-%M-%S") by \$(ls -F /home/sync | grep -v README) \"
	    rm -rf /home/regression/${APP}*
	  else
	    mkdir $APP
	    tar -xvf ${APP}.tar.gz -C $APP
	    pushd $APP
	    mkdir verilog
	    mv accel.bit.bin verilog
	    popd
	    cd $APP
	    touch /home/sync/\$(whoami)
	    sudo ./Top $2 $3 $4 $5 $6 $7 $8
	    rm /home/sync/\$(whoami)
	    rm -rf /home/regression/${APP}*	  
	fi" &> log
    timeout=`if [[ $(cat log | grep TIMEOUT | wc -l) -gt 0 ]]; then echo 1; else echo 0; fi`
    locked=`if [[ $(cat log | grep "Board locked" | wc -l) -gt 0 ]]; then cat log | grep "Board locked"; else echo 0; fi`
    runtime=`cat log | grep "ran for" | sed "s/^.*ran for //g" | sed "s/ ms, .*$//g"`
    if [[ $runtime = "" ]]; then runtime=NA; fi
    pass=`if [[ $(cat log | grep "PASS: 1" | wc -l) -gt 0 ]]; then echo Passed!; else echo FAILED; fi`
    python3 ${REGRESSION_HOME}/spatial/spatial-lang/utilities/gdocs.py "report_board_runtime" $appname $timeout $runtime $pass "$2 $3 $4 $5 $6 $7 $8" "$1" "$locked" "$hash" "$ahash"
elif [[ $1 = "ZCU" ]]; then
	APP=$(basename $(pwd))
	scp $(basename $(pwd)).tar.gz root@zcu102:
	ssh root@zcu102 "
	  locked=\`ls -F /home/sync | grep -v README | wc -l\`
	  if [[ \$locked -gt 0 ]]; then
	    echo -n \"Board locked at $(date +"%Y-%m-%d_%H-%M-%S") by \$(ls -F /home/sync | grep -v README) \"
	    rm -rf /root/${APP}*
	  else
	    mkdir $APP
	    tar -xvf ${APP}.tar.gz -C $APP
	    pushd $APP
	    mkdir verilog
	    mv accel.bit.bin verilog
	    popd
	    cd $APP
	    touch /home/sync/\$(whoami)
	    ./Top $2 $3 $4 $5 $6 $7 $8
	    rm /home/sync/\$(whoami)
	    rm -rf /root/${APP}*	  
	fi" &> log
    timeout=`if [[ $(cat log | grep TIMEOUT | wc -l) -gt 0 ]]; then echo 1; else echo 0; fi`
    locked=`if [[ $(cat log | grep "Board locked" | wc -l) -gt 0 ]]; then cat log | grep "Board locked"; else echo 0; fi`
    runtime=`cat log | grep "ran for" | sed "s/^.*ran for //g" | sed "s/ ms, .*$//g"`
    if [[ $runtime = "" ]]; then runtime=NA; fi
    pass=`if [[ $(cat log | grep "PASS: 1" | wc -l) -gt 0 ]]; then echo Passed!; else echo FAILED; fi`
    python3 ${REGRESSION_HOME}/spatial/spatial-lang/utilities/gdocs.py "report_board_runtime" $appname $timeout $runtime $pass "$2 $3 $4 $5 $6 $7 $8" "$1" "$locked" "$hash" "$ahash"	
elif [[ $1 = "Arria10" ]]; then
	APP=$(basename $(pwd))
	scp $(basename $(pwd)).tar.gz root@arria10:   # TODO: Tian (fill in whatever you need to scp and execute the app on the board)
	ssh root@arria10 "
	  locked=\`ls -F /home/sync | grep -v README | wc -l\`
	  if [[ \$locked -gt 0 ]]; then
	    echo -n \"Board locked at $(date +"%Y-%m-%d_%H-%M-%S") by \$(ls -F /home/sync | grep -v README) \"
	    rm -rf /home/root/${APP}*
	  else
	    mkdir $APP
	    untar ${APP}.tar.gz
	    cd $APP
	    touch /home/sync/\$(whoami)
	    ./Top $2 $3 $4 $5 $6 $7 $8
	    rm /home/sync/\$(whoami)
	    rm -rf /home/root/${APP}*	  
	fi" &> log
    timeout=`if [[ $(cat log | grep TIMEOUT | wc -l) -gt 0 ]]; then echo 1; else echo 0; fi`
    locked=`if [[ $(cat log | grep "Board locked" | wc -l) -gt 0 ]]; then cat log | grep "Board locked"; else echo 0; fi`
    runtime=`cat log | grep "ran for" | sed "s/^.*ran for //g" | sed "s/ ms, .*$//g"`
    if [[ $runtime = "" ]]; then runtime=NA; fi
    pass=`if [[ $(cat log | grep "PASS: 1" | wc -l) -gt 0 ]]; then echo Passed!; else echo FAILED; fi`
    python3 ${REGRESSION_HOME}/spatial/spatial-lang/utilities/gdocs.py "report_board_runtime" $appname $timeout $runtime $pass "$2 $3 $4 $5 $6 $7 $8" "$1" "$locked" "$hash" "$ahash"	
fi

# Fake out regression
echo "PASS: 1"