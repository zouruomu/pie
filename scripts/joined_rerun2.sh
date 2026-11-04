#!/bin/bash
# a_joined_projection_answers_what_its_legs_answer alone again, fix then base one after the other
# (joined_rerun.sh ran them side by side with three suite processes, and rows ran out of device
# memory). Beside it only the opt-level engine-cuda suite runs.
unset CLAUDE_CODE_MESSAGING_TOKEN
out=/root/r2g/evidence/results/tests
t=a_joined_projection_answers_what_its_legs_answer::a_joined_projection_answers_what_its_legs_answer
run() {  # tree target-dir log
  cd $1
  echo "tree $(git log --oneline -1); CARGO_PROFILE_DEV_OPT_LEVEL=2; $t alone" > $3
  CARGO_TARGET_DIR=$2 CARGO_PROFILE_DEV_OPT_LEVEL=2 taskset -c 0-31,64-95 cargo test -p engine-cuda --features cuda \
    --test engine_cuda -- --exact $t --test-threads=1 --nocapture >> $3 2>&1
  echo EXIT=$? >> $3
}
run /root/r2g/pie /root/r2g/target-opt $out/joined-fix2.log
run /root/src/pie-base /root/r2g/target-opt-base $out/joined-base2.log
echo done > $out/joined2.done
