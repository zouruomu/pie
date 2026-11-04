#!/bin/bash
# The opt-level suite's 4 failures, rerun alone on an idle GPU after the final A/B: fix tree, then
# base tree (opt-level-2 dev builds, single-threaded, --nocapture). The suite ran them beside other
# test processes (the joined test's reruns), and the first panics were device-memory ones.
unset CLAUDE_CODE_MESSAGING_TOKEN
out=/root/r2g/evidence/results/tests
until [ -e /root/r2g/evidence/results/final/sweep.done ]; do sleep 15; done
run() {  # name tree target-dir filter...
  n=$1; d=$2; td=$3; shift 3
  cd $d
  f=$out/idle-$n.log
  echo "tree $(git log --oneline -1); CARGO_PROFILE_DEV_OPT_LEVEL=2; alone on the GPU; $*" > $f
  CARGO_TARGET_DIR=$td CARGO_PROFILE_DEV_OPT_LEVEL=2 taskset -c 0-31,64-95 cargo test -p engine-cuda --features cuda \
    --test engine_cuda -- "$@" --test-threads=1 --nocapture >> $f 2>&1
  echo EXIT=$? >> $f
}
S=a_synthesized_region_serves_the_logits_of_its_ops::
J=a_joined_projection_answers_what_its_legs_answer::a_joined_projection_answers_what_its_legs_answer
run synth-fix /root/r2g/pie /root/r2g/target-opt $S
run synth-base /root/src/pie-base /root/r2g/target-opt-base $S
run joined-fix /root/r2g/pie /root/r2g/target-opt --exact $J
run joined-base /root/src/pie-base /root/r2g/target-opt-base --exact $J
echo done > $out/idle.done
