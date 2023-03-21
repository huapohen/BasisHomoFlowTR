#!/bin/sh
rtdir='/home/data/lwb/code'
expdir='/home/data/lwb/experiments'
src='baseshomo'
dst='BasisHomoFlowTR'
# branch='lwb'
branch='fblr'
# branch='lwb_aigc'
commit_detail=' update test pipeline, resize mode need multiply a scale '

cd ${rtdir}/${src}
git checkout ${branch}
cd ../${dst}
git checkout main
git checkout -b ${branch}
git checkout ${branch}
cd ${rtdir}
