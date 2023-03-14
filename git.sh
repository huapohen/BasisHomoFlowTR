#!/bin/sh
rtdir='/home/data/lwb/code'
expdir='/home/data/lwb/experiments'
src='baseshomo'
dst='BasisHomoFlowTR'
# branch='lwb'
# branch='fblr'
branch='lwb_aigc'
commit_detail=' AIGC: pix2pix-stn '

cd ${rtdir}/${src}
rm -f ./experiments/${src}
git checkout ${branch}
cd ../${dst}
git checkout main
git checkout -b ${branch}
git checkout ${branch}
cd ${rtdir}
mv ${dst} ${dst}_bp
cp -r ${rtdir}/${src} ${rtdir}/${dst}
cd ${dst}
rm -rf .git README.md LICENSE
cd ..
cp -r ${dst}_bp/.git ${dst}/
mv ${dst}_bp/README.md ${dst}/README.md
mv ${dst}_bp/LICENSE ${dst}/LICENSE
rm -rf ${dst}_bp

for n in ${dst}
# for n in ${src} ${dst}
do
    cd ${rtdir}/${n}
    git add .
    git status
    git commit -m "${commit_detail}"
    # git push
    git push origin -u ${branch}
    git push
done

# git log
# git reset --hard !@#$%^&*()
# git push origin fblr -f

cd ${rtdir}/${dst}
git branch
cd ${rtdir}/${src}
# ln -s ${expdir}/${src} ./experiments/${src}
git branch