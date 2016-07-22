set -x
set -e


export SPHINX_RESIBOTS_THEME="$HOME/git/sphinx_resibots_theme"
rm -rf /tmp/doc_limbo

mkdir /tmp/doc_limbo/
cd /tmp/doc_limbo
git clone git@github.com:resibots/limbo.git
cd limbo/docs
make html
mv doxygen_doc /tmp
git checkout gh-pages
rm -rf doxygen_doc
cp -r /tmp/doxygen_doc .
cd ..
cp -r docs/_build/html/* .
git add .
git commit -m 'automatic update of the doc [ci skip]'
git push origin gh-pages

rm -rf /tmp/doxygen_doc
rm -rf /tmp/doc_limbo


