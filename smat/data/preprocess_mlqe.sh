cd mlqe-pe/data/direct-assessments/train
cat *.tar.gz | tar -zxvf - -i
rm *.tar.gz
cd ../test
cat *.tar.gz | tar -zxvf - -i
rm *.tar.gz
cd ../dev
cat *.tar.gz | tar -zxvf - -i
rm *.tar.gz
cd ../../../..
cd mlqe-pe/data/post-editing/train
cat *.tar.gz | tar -zxvf - -i
rm *.tar.gz
cd ../test
cat *.tar.gz | tar -zxvf - -i
rm *.tar.gz
cd ../dev
cat *.tar.gz | tar -zxvf - -i
rm *.tar.gz
cd ../../../..
