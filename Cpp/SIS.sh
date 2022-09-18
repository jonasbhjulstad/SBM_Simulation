rm ./data/*Bernoulli_SIS_MC_*.csv
cd build
cmake --build . --target Bernoulli_SIS_MC
./Executables/Generate/Bernoulli_SIS_MC
cmake --build . --target Quantile_Response
cmake --build . --target ERR_Response
./Executables/Regression/ERR_Response
./Executables/Regression/Quantile_Response
cd ../
python ./Executables/Plot/Quantile_Plot.py
