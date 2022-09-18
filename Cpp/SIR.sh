rm ./data/*Bernoulli_SIR_MC_*.csv
cd build
cmake --build . --target Bernoulli_SIR_MC
./Executables/Generate/Bernoulli_SIR_MC SIR
cmake --build . --target Quantile_Response
cmake --build . --target ERR_Response
./Executables/Regression/ERR_Response SIR
./Executables/Regression/Quantile_Response SIR
cd ../
python ./Executables/Plot/Quantile_Plot.py "SIR"
