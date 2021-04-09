helpFunction()
{
   echo ""
   echo "Usage: $0 -f parameterF -m parameterM -o parameterO"
   echo -e "\t-f Fixed image path"
   echo -e "\t-m Moved image path to register"
   echo -e "\t-o Output folder where to store results"
   exit 1 # Exit script after printing help
}

while getopts "f:m:o:" opt
do
   case "$opt" in
      f ) parameterF="$OPTARG" ;;
      m ) parameterM="$OPTARG" ;;
      o ) parameterO="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

# Print helpFunction in case parameters are empty
if [ -z "$parameterF" ] || [ -z "$parameterM" ] || [ -z "$parameterO" ]
then
   echo "Some or all of the parameters are empty";
   helpFunction
fi

# Begin script in case all parameters are correct
echo "Running Elastix on the following images"
echo "$parameterF" "as fixed image"
echo "$parameterM" "as image to be registered"
echo "$parameterO" "as output folder where to store results"
echo "$(pwd) as pwd"

LD_LIBRARY_PATH="$(pwd)\\elastix_linux64_v4.8"
export LD_LIBRARY_PATH

.\\elastix-5.0.1-win64\\elastix -f $parameterF -m $parameterM -p $(pwd)/elastix-5.0.1-win64/params/Par0035.SPREAD.MI.af.0.txt -p $(pwd)/elastix-5.0.1-win64/params/Par0035.SPREAD.MI.bs.1.ASGD.txt -out $parameterO
.\\elastix-5.0.1-win64\\transformix -in $parameterM -tp $(pwd)/temp/TransformParameters.1.txt -out $parameterO

# rm $(pwd)/temp/*.txt
# rm $(pwd)/temp/*.log
# rm $(pwd)/temp/result.0.*

echo "DONE Elastix on the following images"
echo "$parameterF" "as fixed image"
echo "$parameterM" "as image to be registered"
echo "$parameterO" "as output folder where to store results"
echo "$(pwd) as pwd"

read -p "press key to continue..."
