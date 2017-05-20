#include <cstdlib>
#include <vector>
#include <fstream>

#include "opencv2/core/core.hpp"
#include "opencv2/ml/ml.hpp"

using namespace std;
using namespace cv;
using namespace cv::ml;

int main(int argc, char** argv) {

//Read the data from csv file
Ptr<TrainData> cvml = TrainData::loadFromCSV(string("char_datasetNM1.csv"),0,0);


//Select 90% for the training
cvml->setTrainTestSplitRatio(0.9, true);

Ptr<Boost> boost;

ifstream ifile("./trained_classifierNM1.xml");
if (ifile)
{
	//The file exists, so we don't want to train
	printf("Found trained_boost_char.xml file, remove it if you want to retrain with new data ... \n");
	boost = StatModel::load<Boost>("./trained_classifierNM1.xml");
} else {
	//Train with 100 features
	printf("Training ... \n");
	boost = Boost::create();
	boost->setWeightTrimRate(0.0);
	boost->train(cvml);
}

//Calculate the test and train errors
Mat train_responses, test_responses;
float fl1 = boost->calcError(cvml,false,train_responses);
float fl2 = boost->calcError(cvml,true,test_responses);
printf("Error train %f \n", fl1);
printf("Error test %f \n", fl2);


//Try a char
Mat sample = (Mat_<float>(1,4) << 1.063830,0.083372,0.000000,2.000000);
float prediction = boost->predict( sample, noArray(), 0 );
float votes      = boost->predict( sample, noArray(), DTrees::PREDICT_SUM | StatModel::RAW_OUTPUT );

printf("\n The char sample is predicted as: %f (with number of votes = %f)\n", prediction,votes);
printf(" Class probability (using Logistic Correction) is P(r|character) = %f\n", (float)1-(float)1/(1+exp(-2*votes)));

//Try a NONchar
Mat sample2 = (Mat_<float>(1,4) << 2.000000,0.235702,0.000000,2.000000);
prediction = boost->predict( Mat(sample2), noArray(), 0 );
votes      = boost->predict( Mat(sample2), noArray(), DTrees::PREDICT_SUM | StatModel::RAW_OUTPUT );

printf("\n The non_char sample is predicted as: %f (with number of votes = %f)\n", prediction,votes);
printf(" Class probability (using Logistic Correction) is P(r|character) = %f\n\n", (float)1-(float)1/(1+exp(-2*votes)));

// Save the trained classifier
boost->save(string("./trained_classifierNM1.xml"));

return EXIT_SUCCESS;
}
