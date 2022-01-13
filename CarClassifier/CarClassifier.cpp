#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <random>
#include <cmath>
#include "KNN.h"
#include <chrono>

double PERCENTAGE_FOR_TRAINING = 0.7; //by testing we decided that 80% and k=5 is the best combination and it is used when we input custom patients
size_t K = 3;

//already a number
const size_t MODEL_POSITION = 0;
const size_t YEAR_POSITION = 1;
const size_t PRICE_POSITION = 2;
const size_t MILEAGE_POSITION = 4;
const size_t TAX_POSITION = 6;
const size_t MPG_POSITION = 7;
const size_t ENGINE_SIZE_POSITION = 8;

//needs to be processed to a number
const size_t TRANSMISSION_POSITION = 3;

//functions
//splits data into parts for training and parts for testing
void trainTestSplit(std::vector<std::vector<double>>& X, std::vector<std::string>& y,
    std::vector<std::vector<double>>& XTrain, std::vector<std::vector<double>>& XTest,
    std::vector<std::string>& yTrain, std::vector<std::string>& yTest);

//normalizes columns (divides each element of the vector by the lenght of the vector this way the new vector has length = 1)
void normalize(std::vector<std::vector<double>>& values, size_t col);

//collects the raw data from the file
std::vector<std::vector<std::string>> readFromFile(std::ifstream& file);

//separates the classes names from the raw data
std::vector<std::string> getCLasses(std::vector<std::vector<std::string>> rawData);

//pre-processes the raw data so it can be used for kNN
std::vector<std::vector<double>> preProcessData(std::vector<std::vector<std::string>> rawData);

int main()
{
    auto frame_start = std::chrono::high_resolution_clock::now(); //starts a timer

    //open csv file containing the data set
    std::ifstream dataSetFile("hyundi.csv");

    if (!dataSetFile.is_open())
    {
        std::cerr << "File did not open.\n";
        return 0;
    }

    //get the data and pre-process it
    std::vector<std::vector<std::string>> rawData = readFromFile(dataSetFile);
    dataSetFile.close(); //close the file as we no longer need it

    std::vector<std::string> classes = getCLasses(rawData);

    std::vector<std::vector<double>> data = preProcessData(rawData);

    //normalize the data
    const size_t PROCESSED_YEAR_POSITION = YEAR_POSITION - 1;
    const size_t PROCESSED_PRICE_POSITION = PRICE_POSITION - 1;
    const size_t PROCESSED_MILEAGE_POSITION = MILEAGE_POSITION + 1;
    const size_t PROCESSED_TAX_POSITION = TAX_POSITION + 4;
    const size_t PROCESSED_MPG_POSITION = MPG_POSITION + 4;
    const size_t PROCESSED_ENGINE_SIZE_POSITION = ENGINE_SIZE_POSITION + 4;

    normalize(data, PROCESSED_YEAR_POSITION);
    normalize(data, PROCESSED_PRICE_POSITION);
    normalize(data, PROCESSED_MILEAGE_POSITION);
    normalize(data, PROCESSED_TAX_POSITION);
    normalize(data, PROCESSED_MPG_POSITION);
    normalize(data, PROCESSED_ENGINE_SIZE_POSITION);

    //split the data into parts for testing and parts for training
    std::vector<std::vector<double>> XTrain;
    std::vector<std::vector<double>> XTest;

    std::vector<std::string> yTrain;
    std::vector<std::string> yTest;

    trainTestSplit(data, classes, XTrain, XTest, yTrain, yTest);

    //start kNN
    KNN model(K);
    model.fit(XTrain, yTrain);
    std::vector<std::string> yPred = model.predict(XTest);

    size_t correct = 0;
    size_t sizeYPred = yPred.size();
   
    for (size_t i = 0; i < sizeYPred; i++)
    {
        if (yPred[i] == yTest[i])
        {
            correct++;
        }
    }

    std::cout << correct << " out of " << sizeYPred << " are correct\n";
    std::cout << "Success rate: " << (double(correct) / double(sizeYPred)) * 100 << "%\n";

    auto frame_end = std::chrono::high_resolution_clock::now(); //end of timer

    std::chrono::duration<float> duration = frame_end - frame_start;
    std::cout << "Run time:" << duration.count() << '\n';
}

void trainTestSplit(std::vector<std::vector<double>>& X, std::vector<std::string>& y,
    std::vector<std::vector<double>>& XTrain, std::vector<std::vector<double>>& XTest,
    std::vector<std::string>& yTrain, std::vector<std::string>& yTest)
{
    srand((unsigned)time(NULL));

    size_t size = X.size();
    size_t limit = (size_t)(size * PERCENTAGE_FOR_TRAINING);
    ;
    for (size_t i = 0; i < size; i++) //randomizes the order of the samples from the data
    {
        size_t rndIndex = rand() % size;
        std::swap(X[i], X[rndIndex]);
        std::swap(y[i], y[rndIndex]);
    }

    //separates samples for training
    for (size_t i = 0; i < limit; i++)
    {
        XTrain.push_back(X[i]);
        yTrain.push_back(y[i]);
    }

    //separates samples for testing
    for (size_t i = limit; i < size; i++)
    {
        XTest.push_back(X[i]);
        yTest.push_back(y[i]);
    }
}

void normalize(std::vector<std::vector<double>>& values, size_t col)
{
    double sum = 0;
    size_t size = values.size();
    for (size_t i = 0; i < size; i++)
    {
        sum += pow(values[i][col], 2);
    }

    double magnitude = sqrt(sum);

    for (size_t i = 0; i < size; i++)
    {
        values[i][col] /= magnitude;
    }
}

std::vector<std::vector<std::string>> readFromFile(std::ifstream& file)
{
    std::string str = "";
    std::vector<std::vector<std::string>> rawData; //the data we have before we process it
    std::vector<std::string> line; //used to separate each sample
    bool isFirstLine = true;
    char symbol = '\0';

    while (symbol != -1) //savess the data from the file to rawData
    {
        symbol = file.get();

        if (isFirstLine)
        {
            if (symbol == ',')
            {
                str = "";
                continue;
            }
            else if (symbol == '\n')
            {
                isFirstLine = false;
                str = "";
                continue;
            }
        }
        else
        {
            if (symbol == ',')
            {
                line.push_back(str);
                str = "";
                continue;
            }
            else if (symbol == '\n')
            {
                line.push_back(str);
                str = "";
                rawData.push_back(line);
                while (!line.empty())
                {
                    line.pop_back();
                }
                continue;
            }
        }

        str += symbol;
    }

    return rawData;
}

std::vector<std::string> getCLasses(std::vector<std::vector<std::string>> rawData)
{
    std::vector<std::string> classes; //y or the model of the car

    size_t rowsRawData;
    rowsRawData = rawData.size();

    for (size_t i = 0; i < rowsRawData; i++) //separates classes from the rest of the data
    {
        classes.push_back(rawData[i][MODEL_POSITION]);
    }

    return classes;
}

std::vector<std::vector<double>> preProcessData(std::vector<std::vector<std::string>> rawData)
{
    std::vector<std::vector<double>> data; //the procesed data
    std::vector<double> sample;

    size_t rowsRawData, colsRawData;
    rowsRawData = rawData.size();
    colsRawData = rawData[0].size();

    double num = 0;

    double numberValue = 0;
    for (size_t i = 0; i < rowsRawData; i++) //processes and saves all the samples in an array
    {
        for (size_t j = 1; j < colsRawData; j++)
        {
            if (j == YEAR_POSITION || j == PRICE_POSITION ||
                j == MILEAGE_POSITION || j == TAX_POSITION || j == MPG_POSITION ||
                j == ENGINE_SIZE_POSITION)
            {
                sample.push_back(std::atof(rawData[i][j].c_str()));
            }
            else if (j == TRANSMISSION_POSITION)
            {
                if (rawData[i][j] == "Manual")
                {
                    sample.push_back(1);
                    sample.push_back(0);
                    sample.push_back(0);
                }
                else if (rawData[i][j] == "Automatic")
                {
                    sample.push_back(0);
                    sample.push_back(1);
                    sample.push_back(0);
                }
                else //if Semi-Auto
                {
                    sample.push_back(0);
                    sample.push_back(0);
                    sample.push_back(1);
                }
            }
            else //fuelType column
            {
                if (rawData[i][j] == "Petrol")
                {
                    sample.push_back(1);
                    sample.push_back(0);
                    sample.push_back(0);
                    sample.push_back(0);
                }
                else if (rawData[i][j] == "Diesel")
                {
                    sample.push_back(0);
                    sample.push_back(1);
                    sample.push_back(0);
                    sample.push_back(0);
                }
                else if (rawData[i][j] == "Hybrid")
                {
                    sample.push_back(0);
                    sample.push_back(0);
                    sample.push_back(1);
                    sample.push_back(0);
                }
                else //Other
                {
                    sample.push_back(0);
                    sample.push_back(0);
                    sample.push_back(0);
                    sample.push_back(1);
                }
            }
        }

        data.push_back(sample);
        while (!sample.empty())
        {
            sample.pop_back();
        }
    }

    return data;
}